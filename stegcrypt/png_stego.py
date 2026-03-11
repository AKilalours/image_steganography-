"""
stegcrypt.png_stego
~~~~~~~~~~~~~~~~~~~
Password-keyed randomized LSB steganography for lossless images.

Security model
--------------
* **No plaintext magic bytes** are written to the carrier.  The header is
  embedded at a password-derived position, so an observer without the
  password cannot even locate the header — let alone the payload.
* **Header position** is derived via a second, independent scrypt pass
  (different salt than the payload PRNG seed), preventing correlation.
* **Header fields** are encrypted with a lightweight XOR stream derived
  from the embed key, providing confidentiality for metadata too.
* **Payload positions** are drawn without replacement from a CSPRNG seeded
  by password + embed_salt.  Positions are shuffled after the header region
  so header and payload never overlap.
* **ECC** (majority-vote bit repetition 1..9) reduces bit-flip sensitivity
  at the cost of capacity.
* **Alpha channel and ICC profiles** are preserved so the stego image is
  indistinguishable from the cover in most viewers.

Wire format (all in-image, no side files)
-----------------------------------------
Bits 0 .. header_slots-1  : encrypted header  (password-keyed XOR)
Bits in random positions   : payload bitstream (ECC-encoded)

Header (plaintext before XOR, 32 bytes)
  [0:4]   embed_salt (16 bytes)       — random per embed; seeds payload PRNG
  [16:20] payload_len (uint32 BE)
  [20:24] payload_crc32 (uint32 BE)
  [24:25] flags byte  (bits 0..3 = ecc factor)
  [25:32] zero padding
"""

from __future__ import annotations

import os
import struct
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt

from .util import crc32

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_HDR_BYTES      = 32
_HDR_REPEAT     = 3          # header ECC repetition (fixed, not user-tunable)
_MAX_ECC        = 9
_HDR_SCRYPT_N   = 2 ** 14   # lightweight; used only to derive header XOR mask
_SEED_SCRYPT_N  = 2 ** 15   # payload PRNG seed derivation


# ---------------------------------------------------------------------------
# Lossless format helpers
# ---------------------------------------------------------------------------

_LOSSLESS_SUFFIXES = {".png", ".bmp", ".tif", ".tiff"}


def _output_format(path: Path) -> str:
    suf = path.suffix.lower()
    mapping = {".png": "PNG", ".bmp": "BMP", ".tif": "TIFF", ".tiff": "TIFF"}
    if suf not in mapping:
        raise ValueError(
            f"Output must be a lossless format (.png/.bmp/.tif/.tiff), got {path.suffix!r}."
        )
    return mapping[suf]


def _assert_lossless_input(path: Path) -> None:
    if path.suffix.lower() not in _LOSSLESS_SUFFIXES:
        raise ValueError(
            f"Stego images must be stored losslessly; {path.suffix!r} is not supported. "
            "Re-run encryption and output .png/.bmp/.tif/.tiff."
        )


# ---------------------------------------------------------------------------
# Header data-class
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _StegoHeader:
    embed_salt: bytes   # 16 bytes
    payload_len: int
    payload_crc32: int
    ecc: int


# ---------------------------------------------------------------------------
# KDF helpers
# ---------------------------------------------------------------------------

def _derive_header_mask(password: str, embed_salt: bytes) -> bytes:
    """Derive a 32-byte XOR mask used to encrypt the header bits."""
    kdf = Scrypt(salt=embed_salt + b"\x00", length=_HDR_BYTES,
                 n=_HDR_SCRYPT_N, r=8, p=1)
    return kdf.derive(password.encode("utf-8"))


def _derive_payload_seed(password: str, embed_salt: bytes) -> int:
    """Derive a 64-bit PRNG seed for payload position selection."""
    kdf = Scrypt(salt=embed_salt + b"\x01", length=32,
                 n=_SEED_SCRYPT_N, r=8, p=1)
    raw = kdf.derive(password.encode("utf-8"))
    return int.from_bytes(raw[:8], "big", signed=False)


def _derive_header_offset(password: str, carrier_slots: int) -> int:
    """Derive the starting slot index for header embedding.

    Uses a third KDF pass with a static salt so the offset depends
    only on the password and image size — not on embed_salt — allowing
    the reader to find the header with only the password.
    """
    static_salt = b"hdr_offset_v3"
    kdf = Scrypt(salt=static_salt, length=8, n=_HDR_SCRYPT_N, r=8, p=1)
    raw = kdf.derive(password.encode("utf-8"))
    seed = int.from_bytes(raw, "big", signed=False)
    # Header must fit; reserve last header_slots from consideration
    header_slots_needed = _HDR_BYTES * 8 * _HDR_REPEAT
    max_offset = max(0, carrier_slots - header_slots_needed)
    return int(seed % (max_offset + 1)) if max_offset > 0 else 0


# ---------------------------------------------------------------------------
# Bit manipulation helpers
# ---------------------------------------------------------------------------

def _to_bits(b: bytes) -> np.ndarray:
    return np.unpackbits(np.frombuffer(b, dtype=np.uint8))


def _from_bits(bits: np.ndarray) -> bytes:
    return np.packbits(bits[:len(bits) - len(bits) % 8]).tobytes()


def _ecc_encode(bits: np.ndarray, r: int) -> np.ndarray:
    return np.repeat(bits, r) if r > 1 else bits


def _ecc_decode(bits: np.ndarray, r: int) -> np.ndarray:
    if r <= 1:
        return bits
    if len(bits) % r:
        raise ValueError("ECC bitstream length not divisible by repetition factor.")
    return (bits.reshape(-1, r).sum(axis=1) >= (r // 2 + 1)).astype(np.uint8)


# ---------------------------------------------------------------------------
# Header pack/unpack
# ---------------------------------------------------------------------------

def _pack_header(h: _StegoHeader) -> bytes:
    """Pack header into _HDR_BYTES.

    Layout (32 bytes):
      [0:16]  embed_salt         — stored PLAINTEXT (it is public per-message randomness,
                                   like an IV; the reader needs it to derive the XOR mask
                                   for the sensitive fields that follow)
      [16:20] payload_len XOR mask[0:4]
      [20:24] payload_crc32 XOR mask[4:8]
      [24:25] flags XOR mask[8]
      [25:32] zero padding
    """
    flags = h.ecc & 0x0F
    sensitive = (
        struct.pack(">I", h.payload_len)
        + struct.pack(">I", h.payload_crc32)
        + struct.pack(">B", flags)
        + b"\x00" * (_HDR_BYTES - 16 - 4 - 4 - 1)
    )
    assert len(sensitive) == _HDR_BYTES - 16
    return h.embed_salt + sensitive  # mask is applied at embed time


def _apply_sensitive_mask(sensitive: bytes, mask: bytes) -> bytes:
    """XOR the 16 sensitive bytes with the first 16 bytes of mask."""
    return bytes(a ^ b for a, b in zip(sensitive, mask[:len(sensitive)]))


def _unpack_header(raw: bytes) -> _StegoHeader:
    if len(raw) != _HDR_BYTES:
        raise ValueError("Bad header length.")
    embed_salt = raw[0:16]
    sensitive  = raw[16:]   # already XOR-decrypted by the caller
    payload_len   = struct.unpack(">I", sensitive[0:4])[0]
    payload_crc32 = struct.unpack(">I", sensitive[4:8])[0]
    flags         = sensitive[8]
    ecc = flags & 0x0F
    if ecc < 1 or ecc > _MAX_ECC:
        raise ValueError(
            f"Invalid ECC factor {ecc} in header — wrong password or corrupted image."
        )
    return _StegoHeader(
        embed_salt=embed_salt,
        payload_len=payload_len,
        payload_crc32=payload_crc32,
        ecc=ecc,
    )


# ---------------------------------------------------------------------------
# Capacity query (public)
# ---------------------------------------------------------------------------

def png_capacity_bytes(img: Image.Image, *, ecc: int = 1) -> int:
    """Return usable payload capacity in bytes for *img* at *ecc* factor."""
    w, h = img.size
    total_slots    = w * h * 3
    header_slots   = _HDR_BYTES * 8 * _HDR_REPEAT
    usable_slots   = max(0, total_slots - header_slots)
    return (usable_slots // max(1, ecc)) // 8


# ---------------------------------------------------------------------------
# Embed
# ---------------------------------------------------------------------------

def embed_png(
    in_path: Path,
    out_path: Path,
    payload: bytes,
    *,
    password: str,
    ecc: int = 1,
) -> None:
    """Embed *payload* into a lossless image file.

    The header position, header content (XOR-encrypted), and payload positions
    are all password-derived.  No plaintext magic or detectable marker is
    written to the carrier.
    """
    if ecc < 1 or ecc > _MAX_ECC:
        raise ValueError(f"ecc must be 1–{_MAX_ECC}, got {ecc}.")

    im0  = Image.open(in_path)
    icc  = im0.info.get("icc_profile", None)
    mode = "RGBA" if im0.mode == "RGBA" else "RGB"
    im   = im0.convert(mode)
    arr  = np.array(im, dtype=np.uint8)

    rgb  = arr[:, :, :3].copy()
    flat = rgb.reshape(-1)

    total_slots  = flat.size
    header_slots = _HDR_BYTES * 8 * _HDR_REPEAT

    embed_salt = os.urandom(16)

    # --- Build and XOR-encrypt header ---
    h = _StegoHeader(
        embed_salt=embed_salt,
        payload_len=len(payload),
        payload_crc32=crc32(payload),
        ecc=ecc,
    )
    hdr_plain    = _pack_header(h)                            # embed_salt || sensitive_plain
    hdr_mask     = _derive_header_mask(password, embed_salt)  # 32-byte mask
    # Only XOR the sensitive 16 bytes (bytes 16..31); embed_salt (0..15) stays plaintext
    sensitive_enc = _apply_sensitive_mask(hdr_plain[16:], hdr_mask)
    hdr_enc       = hdr_plain[:16] + sensitive_enc            # 32 bytes total

    hdr_bits     = _to_bits(hdr_enc)
    hdr_bits_rep = _ecc_encode(hdr_bits, _HDR_REPEAT)

    # --- Header offset (password-derived, image-size-aware) ---
    hdr_offset = _derive_header_offset(password, total_slots)
    hdr_end    = hdr_offset + header_slots

    if hdr_end > total_slots:
        raise ValueError(
            "Image too small to embed header at password-derived offset. "
            "Use a larger image."
        )

    # --- Payload bits ---
    payload_bits     = _to_bits(payload)
    payload_bits_enc = _ecc_encode(payload_bits, ecc)
    payload_slots    = len(payload_bits_enc)

    # Available positions for payload: everything EXCEPT the header region
    available_mask = np.ones(total_slots, dtype=bool)
    available_mask[hdr_offset:hdr_end] = False
    available_indices = np.where(available_mask)[0]

    if payload_slots > len(available_indices):
        cap = png_capacity_bytes(im, ecc=ecc)
        raise ValueError(
            f"Image capacity too small.  Need {len(payload)} bytes, "
            f"capacity ≈ {cap} bytes at ecc={ecc}.  "
            f"Use a larger cover image or reduce ECC."
        )

    seed = _derive_payload_seed(password, embed_salt)
    rng  = np.random.default_rng(seed)
    pos  = rng.choice(len(available_indices), size=payload_slots, replace=False)
    pos  = available_indices[pos]

    # --- Write header ---
    flat[hdr_offset:hdr_end] = (flat[hdr_offset:hdr_end] & 0xFE) | hdr_bits_rep

    # --- Write payload ---
    flat[pos] = (flat[pos] & 0xFE) | payload_bits_enc

    arr[:, :, :3] = flat.reshape(rgb.shape)

    out_img = Image.fromarray(arr, mode=mode)
    fmt     = _output_format(out_path)
    kwargs  = {}
    if icc and fmt in ("PNG", "TIFF"):
        kwargs["icc_profile"] = icc

    out_img.save(out_path, format=fmt, **kwargs)


# ---------------------------------------------------------------------------
# Extract
# ---------------------------------------------------------------------------

def extract_png(in_path: Path, *, password: str) -> bytes:
    """Extract and verify the payload from a stego image."""
    _assert_lossless_input(in_path)

    im   = Image.open(in_path).convert("RGBA" if Image.open(in_path).mode == "RGBA" else "RGB")
    arr  = np.array(im, dtype=np.uint8)
    rgb  = arr[:, :, :3]
    flat = rgb.reshape(-1)
    lsb  = (flat & 1).astype(np.uint8)

    total_slots  = len(lsb)
    header_slots = _HDR_BYTES * 8 * _HDR_REPEAT

    # --- Locate header using password-derived offset ---
    hdr_offset = _derive_header_offset(password, total_slots)
    hdr_end    = hdr_offset + header_slots

    if hdr_end > total_slots:
        raise ValueError(
            "Image too small or wrong image (header offset exceeds image size)."
        )

    hdr_bits_rep  = lsb[hdr_offset:hdr_end]
    hdr_bits      = _ecc_decode(hdr_bits_rep, _HDR_REPEAT)
    hdr_enc       = _from_bits(hdr_bits)[:_HDR_BYTES]

    # embed_salt is stored plaintext in bytes [0:16] (it is public per-message
    # randomness, like an IV).  Bytes [16:32] are XOR-encrypted with a mask
    # derived from password + embed_salt.
    embed_salt    = bytes(hdr_enc[:16])
    hdr_mask      = _derive_header_mask(password, embed_salt)
    sensitive_dec = _apply_sensitive_mask(bytes(hdr_enc[16:]), hdr_mask)
    hdr_plain     = embed_salt + sensitive_dec

    try:
        h = _unpack_header(bytes(hdr_plain))
    except ValueError as exc:
        raise ValueError(
            f"Header decode failed (wrong password or corrupted image): {exc}"
        ) from exc

    # --- Reproduce payload positions ---
    available_mask = np.ones(total_slots, dtype=bool)
    available_mask[hdr_offset:hdr_end] = False
    available_indices = np.where(available_mask)[0]

    payload_slots = h.payload_len * 8 * h.ecc
    if payload_slots > len(available_indices):
        raise ValueError(
            "Corrupt stego image: declared payload size exceeds image capacity."
        )

    seed = _derive_payload_seed(password, h.embed_salt)
    rng  = np.random.default_rng(seed)
    pos  = rng.choice(len(available_indices), size=payload_slots, replace=False)
    pos  = available_indices[pos]

    payload_bits_enc = lsb[pos]
    payload_bits     = _ecc_decode(payload_bits_enc, h.ecc)
    payload          = _from_bits(payload_bits)[: h.payload_len]

    if crc32(payload) != h.payload_crc32:
        raise ValueError(
            "CRC-32 mismatch: wrong password, wrong file, or corrupted stego image."
        )

    return payload
