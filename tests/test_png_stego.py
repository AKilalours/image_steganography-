"""
tests/test_png_stego.py
~~~~~~~~~~~~~~~~~~~~~~~
Full test suite for the PNG steganography layer.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from stegcrypt.crypto import encrypt_text, decrypt_text
from stegcrypt.payload import pack, unpack
from stegcrypt.png_stego import embed_png, extract_png, png_capacity_bytes


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_cover(tmp_path: Path, w: int = 256, h: int = 256, mode: str = "RGB",
                seed: int = 42) -> Path:
    rng = np.random.default_rng(seed)
    if mode == "RGB":
        arr = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
    else:
        arr = rng.integers(0, 256, (h, w, 4), dtype=np.uint8)
    img = Image.fromarray(arr, mode=mode)
    p   = tmp_path / f"cover_{w}x{h}_{mode}.png"
    img.save(p, format="PNG")
    return p


def _roundtrip(tmp_path, msg: str, password: str, ecc: int = 1,
               w: int = 256, h: int = 256) -> str:
    cover_p = _make_cover(tmp_path, w=w, h=h)
    stego_p = tmp_path / "stego.png"

    payload_raw = encrypt_text(msg, password)
    blob        = pack(payload_raw)
    embed_png(cover_p, stego_p, blob, password=password, ecc=ecc)

    blob2        = extract_png(stego_p, password=password)
    payload_raw2 = unpack(blob2)
    out, _, _    = decrypt_text(payload_raw2, password)
    return out


# ---------------------------------------------------------------------------
# Basic roundtrip
# ---------------------------------------------------------------------------

def test_roundtrip_basic(tmp_path):
    assert _roundtrip(tmp_path, "hello stegcrypt v3", "pw123") == "hello stegcrypt v3"


def test_roundtrip_unicode(tmp_path):
    msg = "日本語 🔐 العربية Ελληνικά"
    assert _roundtrip(tmp_path, msg, "pw_unicode") == msg


def test_roundtrip_empty_message(tmp_path):
    assert _roundtrip(tmp_path, "", "pw") == ""


def test_roundtrip_long_message(tmp_path):
    msg = "X" * 2000
    assert _roundtrip(tmp_path, msg, "pw_long", w=512, h=512) == msg


@pytest.mark.parametrize("ecc", [1, 3, 5, 9])
def test_roundtrip_ecc(tmp_path, ecc):
    assert _roundtrip(tmp_path, "ecc test", "pw", ecc=ecc, w=512, h=512) == "ecc test"


# ---------------------------------------------------------------------------
# RGBA / mode preservation
# ---------------------------------------------------------------------------

def test_roundtrip_rgba_cover(tmp_path):
    cover_p = _make_cover(tmp_path, mode="RGBA")
    stego_p = tmp_path / "stego_rgba.png"
    msg     = "rgba roundtrip"
    blob    = pack(encrypt_text(msg, "pw"))
    embed_png(cover_p, stego_p, blob, password="pw")

    out_img = Image.open(stego_p)
    assert out_img.mode == "RGBA", "RGBA mode must be preserved"

    out, _, _ = decrypt_text(unpack(extract_png(stego_p, password="pw")), "pw")
    assert out == msg


# ---------------------------------------------------------------------------
# Lossless output formats
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("ext", [".png", ".bmp", ".tif"])
def test_roundtrip_lossless_formats(tmp_path, ext):
    cover_p = _make_cover(tmp_path)
    stego_p = tmp_path / f"stego{ext}"
    msg     = f"format {ext}"
    embed_png(cover_p, stego_p, pack(encrypt_text(msg, "pw")), password="pw")
    out, _, _ = decrypt_text(unpack(extract_png(stego_p, password="pw")), "pw")
    assert out == msg


def test_jpeg_output_rejected(tmp_path):
    cover_p = _make_cover(tmp_path)
    stego_p = tmp_path / "stego.jpg"
    with pytest.raises(ValueError, match="lossless"):
        embed_png(cover_p, stego_p, b"x", password="pw")


# ---------------------------------------------------------------------------
# Wrong password
# ---------------------------------------------------------------------------

def test_wrong_password_raises(tmp_path):
    cover_p = _make_cover(tmp_path)
    stego_p = tmp_path / "stego.png"
    embed_png(cover_p, stego_p, pack(encrypt_text("secret", "correct")), password="correct")
    with pytest.raises(Exception):
        extract_png(stego_p, password="wrong")


def test_wrong_password_crc_fails(tmp_path):
    """Wrong password → different PRNG positions → garbage bytes → detection error."""
    cover_p = _make_cover(tmp_path)
    stego_p = tmp_path / "stego.png"
    embed_png(cover_p, stego_p, pack(encrypt_text("secret", "pw")), password="pw")
    # A wrong password will produce either a capacity error (garbage payload_len)
    # or a CRC mismatch — either way it must raise ValueError
    with pytest.raises(ValueError):
        extract_png(stego_p, password="wrong_pw")


# ---------------------------------------------------------------------------
# No detectable magic bytes
# ---------------------------------------------------------------------------

def test_no_plaintext_magic_in_lsb(tmp_path):
    """Stego image must NOT contain the old STEGCRYPT magic strings in raw LSB data."""
    cover_p = _make_cover(tmp_path)
    stego_p = tmp_path / "stego.png"
    embed_png(cover_p, stego_p, pack(encrypt_text("hidden", "pw")), password="pw")

    raw = stego_p.read_bytes()
    assert b"STEGCRYPTv" not in raw
    assert b"SC2!"        not in raw  # old v2 magic


# ---------------------------------------------------------------------------
# Capacity
# ---------------------------------------------------------------------------

def test_capacity_reasonable():
    img = Image.new("RGB", (256, 256))
    cap = png_capacity_bytes(img, ecc=1)
    # 256*256*3 bits / 8 minus header overhead
    assert cap > 20_000, f"Capacity too low: {cap}"


def test_capacity_decreases_with_ecc():
    img  = Image.new("RGB", (256, 256))
    cap1 = png_capacity_bytes(img, ecc=1)
    cap3 = png_capacity_bytes(img, ecc=3)
    cap9 = png_capacity_bytes(img, ecc=9)
    assert cap1 > cap3 > cap9


def test_payload_too_large_raises(tmp_path):
    tiny    = Image.new("RGB", (8, 8))
    cover_p = tmp_path / "tiny.png"
    tiny.save(cover_p, format="PNG")
    with pytest.raises(ValueError):
        embed_png(cover_p, tmp_path / "out.png", b"X" * 10_000, password="pw")


# ---------------------------------------------------------------------------
# Cover image is not modified beyond LSB
# ---------------------------------------------------------------------------

def test_only_lsb_modified(tmp_path):
    cover_p = _make_cover(tmp_path, seed=7)
    stego_p = tmp_path / "stego.png"
    embed_png(cover_p, stego_p, pack(encrypt_text("test", "pw")), password="pw")

    cover = np.array(Image.open(cover_p))
    stego = np.array(Image.open(stego_p))

    diff = np.abs(cover[:, :, :3].astype(int) - stego[:, :, :3].astype(int))
    assert diff.max() <= 1, "Pixel values changed by more than 1 (LSB only allowed)"


# ---------------------------------------------------------------------------
# ECC validity
# ---------------------------------------------------------------------------

def test_invalid_ecc_raises(tmp_path):
    cover_p = _make_cover(tmp_path)
    with pytest.raises(ValueError, match="ecc"):
        embed_png(cover_p, tmp_path / "out.png", b"x", password="pw", ecc=0)
    with pytest.raises(ValueError, match="ecc"):
        embed_png(cover_p, tmp_path / "out.png", b"x", password="pw", ecc=10)


# ---------------------------------------------------------------------------
# Different passwords produce different stego
# ---------------------------------------------------------------------------

def test_different_passwords_produce_different_stego(tmp_path):
    cover_p = _make_cover(tmp_path, seed=99)
    s1 = tmp_path / "s1.png"
    s2 = tmp_path / "s2.png"
    blob = pack(encrypt_text("msg", "pw1"))
    embed_png(cover_p, s1, blob, password="pw1")
    embed_png(cover_p, s2, blob, password="pw2")
    arr1 = np.array(Image.open(s1))
    arr2 = np.array(Image.open(s2))
    assert not np.array_equal(arr1, arr2), "Different passwords should produce different LSB patterns"
