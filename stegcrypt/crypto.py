"""
stegcrypt.crypto
~~~~~~~~~~~~~~~~
Authenticated encryption layer: AES-256-GCM + scrypt KDF.

Design decisions
----------------
* KDF parameters are stored IN the envelope so they can be upgraded
  without breaking existing ciphertexts (KDF agility).
* Envelope version field allows future wire-format changes.
* AAD binds the ciphertext to the application + version, preventing
  cross-context replay.
* SHA-256 of the original plaintext (pre-LLM) is stored as an optional
  audit field and is NOT used for security – GCM tag already covers integrity.
* Compression is applied BEFORE encryption (leak-safe ordering).
* All random material is sourced from os.urandom (CSPRNG).
"""

from __future__ import annotations

import json
import os
import zlib
from dataclasses import asdict, dataclass
from typing import Literal, Optional

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt

from .util import b64d, b64e, sha256_hex

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

Compression = Literal["none", "zlib"]

ENVELOPE_VERSION = 3
APP_AAD = b"stegcrypt:v3"  # domain-separation AAD

# ---------------------------------------------------------------------------
# KDF parameter profiles  (stored inside every envelope)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ScryptParams:
    n: int   # CPU/memory cost  (must be power of 2)
    r: int   # block size
    p: int   # parallelism

# "interactive" – fast enough for CLI, hard enough against offline attack
SCRYPT_INTERACTIVE = ScryptParams(n=2**16, r=8, p=1)
# "sensitive" – for high-value payloads; noticeably slower (~1 s on modern HW)
SCRYPT_SENSITIVE   = ScryptParams(n=2**20, r=8, p=1)

_PROFILE_MAP: dict[str, ScryptParams] = {
    "interactive": SCRYPT_INTERACTIVE,
    "sensitive":   SCRYPT_SENSITIVE,
}

# ---------------------------------------------------------------------------
# Envelope data-class
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CryptoEnvelope:
    """Wire-format for an encrypted payload.

    All binary fields are URL-safe base64, no padding.
    """
    ver: int                          # envelope version
    alg: str                          # e.g. "AES-256-GCM"
    kdf: str                          # e.g. "scrypt"
    kdf_n: int                        # scrypt N
    kdf_r: int                        # scrypt r
    kdf_p: int                        # scrypt p
    kdf_salt_b64: str                 # 16-byte random salt
    nonce_b64: str                    # 12-byte GCM nonce
    ct_b64: str                       # ciphertext + 16-byte GCM tag
    compression: Compression
    pt_sha256: str                    # SHA-256 of plaintext (audit / LLM drift check)
    orig_sha256: Optional[str]        # SHA-256 of pre-LLM original (optional audit)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _derive_key(password: str, salt: bytes, params: ScryptParams) -> bytes:
    kdf = Scrypt(salt=salt, length=32, n=params.n, r=params.r, p=params.p)
    return kdf.derive(password.encode("utf-8"))


def _params_from_envelope(env_dict: dict) -> ScryptParams:
    return ScryptParams(n=env_dict["kdf_n"], r=env_dict["kdf_r"], p=env_dict["kdf_p"])

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def encrypt_text(
    plaintext: str,
    password: str,
    *,
    compression: Compression = "zlib",
    kdf_profile: str = "interactive",
    aad: bytes = APP_AAD,
    orig_sha256: Optional[str] = None,
) -> bytes:
    """Encrypt *plaintext* with *password*.

    Returns a UTF-8-encoded JSON blob (the CryptoEnvelope).
    """
    params = _PROFILE_MAP.get(kdf_profile, SCRYPT_INTERACTIVE)

    pt_bytes = plaintext.encode("utf-8")
    if compression == "zlib":
        pt_bytes = zlib.compress(pt_bytes, level=9)
    elif compression != "none":
        raise ValueError(f"Unknown compression mode: {compression!r}")

    kdf_salt = os.urandom(16)
    key      = _derive_key(password, kdf_salt, params)
    nonce    = os.urandom(12)

    ct = AESGCM(key).encrypt(nonce, pt_bytes, aad)

    env = CryptoEnvelope(
        ver=ENVELOPE_VERSION,
        alg="AES-256-GCM",
        kdf="scrypt",
        kdf_n=params.n,
        kdf_r=params.r,
        kdf_p=params.p,
        kdf_salt_b64=b64e(kdf_salt),
        nonce_b64=b64e(nonce),
        ct_b64=b64e(ct),
        compression=compression,
        pt_sha256=sha256_hex(plaintext),
        orig_sha256=orig_sha256,
    )
    return json.dumps(asdict(env), separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def decrypt_text(
    payload_raw: bytes,
    password: str,
    *,
    aad: bytes = APP_AAD,
) -> tuple[str, str, Optional[str]]:
    """Decrypt and verify a CryptoEnvelope.

    Returns ``(plaintext, pt_sha256, orig_sha256)``.

    Raises
    ------
    ValueError
        On wrong password, corrupt data, or SHA mismatch.
    """
    try:
        obj = json.loads(payload_raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Malformed envelope (not valid UTF-8 JSON).") from exc

    params = _params_from_envelope(obj)
    kdf_salt = b64d(obj["kdf_salt_b64"])
    nonce    = b64d(obj["nonce_b64"])
    ct       = b64d(obj["ct_b64"])
    compression   = obj.get("compression", "none")
    pt_sha_stored = obj.get("pt_sha256", "")
    orig_sha      = obj.get("orig_sha256", None)

    key = _derive_key(password, kdf_salt, params)

    try:
        pt_bytes = AESGCM(key).decrypt(nonce, ct, aad)
    except InvalidTag as exc:
        # Surface a clear, actionable message instead of a cryptic library error.
        raise ValueError(
            "Decryption failed: wrong password or corrupted ciphertext."
        ) from exc

    if compression == "zlib":
        try:
            pt_bytes = zlib.decompress(pt_bytes)
        except zlib.error as exc:
            raise ValueError("Decompression failed after decryption.") from exc
    elif compression != "none":
        raise ValueError(f"Unknown compression in envelope: {compression!r}")

    text = pt_bytes.decode("utf-8", errors="strict")

    # GCM already guarantees integrity; this gives an explicit audit signal
    # when the LLM transform changed the plaintext after the fact.
    if pt_sha_stored and sha256_hex(text) != pt_sha_stored:
        raise ValueError(
            "Plaintext SHA-256 mismatch after decryption "
            "(LLM transform may have altered the payload; check orig_sha256)."
        )

    return text, pt_sha_stored, orig_sha
