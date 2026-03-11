"""stegcrypt.util — shared low-level helpers."""
from __future__ import annotations

import base64
import hashlib
import zlib as _zlib


def b64e(b: bytes) -> str:
    """URL-safe base64 encode, no padding."""
    return base64.urlsafe_b64encode(b).decode("utf-8").rstrip("=")


def b64d(s: str) -> bytes:
    """URL-safe base64 decode, tolerates missing padding."""
    pad = "=" * ((4 - len(s) % 4) % 4)
    return base64.urlsafe_b64decode((s + pad).encode("utf-8"))


def sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def crc32(b: bytes) -> int:
    return _zlib.crc32(b) & 0xFFFF_FFFF
