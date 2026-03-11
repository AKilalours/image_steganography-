"""
stegcrypt.payload
~~~~~~~~~~~~~~~~~
Binary framing layer that sits between the stego carrier and the crypto
envelope.

Version 3 change: NO plaintext magic bytes are written.  The payload is
a raw base64-encoded crypto envelope.  The presence of a StegCrypt payload
is only detectable if you have the password and can derive the correct
embedding positions — which is the entire security goal.

Backward compatibility: we can still unpack v1/v2 payloads (they start
with plaintext magic), but we never write them.
"""

from __future__ import annotations

from .util import b64d, b64e

_MAGIC_V1 = b"STEGCRYPTv1:"
_MAGIC_V2 = b"STEGCRYPTv2:"


def pack(payload_raw: bytes) -> bytes:
    """Wrap *payload_raw* for embedding.  No plaintext magic (v3)."""
    # v3: raw base64 only — no leading magic, no detectable marker.
    return b64e(payload_raw).encode("utf-8")


def unpack(blob: bytes) -> bytes:
    """Unwrap a payload blob, handling v1/v2/v3 wire formats."""
    # Legacy v1/v2: had plaintext magic prefix
    if blob.startswith(_MAGIC_V1):
        return b64d(blob[len(_MAGIC_V1):].decode("utf-8"))
    if blob.startswith(_MAGIC_V2):
        return b64d(blob[len(_MAGIC_V2):].decode("utf-8"))
    # v3: raw base64 (try direct decode)
    try:
        return b64d(blob.decode("utf-8"))
    except Exception as exc:
        raise ValueError("Cannot unpack payload: unknown format or corrupted data.") from exc
