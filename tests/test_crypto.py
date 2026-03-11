"""
tests/test_crypto.py
~~~~~~~~~~~~~~~~~~~~
Full test suite for the crypto layer.
"""

from __future__ import annotations

import json

import pytest

from stegcrypt.crypto import (
    encrypt_text,
    decrypt_text,
    SCRYPT_INTERACTIVE,
    SCRYPT_SENSITIVE,
    APP_AAD,
)


# ---------------------------------------------------------------------------
# Roundtrip
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("compression", ["none", "zlib"])
@pytest.mark.parametrize("kdf_profile", ["interactive"])   # skip 'sensitive' in tests (too slow)
def test_roundtrip(compression, kdf_profile):
    msg = "Hello, StegCrypt-AI v3! 🔐"
    blob = encrypt_text(msg, "pw", compression=compression, kdf_profile=kdf_profile)
    result, _, _ = decrypt_text(blob, "pw")
    assert result == msg


def test_roundtrip_unicode():
    msg  = "日本語テスト — 中文 — Ελληνικά — Русский"
    blob = encrypt_text(msg, "pw_unicode")
    out, _, _ = decrypt_text(blob, "pw_unicode")
    assert out == msg


def test_roundtrip_empty_string():
    blob = encrypt_text("", "pw")
    out, _, _ = decrypt_text(blob, "pw")
    assert out == ""


def test_roundtrip_large_payload():
    msg  = "A" * 50_000
    blob = encrypt_text(msg, "pw_large")
    out, _, _ = decrypt_text(blob, "pw_large")
    assert out == msg


# ---------------------------------------------------------------------------
# Wrong password
# ---------------------------------------------------------------------------

def test_wrong_password_raises():
    blob = encrypt_text("secret", "correct_pw")
    with pytest.raises(ValueError, match="wrong password"):
        decrypt_text(blob, "wrong_pw")


def test_wrong_password_does_not_leak_plaintext():
    """Ensure no partial plaintext is leaked when decryption fails."""
    blob = encrypt_text("top secret message", "correct")
    try:
        decrypt_text(blob, "incorrect")
        pytest.fail("Should have raised ValueError")
    except ValueError as exc:
        assert "top secret" not in str(exc).lower()


# ---------------------------------------------------------------------------
# KDF agility
# ---------------------------------------------------------------------------

def test_kdf_params_stored_in_envelope():
    blob = encrypt_text("test", "pw", kdf_profile="interactive")
    obj  = json.loads(blob)
    assert obj["kdf_n"] == SCRYPT_INTERACTIVE.n
    assert obj["kdf_r"] == SCRYPT_INTERACTIVE.r
    assert obj["kdf_p"] == SCRYPT_INTERACTIVE.p


def test_envelope_version():
    blob = encrypt_text("test", "pw")
    obj  = json.loads(blob)
    assert obj["ver"] == 3
    assert obj["alg"] == "AES-256-GCM"
    assert obj["kdf"] == "scrypt"


def test_envelope_fields_present():
    blob = encrypt_text("test", "pw", orig_sha256="abc123")
    obj  = json.loads(blob)
    for field in ("kdf_salt_b64", "nonce_b64", "ct_b64", "compression",
                  "pt_sha256", "orig_sha256"):
        assert field in obj, f"Missing field: {field}"
    assert obj["orig_sha256"] == "abc123"


# ---------------------------------------------------------------------------
# Integrity / corruption
# ---------------------------------------------------------------------------

def test_tampered_ciphertext_raises():
    blob = bytearray(encrypt_text("tamper me", "pw"))
    # Flip a byte in the middle of the JSON (inside ct_b64)
    blob[len(blob) // 2] ^= 0xFF
    with pytest.raises((ValueError, Exception)):
        decrypt_text(bytes(blob), "pw")


def test_truncated_ciphertext_raises():
    blob = encrypt_text("truncate me", "pw")
    with pytest.raises(Exception):
        decrypt_text(blob[:len(blob) // 2], "pw")


def test_empty_bytes_raises():
    with pytest.raises(Exception):
        decrypt_text(b"", "pw")


def test_invalid_json_raises():
    with pytest.raises(ValueError, match="Malformed"):
        decrypt_text(b"not json at all", "pw")


# ---------------------------------------------------------------------------
# Determinism / uniqueness
# ---------------------------------------------------------------------------

def test_two_encryptions_differ():
    """Each encryption must produce a unique ciphertext (fresh nonce + salt)."""
    b1 = encrypt_text("same", "pw")
    b2 = encrypt_text("same", "pw")
    assert b1 != b2


def test_nonces_differ():
    b1 = json.loads(encrypt_text("msg", "pw"))
    b2 = json.loads(encrypt_text("msg", "pw"))
    assert b1["nonce_b64"] != b2["nonce_b64"]
    assert b1["kdf_salt_b64"] != b2["kdf_salt_b64"]


# ---------------------------------------------------------------------------
# SHA audit
# ---------------------------------------------------------------------------

def test_pt_sha256_correct():
    from stegcrypt.util import sha256_hex
    msg  = "audit me"
    blob = encrypt_text(msg, "pw")
    _, pt_sha, _ = decrypt_text(blob, "pw")
    assert pt_sha == sha256_hex(msg)


def test_orig_sha256_preserved():
    blob = encrypt_text("llm output", "pw", orig_sha256="original_sha")
    _, _, orig = decrypt_text(blob, "pw")
    assert orig == "original_sha"
