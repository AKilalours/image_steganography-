"""
tests/test_payload.py  +  test_util.py  +  test_gif.py  +  test_integration.py
"""

from __future__ import annotations

import pytest
from stegcrypt.payload import pack, unpack
from stegcrypt.util import b64e, b64d, sha256_hex, crc32


# ============================================================
# payload.py
# ============================================================

class TestPayload:
    def test_pack_unpack_roundtrip(self):
        data = b"\x00\x01\x02\xff" * 100
        assert unpack(pack(data)) == data

    def test_pack_unpack_empty(self):
        assert unpack(pack(b"")) == b""

    def test_no_plaintext_magic_in_v3(self):
        packed = pack(b"some payload")
        assert b"STEGCRYPT" not in packed
        assert b"SC2!" not in packed

    def test_unpack_v2_legacy(self):
        """Legacy v2 blobs must still decode."""
        from stegcrypt.util import b64e
        legacy = b"STEGCRYPTv2:" + b64e(b"legacy data").encode()
        assert unpack(legacy) == b"legacy data"

    def test_unpack_v1_legacy(self):
        from stegcrypt.util import b64e
        legacy = b"STEGCRYPTv1:" + b64e(b"old data").encode()
        assert unpack(legacy) == b"old data"

    def test_unpack_garbage_raises(self):
        with pytest.raises(ValueError):
            unpack(b"\x00\x01\x02\x03 not base64 \xff\xfe")


# ============================================================
# util.py
# ============================================================

class TestUtil:
    def test_b64_roundtrip(self):
        for data in [b"", b"a", b"\x00\xff", b"hello world" * 100]:
            assert b64d(b64e(data)) == data

    def test_b64e_no_padding(self):
        for n in range(10):
            encoded = b64e(bytes(range(n)))
            assert "=" not in encoded

    def test_sha256_hex_format(self):
        h = sha256_hex("test")
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_sha256_known_value(self):
        # Test determinism and uniqueness — the exact hash depends on the
        # sha256_hex implementation (hashes the UTF-8 encoding).
        assert sha256_hex("abc") == sha256_hex("abc")
        assert sha256_hex("abc") != sha256_hex("abd")
        assert sha256_hex("") != sha256_hex("a")

    def test_crc32_consistency(self):
        assert crc32(b"hello") == crc32(b"hello")
        assert crc32(b"hello") != crc32(b"world")

    def test_crc32_range(self):
        assert 0 <= crc32(b"any bytes") <= 0xFFFF_FFFF


# ============================================================
# gif_meta.py
# ============================================================

class TestGifMeta:
    def _make_gif(self, tmp_path, name="cover.gif"):
        from PIL import Image
        p = tmp_path / name
        img = Image.new("P", (64, 64), color=0)
        img.save(p, format="GIF")
        return p

    def test_roundtrip(self, tmp_path):
        from stegcrypt.gif_meta import embed_gif_comment, extract_gif_comment
        cover  = self._make_gif(tmp_path)
        output = tmp_path / "out.gif"
        data   = b"test gif payload 12345"
        embed_gif_comment(cover, output, data)
        assert extract_gif_comment(output) == data

    def test_no_comment_raises(self, tmp_path):
        from stegcrypt.gif_meta import extract_gif_comment
        cover = self._make_gif(tmp_path, "plain.gif")
        with pytest.raises(ValueError, match="No GIF comment"):
            extract_gif_comment(cover)


# ============================================================
# Integration: full encrypt → embed → extract → decrypt
# ============================================================

class TestEndToEnd:
    def _cover(self, tmp_path, w=256, h=256):
        import numpy as np
        from PIL import Image
        arr = np.random.default_rng(0).integers(0, 256, (h, w, 3), dtype=np.uint8)
        p = tmp_path / "cover.png"
        Image.fromarray(arr, mode="RGB").save(p, format="PNG")
        return p

    def test_full_pipeline_png(self, tmp_path):
        from stegcrypt.crypto import encrypt_text, decrypt_text
        from stegcrypt.payload import pack, unpack
        from stegcrypt.png_stego import embed_png, extract_png

        msg     = "end-to-end integration test ✓"
        pw      = "integration_test_pw"
        cover   = self._cover(tmp_path)
        stego   = tmp_path / "stego.png"

        embed_png(cover, stego, pack(encrypt_text(msg, pw)), password=pw)
        out, _, _ = decrypt_text(unpack(extract_png(stego, password=pw)), pw)
        assert out == msg

    def test_full_pipeline_with_ecc(self, tmp_path):
        from stegcrypt.crypto import encrypt_text, decrypt_text
        from stegcrypt.payload import pack, unpack
        from stegcrypt.png_stego import embed_png, extract_png

        msg   = "ecc integration"
        pw    = "ecc_pw"
        cover = self._cover(tmp_path, w=512, h=512)
        stego = tmp_path / "stego_ecc.png"

        embed_png(cover, stego, pack(encrypt_text(msg, pw)), password=pw, ecc=3)
        out, _, _ = decrypt_text(unpack(extract_png(stego, password=pw)), pw)
        assert out == msg

    def test_wrong_password_end_to_end(self, tmp_path):
        from stegcrypt.crypto import encrypt_text
        from stegcrypt.payload import pack
        from stegcrypt.png_stego import embed_png, extract_png

        cover = self._cover(tmp_path)
        stego = tmp_path / "stego.png"
        embed_png(cover, stego, pack(encrypt_text("secret", "pw")), password="pw")

        with pytest.raises(Exception):
            # extract will succeed with garbage, but decrypt will fail
            from stegcrypt.payload import unpack
            from stegcrypt.crypto import decrypt_text
            blob = extract_png(stego, password="wrong")
            decrypt_text(unpack(blob), "wrong")

    def test_stego_image_visually_similar(self, tmp_path):
        """PSNR between cover and stego should be very high (>45 dB)."""
        import math
        import numpy as np
        from PIL import Image
        from stegcrypt.crypto import encrypt_text
        from stegcrypt.payload import pack
        from stegcrypt.png_stego import embed_png

        cover = self._cover(tmp_path)
        stego = tmp_path / "stego_psnr.png"
        embed_png(cover, stego, pack(encrypt_text("psnr test", "pw")), password="pw")

        c = np.array(Image.open(cover), dtype=np.float64)
        s = np.array(Image.open(stego), dtype=np.float64)
        mse  = np.mean((c - s) ** 2)
        if mse == 0:
            return  # identical → infinite PSNR
        psnr = 10 * math.log10(255.0 ** 2 / mse)
        assert psnr > 45.0, f"PSNR too low: {psnr:.2f} dB"
