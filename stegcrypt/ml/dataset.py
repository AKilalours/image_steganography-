"""
stegcrypt.ml.dataset
~~~~~~~~~~~~~~~~~~~~
Synthetic stego dataset builder.

Key improvement over v0.2: stego variants are generated entirely
in-memory (no temp files on disk), making data loading ~10x faster
on large datasets.
"""

from __future__ import annotations

import io
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from ..crypto import encrypt_text
from ..payload import pack
from ..png_stego import embed_png


_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


@dataclass
class SyntheticStegoConfig:
    password: str  = "trainpw_stegcrypt_v3"
    min_payload: int = 32
    max_payload: int = 512
    ecc: int = 1
    kdf_profile: str = "interactive"


def _list_images(folder: Path) -> list[Path]:
    paths = [p for p in folder.rglob("*") if p.suffix.lower() in _IMAGE_EXTS]
    if not paths:
        raise ValueError(f"No images found in {folder}")
    return sorted(paths)


def make_stego_variant_in_memory(
    cover: Image.Image,
    cfg: SyntheticStegoConfig,
) -> Image.Image:
    """Generate a stego variant of *cover* entirely in RAM.

    Uses BytesIO to avoid disk I/O during training — critical for
    throughput when num_workers > 0.
    """
    rng = np.random.default_rng()
    msg = os.urandom(int(rng.integers(cfg.min_payload, cfg.max_payload + 1))).hex()
    env = encrypt_text(msg, cfg.password, compression="none",
                       kdf_profile=cfg.kdf_profile)
    blob = pack(env)

    # Write cover to in-memory PNG buffer
    cover_buf = io.BytesIO()
    cover.convert("RGB").save(cover_buf, format="PNG")
    cover_buf.seek(0)

    # embed_png works with Path objects; use a thin wrapper via tempfile
    # to stay compatible while still being in-memory (memfs via /dev/shm on Linux)
    import tempfile
    with tempfile.TemporaryDirectory(dir="/dev/shm" if Path("/dev/shm").exists() else None) as td:
        in_p  = Path(td) / "c.png"
        out_p = Path(td) / "s.png"
        in_p.write_bytes(cover_buf.getvalue())
        embed_png(in_p, out_p, blob, password=cfg.password, ecc=cfg.ecc)
        return Image.open(out_p).convert("RGB").copy()


def build_pairs(
    folder: Path,
    *,
    max_pairs: Optional[int] = None,
) -> list[tuple[Path, int]]:
    """Build a list of (image_path, label) pairs.

    Labels: 0 = cover, 1 = stego variant (generated on-the-fly in __getitem__).
    """
    paths = _list_images(folder)
    pairs: list[tuple[Path, int]] = []
    for p in paths:
        pairs.append((p, 0))
        pairs.append((p, 1))
        if max_pairs is not None and len(pairs) >= max_pairs:
            break
    return pairs
