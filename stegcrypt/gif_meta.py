"""
stegcrypt.gif_meta
~~~~~~~~~~~~~~~~~~
GIF comment-block metadata embedding.

Honesty note: this is NOT covert steganography.  The payload is stored in
the GIF comment extension block, which is visible to any GIF parser.
It is useful for encrypted annotation, not covert communication.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image


def embed_gif_comment(in_path: Path, out_path: Path, payload: bytes) -> None:
    """Store *payload* in the GIF comment block of *in_path*, write to *out_path*."""
    im = Image.open(in_path)

    save_kwargs: dict = dict(
        format="GIF",
        comment=payload,
        optimize=False,
    )

    if getattr(im, "is_animated", False):
        frames: list[Image.Image] = []
        durations: list[int] = []
        for i in range(im.n_frames):
            im.seek(i)
            frames.append(im.copy())
            durations.append(int(im.info.get("duration", 100)))

        frames[0].save(
            out_path,
            save_all=True,
            append_images=frames[1:],
            loop=int(im.info.get("loop", 0)),
            duration=durations,
            disposal=int(im.info.get("disposal", 2)),
            **save_kwargs,
        )
    else:
        im.save(out_path, **save_kwargs)


def extract_gif_comment(in_path: Path) -> bytes:
    """Read the comment block payload from a GIF file."""
    im = Image.open(in_path)
    comment = im.info.get("comment")
    if comment is None:
        raise ValueError("No GIF comment payload found in this file.")
    return comment if isinstance(comment, bytes) else comment.encode("utf-8")
