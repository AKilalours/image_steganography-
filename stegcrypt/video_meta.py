"""
stegcrypt.video_meta
~~~~~~~~~~~~~~~~~~~~
Video metadata comment embedding via ffmpeg.

Honesty note: this is NOT frame-level steganography.  The payload is
stored in the container's comment tag (MP4 / MKV / etc.), which any
metadata tool can read.  It is suitable for encrypted annotation.

Sidecar behaviour: if ffmpeg is unavailable or the payload is too large
for the metadata comment field, a ``.steg`` sidecar is written ALONGSIDE
the output video.  The sidecar path is returned so callers can inform
the user.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Optional

_META_MAX_CHARS = 20_000  # conservative limit for metadata comment fields


def _have_ffmpeg() -> bool:
    return (
        shutil.which("ffmpeg") is not None
        and shutil.which("ffprobe") is not None
    )


def sidecar_path(video_path: Path) -> Path:
    return video_path.with_suffix(video_path.suffix + ".steg")


def embed_video_metadata(
    in_path: Path,
    out_path: Path,
    payload: bytes,
) -> tuple[bool, Optional[Path]]:
    """Embed *payload* as a video metadata comment.

    Returns ``(embedded_in_metadata, sidecar_path_or_None)``.
    """
    payload_str = payload.decode("utf-8", errors="strict")
    sc = sidecar_path(out_path)

    # Always write the sidecar as a fallback / audit copy.
    sc.write_text(payload_str, encoding="utf-8")

    if not _have_ffmpeg():
        shutil.copy2(in_path, out_path)
        return False, sc

    if len(payload_str) > _META_MAX_CHARS:
        shutil.copy2(in_path, out_path)
        return False, sc

    cmd = [
        "ffmpeg", "-y",
        "-i", str(in_path),
        "-map", "0",
        "-c", "copy",
        "-metadata", f"comment={payload_str}",
        str(out_path),
    ]
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        shutil.copy2(in_path, out_path)
        return False, sc

    return True, sc


def extract_video_metadata(in_path: Path) -> bytes:
    """Extract payload from video metadata comment, falling back to sidecar."""
    if _have_ffmpeg():
        cmd = [
            "ffprobe", "-v", "error",
            "-show_entries", "format_tags=comment",
            "-of", "json",
            str(in_path),
        ]
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=30,
        )
        if result.returncode == 0 and result.stdout.strip():
            try:
                obj   = json.loads(result.stdout)
                tags  = (obj.get("format") or {}).get("tags") or {}
                value = tags.get("comment")
                if value:
                    return value.encode("utf-8")
            except (json.JSONDecodeError, AttributeError):
                pass

    sc = sidecar_path(in_path)
    if sc.exists():
        return sc.read_text(encoding="utf-8").encode("utf-8")

    raise ValueError(
        "No payload found: no metadata comment and no .steg sidecar present."
    )
