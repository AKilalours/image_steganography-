"""
stegcrypt.wizard
~~~~~~~~~~~~~~~~
Interactive terminal wizard for StegCrypt-AI v3.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from getpass import getpass
from pathlib import Path
from typing import Optional

from PIL import Image

from .crypto import encrypt_text, decrypt_text
from .payload import pack, unpack
from .png_stego import embed_png, extract_png, png_capacity_bytes
from .gif_meta import embed_gif_comment, extract_gif_comment
from .video_meta import embed_video_metadata, extract_video_metadata

# ---------------------------------------------------------------------------
# Language menu
# ---------------------------------------------------------------------------

_LANGUAGES = [
    "Arabic", "Bengali", "Chinese (Simplified)", "Chinese (Traditional)",
    "Dutch", "English", "French", "German", "Hindi", "Indonesian",
    "Italian", "Japanese", "Korean", "Malay", "Persian", "Polish",
    "Portuguese", "Russian", "Spanish", "Swahili", "Tamil", "Turkish",
    "Ukrainian", "Urdu", "Vietnamese",
]


def _pick_language(prompt: str, default: str = "English") -> str:
    print(f"\n{prompt}:")
    for i, lang in enumerate(_LANGUAGES, 1):
        print(f"  {i:2d}) {lang}")
    print("   0) Enter custom language")
    raw = _ask("Choice", default)
    if raw.isdigit():
        idx = int(raw)
        if idx == 0:
            return _ask("Custom language name", default)
        if 1 <= idx <= len(_LANGUAGES):
            return _LANGUAGES[idx - 1]
    # Treat as a typed language name
    return raw if raw else default


# ---------------------------------------------------------------------------
# Input helpers
# ---------------------------------------------------------------------------

def _ask(prompt: str, default: Optional[str] = None) -> str:
    suffix = f" [{default}]" if default is not None else ""
    val = input(f"{prompt}{suffix}: ").strip()
    return val if val else (default or "")


def _ask_int(prompt: str, default: int) -> int:
    while True:
        raw = _ask(prompt, str(default))
        try:
            return int(raw)
        except ValueError:
            print("  Please enter an integer.")


def _ask_yes(prompt: str, default: bool = False) -> bool:
    d = "y" if default else "n"
    return _ask(f"{prompt} (y/n)", d).lower().startswith("y")


def _ask_password() -> str:
    while True:
        pw = getpass("Password (hidden): ")
        if pw:
            return pw
        print("  Password cannot be empty.")


def _normalize_out_image_path(out_path: Path, cover_path: Path) -> Path:
    raw = str(out_path)
    # Allow typing just an extension: "png", ".png", etc.
    if raw.lower().lstrip(".") in ("png", "bmp", "tif", "tiff"):
        ext = raw.lower().lstrip(".")
        return cover_path.with_name(cover_path.stem + f"_stego.{ext}").resolve()
    if out_path.exists() and out_path.is_dir():
        return (out_path / (cover_path.stem + "_stego.png")).resolve()
    if out_path.suffix == "":
        return out_path.with_suffix(".png").resolve()
    return out_path.resolve()


# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------

def _llm_pre(msg: str, mode: str, model: str, translate_to: str, tone: str) -> tuple[str, Optional[str]]:
    from .util import sha256_hex
    from .llm import preprocess_text
    orig_sha = sha256_hex(msg)
    result = preprocess_text(msg, mode=mode, model=model, translate_to=translate_to, tone=tone)
    return result, orig_sha


def _llm_post(msg: str, mode: str, model: str, translate_to: str, tone: str) -> str:
    from .llm import postprocess_text
    return postprocess_text(msg, mode=mode, model=model, translate_to=translate_to, tone=tone)


def _prompt_llm_settings(pre: bool) -> tuple[bool, str, str, str, str]:
    """Prompt for LLM settings.  Returns (enabled, mode, model, translate_to, tone)."""
    direction = "pre-processing (before encryption)" if pre else "post-processing (after decryption)"
    enabled = _ask_yes(f"Use LLM {direction}? [LOSSY]", False)
    if not enabled:
        return False, "", "llama3", "English", "professional"

    mode_choices = (
        ["compress", "paraphrase", "professionalize", "summarize", "translate"]
        if pre else
        ["summarize", "paraphrase", "translate"]
    )
    print(f"  LLM modes: {', '.join(mode_choices)}")
    mode = _ask("Mode", mode_choices[0])
    if mode not in mode_choices:
        print(f"  Invalid mode; defaulting to {mode_choices[0]}.")
        mode = mode_choices[0]

    translate_to = "English"
    if mode == "translate":
        translate_to = _pick_language("Translate to", "English")

    tone = _ask("Tone (professional/casual)", "professional")

    from .llm import list_models
    models = list_models()
    if models:
        print("  Installed Ollama models: " + ", ".join(models[:10]))
    model = _ask("Ollama model", "llama3")

    return True, mode, model, translate_to, tone


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

@dataclass
class WizardState:
    last_in:  Optional[Path] = None
    last_out: Optional[Path] = None


# ---------------------------------------------------------------------------
# PNG encrypt / decrypt
# ---------------------------------------------------------------------------

def _encrypt_png(state: WizardState) -> None:
    print("\n─── Encrypt → PNG stego ───")
    in_path = Path(_ask("Cover image path", str(state.last_in) if state.last_in else None)).expanduser().resolve()
    if not in_path.exists():
        print(f"  ERROR: {in_path} not found.")
        return

    default_out = str(in_path.with_name(in_path.stem + "_stego.png"))
    out_path    = _normalize_out_image_path(
        Path(_ask("Output stego image path", default_out)).expanduser(), in_path
    )

    msg         = _ask("Message to hide")
    pw          = _ask_password()
    ecc         = _ask_int("ECC repetition factor (1–9)", 1)
    compression = _ask("Compression (none/zlib)", "zlib")
    kdf_profile = _ask("KDF profile (interactive/sensitive)", "interactive")

    llm_on, llm_mode, llm_model, translate_to, tone = _prompt_llm_settings(pre=True)

    msg2: str = msg
    orig_sha: Optional[str] = None
    if llm_on:
        try:
            msg2, orig_sha = _llm_pre(msg, llm_mode, llm_model, translate_to, tone)
        except Exception as exc:
            print(f"  LLM error: {exc}\n  Proceeding with original message.")

    try:
        payload_raw = encrypt_text(msg2, pw, compression=compression,
                                   kdf_profile=kdf_profile, orig_sha256=orig_sha)
        blob        = pack(payload_raw)
        cap         = png_capacity_bytes(Image.open(in_path).convert("RGB"), ecc=ecc)
        if len(blob) > cap:
            print(f"  ERROR: payload ({len(blob)} B) exceeds capacity ({cap} B) at ecc={ecc}.")
            return
        embed_png(in_path, out_path, blob, password=pw, ecc=ecc)
        print(f"  OK: {out_path}")
        state.last_in  = in_path
        state.last_out = out_path
    except Exception as exc:
        print(f"  ERROR: {exc}")


def _decrypt_png(state: WizardState) -> None:
    print("\n─── Decrypt ← PNG stego ───")
    in_path = Path(_ask("Stego image path", str(state.last_out) if state.last_out else None)).expanduser().resolve()
    if not in_path.exists():
        print(f"  ERROR: {in_path} not found.")
        return

    pw = _ask_password()
    llm_on, llm_mode, llm_model, translate_to, tone = _prompt_llm_settings(pre=False)

    try:
        blob        = extract_png(in_path, password=pw)
        payload_raw = unpack(blob)
        text, _, orig_sha = decrypt_text(payload_raw, pw)

        if orig_sha:
            print(f"  [audit] orig_sha256={orig_sha}", file=sys.stderr)

        if llm_on:
            try:
                text = _llm_post(text, llm_mode, llm_model, translate_to, tone)
            except Exception as exc:
                print(f"  LLM post error: {exc}\n  Showing raw decrypted text.")

        print(f"\n  Decrypted message:\n{text}\n")
        state.last_out = in_path
    except Exception as exc:
        print(f"  ERROR: {exc}")


# ---------------------------------------------------------------------------
# GIF / Video
# ---------------------------------------------------------------------------

def _encrypt_gif() -> None:
    print("\n─── Encrypt → GIF comment metadata ───")
    in_path = Path(_ask("Input GIF path")).expanduser().resolve()
    if not in_path.exists():
        print(f"  ERROR: {in_path} not found.")
        return
    out_path = Path(_ask("Output GIF path", str(in_path.with_name(in_path.stem + "_stego.gif")))).expanduser().resolve()
    msg         = _ask("Message to hide")
    pw          = _ask_password()
    compression = _ask("Compression (none/zlib)", "zlib")
    try:
        payload_raw = encrypt_text(msg, pw, compression=compression)
        embed_gif_comment(in_path, out_path, pack(payload_raw))
        print(f"  OK: {out_path}  [GIF comment metadata]")
    except Exception as exc:
        print(f"  ERROR: {exc}")


def _decrypt_gif() -> None:
    print("\n─── Decrypt ← GIF comment metadata ───")
    in_path = Path(_ask("Input GIF path")).expanduser().resolve()
    if not in_path.exists():
        print(f"  ERROR: {in_path} not found.")
        return
    pw = _ask_password()
    try:
        blob        = extract_gif_comment(in_path)
        payload_raw = unpack(blob)
        text, _, orig_sha = decrypt_text(payload_raw, pw)
        if orig_sha:
            print(f"  [audit] orig_sha256={orig_sha}", file=sys.stderr)
        print(f"\n  Decrypted message:\n{text}\n")
    except Exception as exc:
        print(f"  ERROR: {exc}")


def _encrypt_video() -> None:
    print("\n─── Encrypt → Video metadata comment ───")
    in_path = Path(_ask("Input video path")).expanduser().resolve()
    if not in_path.exists():
        print(f"  ERROR: {in_path} not found.")
        return
    out_path = Path(_ask("Output video path",
        str(in_path.with_name(in_path.stem + "_stego" + in_path.suffix)))).expanduser().resolve()
    msg         = _ask("Message to hide")
    pw          = _ask_password()
    compression = _ask("Compression (none/zlib)", "zlib")
    try:
        payload_raw = encrypt_text(msg, pw, compression=compression)
        embedded, sc = embed_video_metadata(in_path, out_path, pack(payload_raw))
        print(f"  OK: {out_path}")
        print("  Embedded in video metadata." if embedded else "  Used sidecar fallback.")
        if sc:
            print(f"  Sidecar: {sc}")
    except Exception as exc:
        print(f"  ERROR: {exc}")


def _decrypt_video() -> None:
    print("\n─── Decrypt ← Video metadata ───")
    in_path = Path(_ask("Input video path")).expanduser().resolve()
    if not in_path.exists():
        print(f"  ERROR: {in_path} not found.")
        return
    pw = _ask_password()
    try:
        blob        = extract_video_metadata(in_path)
        payload_raw = unpack(blob)
        text, _, orig_sha = decrypt_text(payload_raw, pw)
        if orig_sha:
            print(f"  [audit] orig_sha256={orig_sha}", file=sys.stderr)
        print(f"\n  Decrypted message:\n{text}\n")
    except Exception as exc:
        print(f"  ERROR: {exc}")


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def wizard_main() -> int:
    state = WizardState()
    menu = [
        ("Encrypt → PNG (recommended)", _encrypt_png),
        ("Decrypt ← PNG",               _decrypt_png),
        ("Encrypt → GIF metadata",      lambda: _encrypt_gif()),
        ("Decrypt ← GIF metadata",      lambda: _decrypt_gif()),
        ("Encrypt → Video metadata",    lambda: _encrypt_video()),
        ("Decrypt ← Video metadata",    lambda: _decrypt_video()),
    ]
    while True:
        print("\n╔══════════════════════════════╗")
        print("║  StegCrypt-AI  v3  Wizard    ║")
        print("╚══════════════════════════════╝")
        for i, (label, _) in enumerate(menu, 1):
            print(f"  {i}) {label}")
        print("  0) Exit")
        choice = _ask("Choose", "0")
        if choice == "0":
            print("Bye.")
            return 0
        if choice.isdigit() and 1 <= int(choice) <= len(menu):
            fn = menu[int(choice) - 1][1]
            # PNG actions take state; others don't
            try:
                if int(choice) in (1, 2):
                    fn(state)
                else:
                    fn()
            except KeyboardInterrupt:
                print("\n  Cancelled.")
        else:
            print("  Invalid choice.")


if __name__ == "__main__":
    raise SystemExit(wizard_main())
