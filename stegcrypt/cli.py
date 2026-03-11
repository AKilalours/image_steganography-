"""
stegcrypt.cli
~~~~~~~~~~~~~
Command-line interface for StegCrypt-AI v3.
"""

from __future__ import annotations

import argparse
import sys
from getpass import getpass
from pathlib import Path
from typing import Optional

from PIL import Image

from .crypto import encrypt_text, decrypt_text
from .payload import pack, unpack
from .png_stego import embed_png, extract_png, png_capacity_bytes
from .gif_meta import embed_gif_comment, extract_gif_comment
from .video_meta import embed_video_metadata, extract_video_metadata


def _resolve(path: str) -> Path:
    return Path(path).expanduser().resolve()


def _read_password(args: argparse.Namespace) -> str:
    return args.password if args.password is not None else getpass("Password (hidden): ")


def _llm_pre(args: argparse.Namespace, msg: str) -> tuple[str, Optional[str]]:
    if getattr(args, "llm_mode", None) is None:
        return msg, None
    from .util import sha256_hex
    sha = sha256_hex(msg)
    from .llm import preprocess_text
    return preprocess_text(msg, mode=args.llm_mode, model=args.llm_model), sha


def _llm_post(args: argparse.Namespace, msg: str) -> str:
    if getattr(args, "llm_post", None) is None:
        return msg
    from .llm import postprocess_text
    return postprocess_text(msg, mode=args.llm_post, model=args.llm_model)


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------

def cmd_encrypt_png(args: argparse.Namespace) -> int:
    pw  = _read_password(args)
    msg = args.message if args.message is not None else sys.stdin.read()
    msg2, sha_orig = _llm_pre(args, msg)

    payload_raw = encrypt_text(
        msg2, pw,
        compression=args.compression,
        kdf_profile=args.kdf_profile,
        orig_sha256=sha_orig,
    )
    blob     = pack(payload_raw)
    in_path  = _resolve(args.in_path)
    out_path = _resolve(args.out_path)

    cap = png_capacity_bytes(Image.open(in_path).convert("RGB"), ecc=args.ecc)
    if len(blob) > cap:
        raise SystemExit(
            f"Payload too large: need {len(blob)} B, image capacity {cap} B at ecc={args.ecc}."
        )

    embed_png(in_path, out_path, blob, password=pw, ecc=args.ecc)
    print(f"OK: {out_path}")
    return 0


def cmd_decrypt_png(args: argparse.Namespace) -> int:
    pw      = _read_password(args)
    in_path = _resolve(args.in_path)

    blob        = extract_png(in_path, password=pw)
    payload_raw = unpack(blob)
    text, _, orig_sha = decrypt_text(payload_raw, pw)

    text2 = _llm_post(args, text)

    if orig_sha:
        print(f"[audit] orig_sha256={orig_sha}", file=sys.stderr)
    print(text2)
    return 0


def cmd_encrypt_gif(args: argparse.Namespace) -> int:
    pw  = _read_password(args)
    msg = args.message if args.message is not None else sys.stdin.read()
    msg2, sha_orig = _llm_pre(args, msg)

    payload_raw = encrypt_text(msg2, pw, compression=args.compression, orig_sha256=sha_orig)
    blob = pack(payload_raw)
    embed_gif_comment(_resolve(args.in_path), _resolve(args.out_path), blob)
    print(f"OK: {_resolve(args.out_path)}  [GIF comment metadata]")
    return 0


def cmd_decrypt_gif(args: argparse.Namespace) -> int:
    pw          = _read_password(args)
    blob        = extract_gif_comment(_resolve(args.in_path))
    payload_raw = unpack(blob)
    text, _, orig_sha = decrypt_text(payload_raw, pw)
    text2 = _llm_post(args, text)
    if orig_sha:
        print(f"[audit] orig_sha256={orig_sha}", file=sys.stderr)
    print(text2)
    return 0


def cmd_encrypt_video(args: argparse.Namespace) -> int:
    pw  = _read_password(args)
    msg = args.message if args.message is not None else sys.stdin.read()
    msg2, sha_orig = _llm_pre(args, msg)

    payload_raw = encrypt_text(msg2, pw, compression=args.compression, orig_sha256=sha_orig)
    blob = pack(payload_raw)
    embedded, sc = embed_video_metadata(_resolve(args.in_path), _resolve(args.out_path), blob)
    print(f"OK: {_resolve(args.out_path)}")
    print("Embedded in video metadata." if embedded else "Metadata embed failed; used sidecar.")
    if sc:
        print(f"Sidecar: {sc}")
    return 0


def cmd_decrypt_video(args: argparse.Namespace) -> int:
    pw          = _read_password(args)
    blob        = extract_video_metadata(_resolve(args.in_path))
    payload_raw = unpack(blob)
    text, _, orig_sha = decrypt_text(payload_raw, pw)
    text2 = _llm_post(args, text)
    if orig_sha:
        print(f"[audit] orig_sha256={orig_sha}", file=sys.stderr)
    print(text2)
    return 0


def cmd_detect(args: argparse.Namespace) -> int:
    from .cnn import StegoCNN
    model = StegoCNN(weights_path=Path(args.weights))
    prob  = model.detect_stego(Image.open(_resolve(args.in_path)))
    print(f"P(stego)={prob:.4f}  P(cover)={1-prob:.4f}")
    return 0


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def _add_llm_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--llm-mode",  choices=["compress", "paraphrase", "professionalize",
                                            "summarize", "translate"], default=None,
                   help="Pre-encryption LLM transform (LOSSY). Requires Ollama.")
    p.add_argument("--llm-post",  choices=["summarize", "paraphrase", "translate"], default=None,
                   help="Post-decryption LLM transform. Requires Ollama.")
    p.add_argument("--llm-model", default="llama3", help="Ollama model name.")


def _add_crypto_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--password",    default=None)
    p.add_argument("--compression", choices=["none", "zlib"], default="zlib")
    p.add_argument("--kdf-profile", choices=["interactive", "sensitive"],
                   default="interactive",
                   help="KDF strength: 'interactive' (fast) or 'sensitive' (~1 s).")


def build_parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="stegcrypt", description="StegCrypt-AI v3 CLI")
    sub  = root.add_subparsers(dest="cmd", required=True)

    # encrypt-png / encrypt-image
    for name, help_text in [
        ("encrypt-png",   "Encrypt + embed into lossless image (PNG/BMP/TIFF) [randomized LSB]."),
        ("encrypt-image", "Alias for encrypt-png."),
    ]:
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--in",  dest="in_path",  required=True)
        p.add_argument("--out", dest="out_path", required=True)
        p.add_argument("--message", default=None)
        p.add_argument("--ecc", type=int, default=1, help="Bit repetition ECC factor (1–9).")
        _add_crypto_args(p); _add_llm_args(p)
        p.set_defaults(func=cmd_encrypt_png)

    # decrypt-png / decrypt-image
    for name, help_text in [
        ("decrypt-png",   "Extract + decrypt from lossless stego image."),
        ("decrypt-image", "Alias for decrypt-png."),
    ]:
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--in", dest="in_path", required=True)
        _add_crypto_args(p); _add_llm_args(p)
        p.set_defaults(func=cmd_decrypt_png)

    # encrypt/decrypt gif
    eg = sub.add_parser("encrypt-gif", help="Encrypt + embed into GIF comment metadata.")
    eg.add_argument("--in", dest="in_path", required=True)
    eg.add_argument("--out", dest="out_path", required=True)
    eg.add_argument("--message", default=None)
    _add_crypto_args(eg); _add_llm_args(eg)
    eg.set_defaults(func=cmd_encrypt_gif)

    dg = sub.add_parser("decrypt-gif", help="Extract + decrypt from GIF comment metadata.")
    dg.add_argument("--in", dest="in_path", required=True)
    _add_crypto_args(dg); _add_llm_args(dg)
    dg.set_defaults(func=cmd_decrypt_gif)

    # encrypt/decrypt video
    ev = sub.add_parser("encrypt-video", help="Encrypt + embed into video metadata comment.")
    ev.add_argument("--in", dest="in_path", required=True)
    ev.add_argument("--out", dest="out_path", required=True)
    ev.add_argument("--message", default=None)
    _add_crypto_args(ev); _add_llm_args(ev)
    ev.set_defaults(func=cmd_encrypt_video)

    dv = sub.add_parser("decrypt-video", help="Extract + decrypt from video metadata.")
    dv.add_argument("--in", dest="in_path", required=True)
    _add_crypto_args(dv); _add_llm_args(dv)
    dv.set_defaults(func=cmd_decrypt_video)

    # detect
    det = sub.add_parser("detect", help="Run CNN stego detector on an image.")
    det.add_argument("--in",      dest="in_path", required=True)
    det.add_argument("--weights", required=True, help="Path to stego_cnn.pth")
    det.set_defaults(func=cmd_detect)

    return root


def main(argv: Optional[list[str]] = None) -> None:
    if argv is None:
        argv = sys.argv[1:]
    if not argv:
        from .wizard import wizard_main
        raise SystemExit(wizard_main())

    parser = build_parser()
    args   = parser.parse_args(argv)
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
