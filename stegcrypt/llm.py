"""
stegcrypt.llm
~~~~~~~~~~~~~
Optional LLM pre/post-processing transforms via a local Ollama instance.

All transforms are explicitly LOSSY: we store the SHA-256 of the original
plaintext in the crypto envelope so the receiver can detect drift.

Supported modes
---------------
compress       – shorten while preserving meaning
paraphrase     – rewrite preserving all key details
professionalize – rewrite in professional, concise language
summarize      – 1–2 sentence summary
translate      – translate to a target language
"""

from __future__ import annotations

import logging
from typing import Literal, Optional

logger = logging.getLogger(__name__)

Mode = Literal["compress", "paraphrase", "professionalize", "summarize", "translate"]

_PREAMBLE_PREFIXES = (
    "here is", "here's", "translation:", "translated text:",
    "note:", "sure,", "okay,", "certainly,", "of course,",
    "below is", "the following",
)


def _import_ollama():
    try:
        import ollama  # type: ignore
        return ollama
    except ImportError as exc:
        raise RuntimeError(
            "Ollama client not installed.  Run: pip install -e '.[llm]'"
        ) from exc


def list_models() -> list[str]:
    """Return installed Ollama model names, or empty list on failure."""
    try:
        ollama = _import_ollama()
        data = ollama.list()
        models = data.get("models", []) if isinstance(data, dict) else []
        return [m["name"] for m in models if isinstance(m, dict) and m.get("name")]
    except Exception:
        return []


def _call_ollama(prompt: str, model: str) -> str:
    ollama = _import_ollama()
    try:
        res = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
        content = (res.get("message") or {}).get("content", "")
        return content.strip()
    except Exception as exc:
        raise RuntimeError(
            f"Ollama call failed.  Ensure Ollama is running and "
            f"`ollama pull {model}` has been executed.\nCause: {exc}"
        ) from exc


def _strip_preamble(text: str) -> str:
    """Remove common LLM preamble lines that appear despite prompt instructions."""
    lines = [ln.strip() for ln in text.splitlines()]
    # Drop leading blank lines
    while lines and not lines[0]:
        lines.pop(0)
    # Drop a single preamble line if present
    if lines and any(lines[0].lower().startswith(p) for p in _PREAMBLE_PREFIXES):
        lines.pop(0)
        while lines and not lines[0]:
            lines.pop(0)
    return "\n".join(lines).strip()


def _build_prompt(
    text: str,
    mode: Mode,
    tone: str,
    translate_to: str,
) -> str:
    rules = (
        "Rules: Output ONLY the transformed text. "
        "No preface, no commentary, no quotes, no extra lines."
    )
    if mode == "compress":
        return (
            f"Task: shorten the following message while preserving all meaning.\n"
            f"Tone: {tone}.\n{rules}\n\n{text}"
        )
    if mode == "paraphrase":
        return (
            f"Task: paraphrase the following message, preserving all key details.\n"
            f"Tone: {tone}.\n{rules}\n\n{text}"
        )
    if mode == "professionalize":
        return (
            f"Task: rewrite the following message to sound professional, clear, and concise.\n"
            f"Do NOT add greetings or signatures unless already present.\n"
            f"{rules}\n\n{text}"
        )
    if mode == "summarize":
        return (
            f"Task: summarize the following message in 1–2 sentences.\n"
            f"Tone: {tone}.\n{rules}\n\n{text}"
        )
    if mode == "translate":
        return (
            f"Task: translate the following message into {translate_to}.\n"
            f"Tone: {tone}.\n"
            f"Output ONLY the translated text in {translate_to}. "
            f"No explanations, no 'here is the translation', no quotes.\n\n{text}"
        )
    return text  # passthrough for unknown modes


def preprocess_text(
    raw_text: str,
    *,
    mode: Mode = "compress",
    model: str = "llama3",
    translate_to: str = "English",
    tone: str = "professional",
) -> str:
    """Apply an LLM transform to *raw_text* before encryption."""
    prompt = _build_prompt(raw_text, mode, tone, translate_to)
    result = _call_ollama(prompt, model)
    return _strip_preamble(result) or raw_text  # fallback to original if LLM returns empty


def postprocess_text(
    text: str,
    *,
    mode: Mode = "summarize",
    model: str = "llama3",
    translate_to: str = "English",
    tone: str = "professional",
) -> str:
    """Apply an LLM transform to *text* after decryption."""
    return preprocess_text(
        text,
        mode=mode,
        model=model,
        translate_to=translate_to,
        tone=tone,
    )
