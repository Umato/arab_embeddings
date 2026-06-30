from __future__ import annotations

import copy
import importlib
import sys
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer


DEFAULT_ARABIC_NORMALIZER_CONFIG = {
    "unicode_normalization": "NFKC",
    "remove_diacritics": True,
    "lowercase_latin": False,
    "preserve_alif_variants": True,
}

DEFAULT_ARABIC_PRETOKENIZER_CONFIG = {
    "with_prefix_space": False,
    "add_prefix_space": False,
    "fuse_numeric_sequences": False,
}


def _ensure_tokenizer_arabic_importable(tokenizer_arabic_src_path: str | None = None):
    """Make the local tokenizer_arabic package importable when present in the repo."""
    search_paths: list[Path] = []
    if tokenizer_arabic_src_path is not None:
        search_paths.append(Path(tokenizer_arabic_src_path).expanduser().resolve())

    repo_root = Path(__file__).resolve().parents[2]
    search_paths.append(repo_root / "demo" / "tokenizer" / "src")

    for path in search_paths:
        if path.exists():
            str_path = str(path)
            if str_path not in sys.path:
                sys.path.insert(0, str_path)

    try:
        return importlib.import_module("tokenizer_arabic")
    except ModuleNotFoundError as exc:
        searched = ", ".join(str(path) for path in search_paths)
        raise ModuleNotFoundError(
            "Could not import local `tokenizer_arabic` helpers. "
            f"Searched: {searched}. Pass `tokenizer_arabic_src_path` if your local path differs."
        ) from exc


def apply_custom_arabic_tokenizer_components(
    tokenizer,
    *,
    normalizer_config: dict[str, Any] | None = None,
    pretokenizer_config: dict[str, Any] | None = None,
    tokenizer_arabic_src_path: str | None = None,
):
    """Apply the local Arabic normalizer and pre-tokenizer to a tokenizer instance."""
    tokenizer_arabic = _ensure_tokenizer_arabic_importable(tokenizer_arabic_src_path)

    normalizer_config = normalizer_config or DEFAULT_ARABIC_NORMALIZER_CONFIG
    pretokenizer_config = pretokenizer_config or DEFAULT_ARABIC_PRETOKENIZER_CONFIG

    normalizer_cfg = tokenizer_arabic.ArabicNormalizerConfig(**normalizer_config)
    pretokenizer_cfg = tokenizer_arabic.ArabicPreTokenizerConfig(**pretokenizer_config)

    tokenizer.backend_tokenizer.normalizer = tokenizer_arabic.build_qwen_like_normalizer(normalizer_cfg)
    tokenizer.backend_tokenizer.pre_tokenizer = tokenizer_arabic.build_qwen_like_pretokenizer(pretokenizer_cfg)
    return tokenizer


def load_custom_tokenizer(
    tokenizer_path: str,
    *,
    normalizer_config: dict[str, Any] | None = None,
    pretokenizer_config: dict[str, Any] | None = None,
    tokenizer_arabic_src_path: str | None = None,
):
    """Load a tokenizer and apply the repo-local Arabic normalization/pre-tokenization pipeline."""
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    return apply_custom_arabic_tokenizer_components(
        tokenizer,
        normalizer_config=normalizer_config,
        pretokenizer_config=pretokenizer_config,
        tokenizer_arabic_src_path=tokenizer_arabic_src_path,
    )


def clone_tokenizer_with_custom_arabic_components(
    tokenizer,
    *,
    normalizer_config: dict[str, Any] | None = None,
    pretokenizer_config: dict[str, Any] | None = None,
    tokenizer_arabic_src_path: str | None = None,
):
    """Deep-copy a tokenizer and apply the Arabic normalizer/pre-tokenizer to the copy."""
    cloned = copy.deepcopy(tokenizer)
    return apply_custom_arabic_tokenizer_components(
        cloned,
        normalizer_config=normalizer_config,
        pretokenizer_config=pretokenizer_config,
        tokenizer_arabic_src_path=tokenizer_arabic_src_path,
    )
