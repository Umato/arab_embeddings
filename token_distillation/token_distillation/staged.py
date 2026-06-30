from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from .ahocorasick import collect_snippets_with_patterns_from_dataset, map_int_seq_to_str

LABEL_PRESETS: dict[str, dict[str, list[str]]] = {
    "cw_name": {
        "distill": ["clean_content_word", "proper_name_or_foreign"],
        "support": [
            "fragment_or_affix",
            "clean_function_word",
            "attached_function_plus_content",
            "ambiguous",
        ],
        "drop": ["nonword_or_noise"],
    },
    "cw_name_afc": {
        "distill": [
            "clean_content_word",
            "proper_name_or_foreign",
            "attached_function_plus_content",
        ],
        "support": [
            "fragment_or_affix",
            "clean_function_word",
            "ambiguous",
        ],
        "drop": ["nonword_or_noise"],
    },
    "cw_name_afc_fw": {
        "distill": [
            "clean_content_word",
            "proper_name_or_foreign",
            "attached_function_plus_content",
            "clean_function_word",
        ],
        "support": [
            "fragment_or_affix",
            "ambiguous",
        ],
        "drop": ["nonword_or_noise"],
    },
    "all_except_noise": {
        "distill": [
            "clean_content_word",
            "proper_name_or_foreign",
            "attached_function_plus_content",
            "clean_function_word",
            "fragment_or_affix",
            "ambiguous",
        ],
        "support": [],
        "drop": ["nonword_or_noise"],
    },
}


@dataclass
class SelectedTokenSets:
    """Selected token groups ordered by the custom tokenizer's token ids."""

    ordered_added_tokens: list[str]
    distill_tokens: list[str]
    support_tokens: list[str]
    dropped_tokens: list[str]
    token_to_label: dict[str, str]
    token_to_custom_id: dict[str, int]


@dataclass
class StageTextSnippetCollection:
    """Final-tokenizer snippet collection for one staged distillation step."""

    token_to_pattern_ids: dict[str, list[int]]
    token_to_snippets: dict[str, list[list[int]]]
    skipped: list[tuple[str, int, int]]
    docs_scanned: int
    snippets_collected_per_token: dict[str, int]


def load_label_records(labels_jsonl_path: str | Path) -> list[dict[str, Any]]:
    """Load classification records from JSONL."""
    path = Path(labels_jsonl_path)
    records: list[dict[str, Any]] = []
    with path.open() as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def resolve_label_selection(
    *,
    preset_name: str | None,
    distill_labels: Sequence[str] | None,
    support_labels: Sequence[str] | None,
    drop_labels: Sequence[str] | None,
) -> tuple[set[str], set[str], set[str]]:
    """Resolve explicit label groups, optionally seeded by a named preset."""
    preset = LABEL_PRESETS.get(preset_name or "", {})
    resolved_distill = set(distill_labels or preset.get("distill", []))
    resolved_support = set(support_labels or preset.get("support", []))
    resolved_drop = set(drop_labels or preset.get("drop", []))

    overlap = (resolved_distill & resolved_support) | (resolved_distill & resolved_drop) | (resolved_support & resolved_drop)
    if overlap:
        raise ValueError(f"Label groups must be disjoint, overlap found: {sorted(overlap)}")
    return resolved_distill, resolved_support, resolved_drop


def select_tokens_from_label_records(
    records: Sequence[dict[str, Any]],
    *,
    base_tokenizer,
    custom_tokenizer,
    distill_labels: Sequence[str],
    support_labels: Sequence[str],
    drop_labels: Sequence[str],
) -> SelectedTokenSets:
    """Choose distill/support/drop tokens using canonical `token` strings from the labels file."""
    base_vocab = base_tokenizer.get_vocab()
    custom_vocab = custom_tokenizer.get_vocab()

    distill_label_set = set(distill_labels)
    support_label_set = set(support_labels)
    drop_label_set = set(drop_labels)

    selected_rows: list[tuple[int, str, str]] = []
    dropped_tokens: list[str] = []
    token_to_label: dict[str, str] = {}
    token_to_custom_id: dict[str, int] = {}
    seen_tokens: set[str] = set()

    for record in records:
        token = record["token"]
        label_name = record["label_name"]
        if token in seen_tokens:
            continue
        seen_tokens.add(token)

        custom_token_id = custom_vocab.get(token)
        if custom_token_id is None:
            continue
        if token in base_vocab:
            continue

        if label_name in drop_label_set:
            dropped_tokens.append(token)
            token_to_label[token] = label_name
            token_to_custom_id[token] = custom_token_id
            continue
        if label_name not in distill_label_set and label_name not in support_label_set:
            continue

        selected_rows.append((custom_token_id, token, label_name))
        token_to_label[token] = label_name
        token_to_custom_id[token] = custom_token_id

    selected_rows.sort(key=lambda row: row[0])
    ordered_added_tokens = [token for _, token, _ in selected_rows]
    distill_tokens = [token for _, token, label in selected_rows if label in distill_label_set]
    support_tokens = [token for _, token, label in selected_rows if label in support_label_set]

    return SelectedTokenSets(
        ordered_added_tokens=ordered_added_tokens,
        distill_tokens=distill_tokens,
        support_tokens=support_tokens,
        dropped_tokens=dropped_tokens,
        token_to_label=token_to_label,
        token_to_custom_id=token_to_custom_id,
    )


def build_tokenizer_with_ordered_added_tokens(base_tokenizer, ordered_added_tokens: Sequence[str], included_tokens: set[str] | None = None):
    """Copy a tokenizer and add a filtered subset of tokens in the shared master order."""
    tokenizer = copy.deepcopy(base_tokenizer)
    if included_tokens is None:
        tokens_to_add = list(ordered_added_tokens)
    else:
        tokens_to_add = [token for token in ordered_added_tokens if token in included_tokens]
    if tokens_to_add:
        tokenizer.add_tokens(tokens_to_add)
    return tokenizer


def get_explicit_token_ids(tokenizer) -> list[int]:
    """Return the tokenizer's explicit token ids without assuming contiguity."""
    return sorted(set(int(token_id) for token_id in tokenizer.get_vocab().values()))


def get_token_id_map(tokenizer, tokens: Sequence[str]) -> dict[str, int]:
    """Map token strings to explicit tokenizer ids."""
    return {token: int(tokenizer.convert_tokens_to_ids(token)) for token in tokens}


def remap_teacher_token_ids_to_final_ids(
    teacher_token_ids: Sequence[int],
    *,
    teacher_tokenizer,
    final_tokenizer,
) -> list[int]:
    """Map teacher-local token ids into the corresponding ids of the final tokenizer."""
    teacher_tokens = teacher_tokenizer.convert_ids_to_tokens(list(teacher_token_ids))
    if isinstance(teacher_tokens, str):
        teacher_tokens = [teacher_tokens]

    final_ids: list[int] = []
    missing_tokens: list[str] = []
    for token in teacher_tokens:
        final_token_id = final_tokenizer.convert_tokens_to_ids(token)
        if final_token_id is None or final_token_id == getattr(final_tokenizer, "unk_token_id", None):
            if token not in final_tokenizer.get_vocab():
                missing_tokens.append(token)
                continue
        final_ids.append(int(final_token_id))

    if missing_tokens:
        raise KeyError(
            "Teacher tokenizer produced tokens that do not exist in the final tokenizer: "
            f"{missing_tokens}"
        )
    return final_ids


def tokenize_with_teacher_tokenizer_to_final_ids(
    text: str,
    *,
    teacher_tokenizer,
    final_tokenizer,
    model_path: str,
    tokenize_text_fn: Callable[[str, Any, str], list[int]] | None = None,
) -> list[int]:
    """Tokenize with the stage teacher tokenizer, then remap the resulting token pieces into final-tokenizer ids."""
    tokenize_text_fn = tokenize_text_fn or default_teacher_tokenize_token
    teacher_token_ids = [int(token_id) for token_id in tokenize_text_fn(text, teacher_tokenizer, model_path)]
    return remap_teacher_token_ids_to_final_ids(
        teacher_token_ids,
        teacher_tokenizer=teacher_tokenizer,
        final_tokenizer=final_tokenizer,
    )


def default_teacher_tokenize_token(token: str, teacher_tokenizer, model_path: str) -> list[int]:
    return [int(x) for x in teacher_tokenizer(token, add_special_tokens=False)["input_ids"]]


def _truncate_snippet_around_pattern(
    snippet_tokens: Sequence[int],
    *,
    pattern_start_idx: int,
    pattern_len: int,
    context_length: int,
) -> list[int] | None:
    """Trim a matched snippet to the requested context length while keeping the full pattern."""
    if pattern_len > context_length:
        return None

    available_before = pattern_start_idx
    available_after = len(snippet_tokens) - pattern_start_idx - pattern_len
    remaining = context_length - pattern_len

    before = min(available_before, remaining // 2)
    after = min(available_after, remaining - before)

    if before + after < remaining:
        extra_after = min(available_after - after, remaining - before - after)
        after += extra_after
    if before + after < remaining:
        extra_before = min(available_before - before, remaining - before - after)
        before += extra_before

    start = pattern_start_idx - before
    end = pattern_start_idx + pattern_len + after
    return [int(token_id) for token_id in snippet_tokens[start:end]]


def collect_stage_token_snippets_from_texts(
    *,
    texts: Iterable[str],
    todo_tokens: Sequence[tuple[str, int]],
    teacher_tokenizer,
    final_tokenizer,
    model_path: str,
    context_length: int,
    snippets_per_token: int,
    max_docs: int | None = None,
    tokenize_text_fn: Callable[[str, Any, str], list[int]] | None = None,
) -> StageTextSnippetCollection:
    """Collect snippets by matching in teacher-tokenized docs, then remap snippets to final-tokenizer ids."""
    tokenize_text_fn = tokenize_text_fn or default_teacher_tokenize_token

    token_to_teacher_pattern_ids: dict[str, list[int]] = {}
    token_to_final_pattern_ids: dict[str, list[int]] = {}
    teacher_patterns_ids: list[list[int]] = []

    for token, _ in todo_tokens:
        teacher_pattern_ids = [int(x) for x in tokenize_text_fn(token, teacher_tokenizer, model_path)]
        final_pattern_ids = remap_teacher_token_ids_to_final_ids(
            teacher_pattern_ids,
            teacher_tokenizer=teacher_tokenizer,
            final_tokenizer=final_tokenizer,
        )

        token_to_teacher_pattern_ids[token] = teacher_pattern_ids
        token_to_final_pattern_ids[token] = final_pattern_ids
        teacher_patterns_ids.append(teacher_pattern_ids)

    docs_scanned = 0

    def _iter_teacher_tokenized_docs():
        nonlocal docs_scanned
        for text in texts:
            if max_docs is not None and docs_scanned >= max_docs:
                break
            docs_scanned += 1
            tokenized = teacher_tokenizer(text, add_special_tokens=False, truncation=True, max_length=65536, )["input_ids"]
            if tokenized:
                yield [int(token_id) for token_id in tokenized]

    collected = collect_snippets_with_patterns_from_dataset(
        teacher_patterns_ids,
        teacher_tokenizer,
        _iter_teacher_tokenized_docs(),
        max_docs=max_docs if max_docs is not None else 2**31 - 1,
        offset_before=context_length,
        offset_after=context_length,
        batch_start=1,
        batch_max=4_096,
        max_necessary_samples=snippets_per_token * 2,
        verbose=False,
    )

    token_to_snippets: dict[str, list[list[int]]] = {}
    snippets_collected_per_token: dict[str, int] = {}
    skipped: list[tuple[str, int, int]] = []

    for token, token_id in todo_tokens:
        teacher_pattern_ids = token_to_teacher_pattern_ids[token]
        final_pattern_ids = token_to_final_pattern_ids[token]

        collected_snippets = collected.get(map_int_seq_to_str(teacher_pattern_ids), [])
        finalized_snippets: list[list[int]] = []

        for snippet_tokens_teacher, pattern_start_idx in collected_snippets:
            snippet_tokens_final = remap_teacher_token_ids_to_final_ids(
                snippet_tokens_teacher,
                teacher_tokenizer=teacher_tokenizer,
                final_tokenizer=final_tokenizer,
            )

            truncated = _truncate_snippet_around_pattern(
                snippet_tokens_final,
                pattern_start_idx=pattern_start_idx,
                pattern_len=len(final_pattern_ids),
                context_length=context_length,
            )
            if truncated is None:
                continue

            finalized_snippets.append(truncated)
            if len(finalized_snippets) >= snippets_per_token:
                break

        snippets_collected_per_token[token] = len(finalized_snippets)
        if len(finalized_snippets) < snippets_per_token:
            skipped.append((token, token_id, len(finalized_snippets)))
            continue

        token_to_snippets[token] = finalized_snippets

    return StageTextSnippetCollection(
        token_to_pattern_ids=token_to_final_pattern_ids,
        token_to_snippets=token_to_snippets,
        skipped=skipped,
        docs_scanned=docs_scanned,
        snippets_collected_per_token=snippets_collected_per_token,
    )


def compute_stage_eligibility(
    pending_distill_tokens: Sequence[str],
    *,
    teacher_tokenizer,
    model_path: str,
    tokenize_token_fn: Callable[[str, Any, str], list[int]] | None = None,
) -> dict[str, dict[str, Any]]:
    """Evaluate which pending distill tokens can be represented by the current teacher tokenizer."""
    tokenize_token_fn = tokenize_token_fn or default_teacher_tokenize_token
    teacher_vocab_ids = set(get_explicit_token_ids(teacher_tokenizer))
    unk_token_id = getattr(teacher_tokenizer, "unk_token_id", None)

    eligibility: dict[str, dict[str, Any]] = {}
    for token in pending_distill_tokens:
        teacher_ids = [int(token_id) for token_id in tokenize_token_fn(token, teacher_tokenizer, model_path)]
        missing_ids = [token_id for token_id in teacher_ids if token_id not in teacher_vocab_ids]
        uses_unk = unk_token_id is not None and unk_token_id in teacher_ids
        eligible = bool(teacher_ids) and not missing_ids and not uses_unk
        eligibility[token] = {
            "teacher_token_ids": teacher_ids,
            "eligible": eligible,
            "missing_teacher_token_ids": missing_ids,
            "uses_unk": uses_unk,
        }
    return eligibility


def build_stagewise_distill_stages(
    *,
    base_tokenizer,
    ordered_added_tokens: Sequence[str],
    support_tokens: Sequence[str],
    distill_tokens: Sequence[str],
    model_path: str,
    tokenize_token_fn: Callable[[str, Any, str], list[int]] | None = None,
) -> list[dict[str, Any]]:
    """Build iterative stage assignments from teacher-tokenizer eligibility."""
    available_support = set(support_tokens)
    pending = list(distill_tokens)
    completed: set[str] = set()
    stages: list[dict[str, Any]] = []

    while pending:
        teacher_available_tokens = available_support | completed
        teacher_tokenizer = build_tokenizer_with_ordered_added_tokens(
            base_tokenizer,
            ordered_added_tokens,
            included_tokens=teacher_available_tokens,
        )
        eligibility = compute_stage_eligibility(
            pending,
            teacher_tokenizer=teacher_tokenizer,
            model_path=model_path,
            tokenize_token_fn=tokenize_token_fn,
        )
        eligible_tokens = [token for token in pending if eligibility[token]["eligible"]]
        blocked_tokens = [token for token in pending if not eligibility[token]["eligible"]]

        stages.append(
            {
                "stage_index": len(stages) + 1,
                "teacher_available_tokens": [token for token in ordered_added_tokens if token in teacher_available_tokens],
                "eligible_tokens": eligible_tokens,
                "blocked_tokens": blocked_tokens,
                "eligibility": eligibility,
            }
        )

        if not eligible_tokens:
            break

        completed.update(eligible_tokens)
        pending = blocked_tokens

    return stages
