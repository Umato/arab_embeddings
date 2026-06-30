from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import hashlib

REPO_ROOT = Path(__file__).resolve().parent
LOCAL_PACKAGE_ROOT = REPO_ROOT / "token_distillation"
if str(LOCAL_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(LOCAL_PACKAGE_ROOT))

from token_distillation.staged import (
    build_tokenizer_with_ordered_added_tokens,
    collect_stage_token_snippets_from_texts,
    compute_stage_eligibility,
    get_explicit_token_ids,
    get_token_id_map,
    load_label_records,
    remap_teacher_token_ids_to_final_ids,
    resolve_label_selection,
    select_tokens_from_label_records,
    tokenize_with_teacher_tokenizer_to_final_ids,
)
from token_distillation.tokdist import (
    DistillationConfig,
    GeneratedDataSource,
    HFDataSource,
    OutputEmbeddingInit,
    compute_subtoken_means,
    extend_pretrained_with_tokens_and_embeddings,
)
from token_distillation.tokenizer_utils import (
    clone_tokenizer_with_custom_arabic_components,
    load_custom_tokenizer,
)
from token_distillation.train_loop import train_embeddings
from token_distillation.utils import generate_samples_with_patterns, seed_everything

def _load_embedding_tensor(path: str) -> torch.Tensor:
    obj = torch.load(path, map_location="cpu")

    if torch.is_tensor(obj):
        return obj.detach()

    if hasattr(obj, "weight") and torch.is_tensor(obj.weight):
        return obj.weight.detach()

    if isinstance(obj, dict):
        for key in ["weight", "embeddings", "input_embeddings", "new_embs", "state_dict"]:
            if key in obj:
                value = obj[key]
                if torch.is_tensor(value):
                    return value.detach()

    raise TypeError(f"Could not extract an embedding tensor from {path!r}")

def _load_optional_json(path: str | None) -> dict[str, Any] | None:
    if path is None:
        return None
    with Path(path).open() as handle:
        return json.load(handle)
    
@torch.no_grad()
def _build_input_preinit(
    *,
    model,
    teacher_base_tokenizer,
    base_model_name: str,
    all_added_tokens: list[str],
    all_added_todo: list[tuple[str, int]],
    selected,
    init_embeddings_path: str | None,
    pre_init_strategy: str,
) -> dict[str, torch.Tensor]:
    emb_weight = model.get_input_embeddings().weight

    if init_embeddings_path is None:
        return compute_subtoken_means(
            model=model,
            tokenizer=teacher_base_tokenizer,
            model_path=base_model_name,
            todo_tokens=all_added_todo,
            input_or_output="input",
            method=pre_init_strategy,
        )

    focus_matrix = _load_embedding_tensor(init_embeddings_path)

    if focus_matrix.ndim != 2:
        raise ValueError(f"Expected a 2D embedding matrix, got shape {tuple(focus_matrix.shape)}")

    if focus_matrix.shape[1] != emb_weight.shape[1]:
        raise ValueError(
            f"Embedding hidden size mismatch: init matrix has {focus_matrix.shape[1]}, "
            f"model has {emb_weight.shape[1]}"
        )

    max_needed_custom_id = max(selected.token_to_custom_id[token] for token in all_added_tokens)
    if focus_matrix.shape[0] <= max_needed_custom_id:
        raise ValueError(
            f"Init matrix has only {focus_matrix.shape[0]} rows, "
            f"but selected tokens need custom id {max_needed_custom_id}"
        )

    out = {}
    for token in all_added_tokens:
        custom_id = selected.token_to_custom_id[token]
        out[token] = focus_matrix[custom_id].to(device=emb_weight.device, dtype=emb_weight.dtype).clone()

    return out


def _default_device() -> str:
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run staged token distillation from labeled token classifications.")
    parser.add_argument("--base-model", default="Qwen/Qwen3-0.6B-Base")
    parser.add_argument("--custom-tokenizer", default="jais_qwen")
    parser.add_argument("--labels-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)

    parser.add_argument("--label-preset", choices=["cw_name", "cw_name_afc", "cw_name_afc_fw", "all_except_noise"], default="cw_name")
    parser.add_argument("--distill-labels", nargs="+")
    parser.add_argument("--support-labels", nargs="+")
    parser.add_argument("--drop-labels", nargs="+")

    parser.add_argument("--target-layers", type=int, nargs="+", default=[27])
    parser.add_argument("--sequential-layers", action="store_true")
    parser.add_argument("--max-snippets-per-token", type=int, default=32)
    parser.add_argument("--context-length", type=int, default=64)
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--snippet-source", choices=["generated", "hf"], default="generated")
    parser.add_argument("--dataset-path", default="epfml/FineWeb2-HQ")
    parser.add_argument("--dataset-name", default="arb_Arab")
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument("--min-quality-score", type=float, default=0.7)
    parser.add_argument("--dataset-streaming", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dataset-max-docs", type=int)
    parser.add_argument("--dataset-num-proc", type=int)
    parser.add_argument("--dataset-revision")
    parser.add_argument("--dataset-trust-remote-code", action="store_true")

    parser.add_argument("--pre-init-strategy", choices=["fvt", "adapti-vocab"], default="fvt")
    parser.add_argument(
        "--output-embedding-policy",
        choices=[policy.value for policy in OutputEmbeddingInit],
        default=OutputEmbeddingInit.TRAIN_WITH_CE.value,
    )
    parser.add_argument("--device", default=_default_device())
    parser.add_argument("--attn-impl", default="sdpa")
    parser.add_argument("--skip-distillation", action="store_true")

    parser.add_argument("--use-custom-tokenizer-processing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--custom-normalizer-config-json")
    parser.add_argument("--custom-pretokenizer-config-json")
    parser.add_argument("--tokenizer-arabic-src-path")

    parser.add_argument(
    "--loss-methods",
    nargs="+",
    default=["MSE-on-hiddens"],
    choices=[
        "MSE-on-hiddens",
        "MSE-on-logits",
        "KL-on-logits",
        "CE",
        "CE-auto-weighted",
    ],
    )

    parser.add_argument("--dataset-partition", choices=["train", "eval", "all"], default="train")
    parser.add_argument("--dataset-eval-percent", type=float, default=5.0)
    parser.add_argument("--dataset-hash-salt", default="arabic_td_fineweb2hq_v1")

    parser.add_argument(
        "--init-embeddings-path",
        help=(
            "Optional .pt tensor with embeddings aligned to the custom tokenizer. "
            "Rows for selected added tokens are copied by custom_tokenizer_id."
        ),
    )

    return parser.parse_args()


def _build_data_source(args: argparse.Namespace):
    if args.snippet_source == "generated":
        return GeneratedDataSource(seed=args.seed)
    return HFDataSource(
        dataset_path=args.dataset_path,
        name=args.dataset_name,
        split=args.dataset_split,
        streaming=args.dataset_streaming,
        max_docs=args.dataset_max_docs,
        num_proc=args.dataset_num_proc,
        revision=args.dataset_revision,
        trust_remote_code=args.dataset_trust_remote_code,
        min_quality_score=args.min_quality_score,
        partition=args.dataset_partition,
        eval_percent=args.dataset_eval_percent,
        hash_salt=args.dataset_hash_salt,
    )


def _snippet_source_metadata(data_source) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "snippet_source": "generated" if isinstance(data_source, GeneratedDataSource) else "hf",
    }
    if isinstance(data_source, HFDataSource):
        metadata.update(
            {
                "dataset_path": data_source.dataset_path,
                "dataset_name": data_source.name,
                "dataset_split": data_source.split,
                "dataset_streaming": data_source.streaming,
                "dataset_max_docs": data_source.max_docs,
                "dataset_num_proc": data_source.num_proc,
                "min_quality_score": data_source.min_quality_score,
                "dataset_partition": data_source.partition,
                "dataset_eval_percent": data_source.eval_percent,
                "dataset_hash_salt": data_source.hash_salt,
            }
        )
    return metadata


def _stable_text_bucket(text: str, *, salt: str, buckets: int = 10_000) -> int:
    normalized = " ".join(text.split())
    digest = hashlib.sha256((salt + "\n" + normalized).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % buckets


def _dataset_row_is_eligible(
    row: dict[str, Any],
    *,
    text_field: str,
    quality_score_field: str,
    min_quality_score: float | None,
    partition: str = "train",
    eval_percent: float = 5.0,
    hash_salt: str = "arabic_td_fineweb2hq_v1",
) -> bool:
    text = row.get(text_field)
    if not isinstance(text, str) or not text.strip():
        return False

    if min_quality_score is not None:
        quality_score = row.get(quality_score_field)
        try:
            if quality_score is None or float(quality_score) < min_quality_score:
                return False
        except (TypeError, ValueError):
            return False

    if partition == "all":
        return True

    buckets = 10_000
    cutoff = int(buckets * eval_percent / 100.0)
    bucket = _stable_text_bucket(text, salt=hash_salt, buckets=buckets)

    is_eval = bucket < cutoff

    if partition == "eval":
        return is_eval
    if partition == "train":
        return not is_eval

    raise ValueError(f"Unknown partition: {partition}")


def _iter_texts_from_hf_dataset(data_source: HFDataSource):
    from datasets import load_dataset

    dataset = load_dataset(
        data_source.dataset_path,
        name=data_source.name,
        split=data_source.split,
        streaming=data_source.streaming,
        revision=data_source.revision,
        trust_remote_code=data_source.trust_remote_code,
    )

    if not data_source.streaming:
        dataset = dataset.filter(
            _dataset_row_is_eligible,
            fn_kwargs={
                "text_field": data_source.text_field,
                "quality_score_field": data_source.quality_score_field,
                "min_quality_score": data_source.min_quality_score,
                "partition": data_source.partition,
                "eval_percent": data_source.eval_percent,
                "hash_salt": data_source.hash_salt,
            },
            num_proc=data_source.num_proc,
        )
        for row in dataset:
            yield row[data_source.text_field]
        return

    for row in dataset:
        if _dataset_row_is_eligible(
            row,
            text_field=data_source.text_field,
            quality_score_field=data_source.quality_score_field,
            min_quality_score=data_source.min_quality_score,
            partition=data_source.partition,
            eval_percent=data_source.eval_percent,
            hash_salt=data_source.hash_salt,
        ):
            yield row[data_source.text_field]


def _train_one_stage(
    *,
    model,
    final_tokenizer,
    teacher_tokenizer,
    base_model_name: str,
    target_layer_schedule: list[list[int]],
    stage_tokens: list[str],
    token_to_final_id: dict[str, int],
    snippets_per_token: int,
    context_length: int,
    data_source,
    training_cfg: DistillationConfig,
    device: str,
    learn_output_with_ce: bool,
):
    todo_tokens = [(token, token_to_final_id[token]) for token in stage_tokens]
    token_to_pattern = {
        token: tokenize_with_teacher_tokenizer_to_final_ids(
            token,
            teacher_tokenizer=teacher_tokenizer,
            final_tokenizer=final_tokenizer,
            model_path=base_model_name,
        )
        for token, _ in todo_tokens
    }
    if isinstance(data_source, HFDataSource):
        snippet_collection = collect_stage_token_snippets_from_texts(
            texts=_iter_texts_from_hf_dataset(data_source),
            todo_tokens=todo_tokens,
            teacher_tokenizer=teacher_tokenizer,
            final_tokenizer=final_tokenizer,
            model_path=base_model_name,
            context_length=context_length,
            snippets_per_token=snippets_per_token,
            max_docs=data_source.max_docs,
        )
        token_to_pattern = snippet_collection.token_to_pattern_ids
        tokens_to_new_snippets = {
            token: [torch.tensor(snippet_ids) for snippet_ids in snippet_collection.token_to_snippets[token]]
            for token in snippet_collection.token_to_snippets
        }
        skipped = snippet_collection.skipped
        snippet_metadata = {
            **_snippet_source_metadata(data_source),
            "docs_scanned": snippet_collection.docs_scanned,
            "snippets_collected_per_token": snippet_collection.snippets_collected_per_token,
        }
    else:
        tokens_to_new_snippets = generate_samples_with_patterns(
            model,
            final_tokenizer,
            token_to_pattern,
            num_samples_per_pattern=snippets_per_token,
            seed=training_cfg.seed,
            max_length=context_length,
        )
        skipped = []
        snippet_metadata = {
            **_snippet_source_metadata(data_source),
            "docs_scanned": None,
            "snippets_collected_per_token": {
                token: len(tokens_to_new_snippets[token])
                for token, _ in todo_tokens
            },
        }

    skipped_ids = {token_id for _, token_id, _ in skipped}
    trained_tokens = [token for token in stage_tokens if token_to_final_id[token] not in skipped_ids]
    if not trained_tokens:
        return model, [], skipped, [], snippet_metadata

    trained_todo = [(token, token_to_final_id[token]) for token in trained_tokens]
    assigned_new_phrases = [tuple(token_to_pattern[token]) for token, _ in trained_todo]
    new_phrases_snippets_ids = [tokens_to_new_snippets[token] for token, _ in trained_todo]

    phrase_to_new_id = {
        phrase: token_id
        for (token, token_id), phrase in zip(trained_todo, assigned_new_phrases)
    }
    teacher_available_ids = sorted(
        set(
            remap_teacher_token_ids_to_final_ids(
                get_explicit_token_ids(teacher_tokenizer),
                teacher_tokenizer=teacher_tokenizer,
                final_tokenizer=final_tokenizer,
            )
        )
    )
    stage_trainable_ids = [token_id for _, token_id in trained_todo]

    for target_layers in target_layer_schedule:
        model = train_embeddings(
            model,
            new_phrases_snippets_ids,
            phrase_to_new_id,
            assigned_new_phrases=assigned_new_phrases,
            tokenizer=final_tokenizer,
            epochs=training_cfg.epochs,
            batch_size=training_cfg.batch_size,
            learning_rate=training_cfg.learning_rate,
            loss_methods=list(training_cfg.loss_methods),
            preserve_original_embeddings=True,
            seed=training_cfg.seed,
            original_token_ids=teacher_available_ids,
            trainable_token_ids=stage_trainable_ids,
            logit_token_ids=teacher_available_ids,
            target_layers=target_layers,
            mixed_precision=training_cfg.mixed_precision,
            learn_output_with_ce=learn_output_with_ce,
        )

    return model, trained_tokens, skipped, assigned_new_phrases, snippet_metadata


def main() -> None:
    args = _parse_args()
    seed_everything(args.seed)
    model_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    normalizer_config = _load_optional_json(args.custom_normalizer_config_json)
    pretokenizer_config = _load_optional_json(args.custom_pretokenizer_config_json)

    base_tokenizer = AutoTokenizer.from_pretrained(args.base_model, legacy=False, add_prefix_space=False)
    if args.use_custom_tokenizer_processing:
        teacher_base_tokenizer = clone_tokenizer_with_custom_arabic_components(
            base_tokenizer,
            normalizer_config=normalizer_config,
            pretokenizer_config=pretokenizer_config,
            tokenizer_arabic_src_path=args.tokenizer_arabic_src_path,
        )
        student_base_tokenizer = clone_tokenizer_with_custom_arabic_components(
            base_tokenizer,
            normalizer_config=normalizer_config,
            pretokenizer_config=pretokenizer_config,
            tokenizer_arabic_src_path=args.tokenizer_arabic_src_path,
        )
    else:
        teacher_base_tokenizer = base_tokenizer
        student_base_tokenizer = base_tokenizer

    custom_tokenizer = load_custom_tokenizer(
        args.custom_tokenizer,
        normalizer_config=normalizer_config,
        pretokenizer_config=pretokenizer_config,
        tokenizer_arabic_src_path=args.tokenizer_arabic_src_path,
    )

    distill_labels, support_labels, drop_labels = resolve_label_selection(
        preset_name=args.label_preset,
        distill_labels=args.distill_labels,
        support_labels=args.support_labels,
        drop_labels=args.drop_labels,
    )

    label_records = load_label_records(args.labels_jsonl)
    selected = select_tokens_from_label_records(
        label_records,
        base_tokenizer=base_tokenizer,
        custom_tokenizer=custom_tokenizer,
        distill_labels=sorted(distill_labels),
        support_labels=sorted(support_labels),
        drop_labels=sorted(drop_labels),
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        device_map="cpu",
        dtype=model_dtype,
        attn_implementation=args.attn_impl,
    )
    model.eval()

    all_added_tokens = list(selected.ordered_added_tokens)
    all_added_todo = [(token, index) for index, token in enumerate(all_added_tokens)]
    input_preinit = _build_input_preinit(
        model=model,
        teacher_base_tokenizer=teacher_base_tokenizer,
        base_model_name=args.base_model,
        all_added_tokens=all_added_tokens,
        all_added_todo=all_added_todo,
        selected=selected,
        init_embeddings_path=args.init_embeddings_path,
        pre_init_strategy=args.pre_init_strategy,
    )
    output_preinit = None
    if not model.config.tie_word_embeddings:
        output_preinit = compute_subtoken_means(
            model=model,
            tokenizer=teacher_base_tokenizer,
            model_path=args.base_model,
            todo_tokens=all_added_todo,
            input_or_output="output",
            method=args.pre_init_strategy,
        )

    model, final_tokenizer = extend_pretrained_with_tokens_and_embeddings(
        str(output_dir / "heuristic_initialized"),
        model=model,
        new_tokens_to_input_embs=input_preinit,
        source_tokenizer=student_base_tokenizer,
        new_tokens_to_output_embs=output_preinit,
        save=False,
    )
    model.to(args.device)

    token_to_final_id = get_token_id_map(final_tokenizer, all_added_tokens)
    base_token_ids = get_explicit_token_ids(student_base_tokenizer)
    support_token_ids = [token_to_final_id[token] for token in selected.support_tokens]
    distill_token_ids = [token_to_final_id[token] for token in selected.distill_tokens]
    all_added_token_ids = [token_to_final_id[token] for token in all_added_tokens]

    if args.sequential_layers:
        target_layer_schedule = [[layer] for layer in args.target_layers]
    else:
        target_layer_schedule = [list(args.target_layers)]

    training_cfg = DistillationConfig(
        epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        target_layers=list(args.target_layers),
        loss_methods=args.loss_methods,
    )
    data_source = _build_data_source(args)

    completed_distill_tokens: set[str] = set()
    stage_artifacts: list[dict[str, Any]] = []
    pending_distill_tokens = list(selected.distill_tokens)
    stage_index = 0
    while pending_distill_tokens:
        stage_index += 1
        teacher_available_tokens = set(selected.support_tokens) | completed_distill_tokens
        teacher_tokenizer = build_tokenizer_with_ordered_added_tokens(
            teacher_base_tokenizer,
            selected.ordered_added_tokens,
            included_tokens=teacher_available_tokens,
        )
        stage_eligibility = compute_stage_eligibility(
            pending_distill_tokens,
            teacher_tokenizer=teacher_tokenizer,
            model_path=args.base_model,
        )
        stage_tokens = [token for token in pending_distill_tokens if stage_eligibility[token]["eligible"]]
        blocked_tokens = [token for token in pending_distill_tokens if not stage_eligibility[token]["eligible"]]

        if args.skip_distillation or not stage_tokens:
            stage_artifacts.append(
                {
                    "stage_index": stage_index,
                    "teacher_available_tokens": [
                        token for token in selected.ordered_added_tokens if token in teacher_available_tokens
                    ],
                    "eligible_tokens": stage_tokens,
                    "blocked_tokens": blocked_tokens,
                    "eligibility": stage_eligibility,
                    "trained_tokens": [],
                    "skipped_for_snippets": [],
                    "docs_scanned": None,
                    "snippets_collected_per_token": {},
                    "target_layer_schedule": target_layer_schedule,
                    **_snippet_source_metadata(data_source),
                }
            )
            break

        model, trained_tokens, skipped, assigned_new_phrases, snippet_metadata = _train_one_stage(
            model=model,
            final_tokenizer=final_tokenizer,
            teacher_tokenizer=teacher_tokenizer,
            base_model_name=args.base_model,
            target_layer_schedule=target_layer_schedule,
            stage_tokens=stage_tokens,
            token_to_final_id=token_to_final_id,
            snippets_per_token=args.max_snippets_per_token,
            context_length=args.context_length,
            data_source=data_source,
            training_cfg=training_cfg,
            device=args.device,
            learn_output_with_ce=(
                args.output_embedding_policy == OutputEmbeddingInit.TRAIN_WITH_CE.value
                and not model.config.tie_word_embeddings
            ),
        )
        completed_distill_tokens.update(trained_tokens)
        stage_artifacts.append(
            {
                "stage_index": stage_index,
                "teacher_available_tokens": [
                    token for token in selected.ordered_added_tokens if token in teacher_available_tokens
                ],
                "eligible_tokens": stage_tokens,
                "blocked_tokens": blocked_tokens,
                "eligibility": stage_eligibility,
                "trained_tokens": trained_tokens,
                "assigned_new_phrases": assigned_new_phrases,
                "skipped_for_snippets": [
                    {"token": token, "token_id": token_id, "available_snippets": available}
                    for token, token_id, available in skipped
                ],
                **snippet_metadata,
                "target_layer_schedule": target_layer_schedule,
            }
        )
        pending_distill_tokens = [token for token in blocked_tokens if token not in trained_tokens]
        if not trained_tokens:
            break

    if not model.config.tie_word_embeddings:
        if args.output_embedding_policy == OutputEmbeddingInit.ZERO.value:
            for token_id in all_added_token_ids:
                model.get_output_embeddings().weight.data[token_id].zero_()
        elif args.output_embedding_policy == OutputEmbeddingInit.SUBTOKEN_MEAN.value:
            output_map = compute_subtoken_means(
                model=model,
                tokenizer=teacher_base_tokenizer,
                model_path=args.base_model,
                todo_tokens=all_added_todo,
                input_or_output="output",
                method=args.pre_init_strategy,
            )
            for token, embedding in output_map.items():
                model.get_output_embeddings().weight.data[token_to_final_id[token]] = embedding

    final_model_dir = output_dir / "final_model"
    final_model_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(final_model_dir, safe_serialization=False)
    final_tokenizer.save_pretrained(final_model_dir)

    config_payload = vars(args).copy()
    config_payload["distill_labels"] = sorted(distill_labels)
    config_payload["support_labels"] = sorted(support_labels)
    config_payload["drop_labels"] = sorted(drop_labels)

    summary = {
        "base_tokens": len(base_token_ids),
        "added_tokens": len(all_added_tokens),
        "support_tokens": len(selected.support_tokens),
        "distill_tokens": len(selected.distill_tokens),
        "dropped_tokens": len(selected.dropped_tokens),
        "support_token_ids": support_token_ids,
        "distill_token_ids": distill_token_ids,
        "all_added_token_ids": all_added_token_ids,
        "distill_eligible_per_stage": [len(stage["eligible_tokens"]) for stage in stage_artifacts],
        "actually_trained_per_stage": [len(stage["trained_tokens"]) for stage in stage_artifacts],
        "completed_distill_tokens": len(completed_distill_tokens),
    }

    token_rows = []
    for token in all_added_tokens:
        role = "support" if token in set(selected.support_tokens) else "distill"
        token_rows.append(
            {
                "token": token,
                "id": token_to_final_id[token],
                "role": role,
                "label_name": selected.token_to_label[token],
                "custom_tokenizer_id": selected.token_to_custom_id[token],
            }
        )

    (output_dir / "config.json").write_text(json.dumps(config_payload, ensure_ascii=False, indent=2))
    (output_dir / "selected_tokens.json").write_text(
        json.dumps(
            {
                "ordered_added_tokens": all_added_tokens,
                "support_tokens": selected.support_tokens,
                "distill_tokens": selected.distill_tokens,
                "dropped_tokens": selected.dropped_tokens,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    (output_dir / "token_id_map.json").write_text(json.dumps(token_rows, ensure_ascii=False, indent=2))
    (output_dir / "stages.json").write_text(json.dumps(stage_artifacts, ensure_ascii=False, indent=2))
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
