import os
import csv
import json
import time
import argparse
from pathlib import Path
from typing import Iterator, Literal

from tqdm import tqdm
from pydantic import BaseModel, ConfigDict
from openai import OpenAI


SYSTEM_PROMPT = """
You classify isolated Arabic tokenizer tokens by surface form only.

Input is a JSON object:
{"items":[{"id":123,"token":"word"}]}

Classify each token using only the token string itself.
Ignore original leading and trailing whitespace; tokens are already normalized for you.

Return compact JSON only on one line:
{"items":[{"id":123,"label":"CW"}]}

Labels:
- CW = standalone lexical Arabic word
- FW = standalone Arabic function word
- AFC = attached function/clitic + lexical stem
- FRAG = shard, affix, bound morpheme, incomplete fragment
- NAME = proper name, named entity, transliterated or foreign item in Arabic script
- NOISE = punctuation, digits, emoji, Latin/mixed script, markup, whitespace-like, corrupted text
- AMB = uncertain from token alone

Rules:
- Use only the token itself, not context.
- Output exactly one item for each input item.
- Keep the same order as input.
- Use exactly the provided ids.
- Do not add, remove, duplicate, or reorder items.
- Be conservative; use AMB when unsure.
- Do not use AFC for an ordinary standalone word just because it begins with ال.
- Use FW only for clear standalone function words.
- Standalone verbs, nouns, adjectives, and adverbs are CW.

Examples:
كتاب -> CW
من -> FW
والشعر -> AFC
بالبريد -> AFC
للاستمرار -> AFC
الدخول -> CW
القرن -> CW
تشهد -> CW
فرض -> CW
ـها -> FRAG
باريس -> NAME
123 -> NOISE
عين -> AMB
""".strip()


LABEL_MAP = {
    "CW": "clean_content_word",
    "FW": "clean_function_word",
    "AFC": "attached_function_plus_content",
    "FRAG": "fragment_or_affix",
    "NAME": "proper_name_or_foreign",
    "NOISE": "nonword_or_noise",
    "AMB": "ambiguous",
}

LABEL_ENUM = ["CW", "FW", "AFC", "FRAG", "NAME", "NOISE", "AMB"]
LabelCode = Literal["CW", "FW", "AFC", "FRAG", "NAME", "NOISE", "AMB"]


class TokenLabel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: int
    label: LabelCode


class BatchOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    items: list[TokenLabel]


def make_response_format(batch: list[dict]) -> dict:
    prefix_items = []
    for item in batch:
        prefix_items.append(
            {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "id": {"type": "integer", "const": item["id"]},
                    "label": {"type": "string", "enum": LABEL_ENUM},
                },
                "required": ["id", "label"],
            }
        )

    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "items": {
                "type": "array",
                "prefixItems": prefix_items,
                "items": False,
                "minItems": len(batch),
                "maxItems": len(batch),
            }
        },
        "required": ["items"],
    }

    return {
        "type": "json_schema",
        "json_schema": {
            "name": "BatchOutputExact",
            "schema": schema,
            "strict": True,
        },
    }


def load_items(path: Path) -> list[dict]:
    ext = path.suffix.lower()

    if ext == ".txt":
        with path.open("r", encoding="utf-8") as f:
            tokens = [line.rstrip("\n") for line in f if line.rstrip("\n") != ""]
        return [{"id": i, "token": tok} for i, tok in enumerate(tokens)]

    if ext in {".csv", ".tsv"}:
        delimiter = "," if ext == ".csv" else "\t"
        with path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            if not reader.fieldnames or "token" not in reader.fieldnames:
                raise ValueError(f"{path} must contain a 'token' column")
            items = []
            for i, row in enumerate(reader):
                item_id = int(row["id"]) if "id" in row and row["id"] != "" else i
                items.append({"id": item_id, "token": row["token"]})
            return items

    if ext == ".jsonl":
        items = []
        with path.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                obj = json.loads(line)
                if "token" not in obj:
                    raise ValueError(f"{path} JSONL rows must contain 'token'")
                item_id = int(obj["id"]) if "id" in obj else i
                items.append({"id": item_id, "token": obj["token"]})
        return items

    raise ValueError("Supported input formats: .txt, .csv, .tsv, .jsonl")


def load_done_ids(results_path: Path) -> set[int]:
    done = set()
    if not results_path.exists():
        return done

    with results_path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                done.add(int(obj["id"]))
            except Exception:
                continue
    return done


def chunked(seq: list[dict], size: int) -> Iterator[list[dict]]:
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def build_user_payload(batch: list[dict]) -> str:
    payload = {
        "items": [{"id": item["id"], "token": item["token"].strip()} for item in batch]
    }
    return json.dumps(payload, ensure_ascii=False)


def validate_batch_output(batch: list[dict], parsed: BatchOutput) -> BatchOutput:
    expected_ids = [item["id"] for item in batch]
    got_ids = [item.id for item in parsed.items]

    if len(got_ids) != len(expected_ids):
        raise ValueError(
            f"Wrong number of items in response. Expected {len(expected_ids)}, got {len(got_ids)}"
        )

    if got_ids != expected_ids:
        raise ValueError(
            f"IDs/order mismatch. Expected {expected_ids}, got {got_ids}"
        )

    return parsed


def classify_batch(
    client: OpenAI,
    model: str,
    batch: list[dict],
    max_retries: int,
    temperature: float,
    max_tokens: int,
) -> BatchOutput:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_payload(batch)},
    ]

    last_err = None

    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                response_format=make_response_format(batch),
                temperature=temperature,
                max_tokens=max_tokens,
            )

            finish_reason = response.choices[0].finish_reason
            content = response.choices[0].message.content

            if finish_reason == "length":
                raise ValueError("Response was truncated: finish_reason=length")

            if not content:
                raise ValueError("Empty response content")

            parsed = BatchOutput.model_validate_json(content)
            return validate_batch_output(batch, parsed)

        except Exception as e:
            last_err = e
            msg = str(e)

            if (
                "402" in msg
                or "Payment Required" in msg
                or "depleted your monthly included credits" in msg
            ):
                raise RuntimeError(
                    "Inference credits are exhausted. Stop the run and resume later with the same output directory."
                ) from e

            sleep_s = min(60, 2 ** attempt)
            print(f"[warn] batch failed on attempt {attempt}/{max_retries}: {last_err}")
            if attempt < max_retries:
                time.sleep(sleep_s)

    raise RuntimeError(f"Batch failed after {max_retries} attempts: {last_err}")


def save_state(state_path: Path, state: dict):
    tmp = state_path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    tmp.replace(state_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to tokens file: txt/csv/tsv/jsonl")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-32B-Instruct")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key", default="local-token")
    parser.add_argument("--provider", default="local-vllm")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=768)
    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results_path = out_dir / "results.jsonl"
    errors_path = out_dir / "errors.jsonl"
    state_path = out_dir / "state.json"

    all_items = load_items(input_path)
    done_ids = load_done_ids(results_path)
    pending = [item for item in all_items if item["id"] not in done_ids]

    print(f"Total tokens: {len(all_items)}")
    print(f"Already done: {len(done_ids)}")
    print(f"Pending: {len(pending)}")

    client = OpenAI(base_url=args.base_url, api_key=args.api_key)

    started_at = time.time()
    attempted_this_run = 0
    succeeded_this_run = 0
    failed_batches = 0
    failed_items = 0

    with results_path.open("a", encoding="utf-8") as results_f, errors_path.open("a", encoding="utf-8") as errors_f:
        pbar = tqdm(total=len(pending), desc="Classifying tokens")

        for batch_idx, batch in enumerate(chunked(pending, args.batch_size), start=1):
            ts = time.strftime("%Y-%m-%d %H:%M:%S")

            try:
                parsed = classify_batch(
                    client=client,
                    model=args.model,
                    batch=batch,
                    max_retries=args.max_retries,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                )

                label_by_id = {item.id: item for item in parsed.items}

                for item in batch:
                    pred = label_by_id[item["id"]]
                    record = {
                        "id": item["id"],
                        "token": item["token"],
                        "normalized_token": item["token"].strip(),
                        "label": pred.label,
                        "label_name": LABEL_MAP[pred.label],
                        "model": args.model,
                        "provider": args.provider,
                        "prompt_version": "arabic_token_labels_v2_exact_schema",
                        "timestamp": ts,
                    }
                    results_f.write(json.dumps(record, ensure_ascii=False) + "\n")

                results_f.flush()
                os.fsync(results_f.fileno())
                succeeded_this_run += len(batch)

            except Exception as e:
                failed_batches += 1
                failed_items += len(batch)

                error_record = {
                    "batch_idx": batch_idx,
                    "error": str(e),
                    "batch": batch,
                    "timestamp": ts,
                }
                errors_f.write(json.dumps(error_record, ensure_ascii=False) + "\n")
                errors_f.flush()
                os.fsync(errors_f.fileno())

            attempted_this_run += len(batch)
            pbar.update(len(batch))

            elapsed = time.time() - started_at
            rate = attempted_this_run / elapsed if elapsed > 0 else 0.0
            eta_s = (len(pending) - attempted_this_run) / rate if rate > 0 else None

            state = {
                "input": str(input_path),
                "output_dir": str(out_dir),
                "model": args.model,
                "provider": args.provider,
                "batch_size": args.batch_size,
                "attempted_this_run": attempted_this_run,
                "succeeded_this_run": succeeded_this_run,
                "failed_batches": failed_batches,
                "failed_items": failed_items,
                "pending_total_at_start": len(pending),
                "rate_items_per_sec": rate,
                "eta_seconds": eta_s,
                "last_batch_idx": batch_idx,
                "updated_at": ts,
            }
            save_state(state_path, state)

        pbar.close()

    print(f"Done. Results: {results_path}")
    print(f"Errors:  {errors_path}")
    print(f"State:   {state_path}")


if __name__ == "__main__":
    main()