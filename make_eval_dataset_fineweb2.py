from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from datasets import Dataset, load_dataset
from tqdm import tqdm


def stable_text_bucket(text: str, *, salt: str, buckets: int = 10_000) -> int:
    normalized = " ".join(text.split())
    digest = hashlib.sha256((salt + "\n" + normalized).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % buckets


def row_ok(
    row: dict,
    *,
    text_field: str,
    quality_score_field: str,
    min_quality_score: float | None,
    eval_percent: float,
    hash_salt: str,
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

    bucket = stable_text_bucket(text, salt=hash_salt)
    cutoff = int(10_000 * eval_percent / 100.0)
    return bucket < cutoff


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", default="epfml/FineWeb2-HQ")
    parser.add_argument("--dataset-name", default="arb_Arab")
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument("--min-quality-score", type=float, default=0.7)
    parser.add_argument("--eval-percent", type=float, default=5.0)
    parser.add_argument("--hash-salt", default="arabic_td_fineweb2hq_v1")
    parser.add_argument("--max-eval-docs", type=int, default=5000)
    parser.add_argument("--output-dir", default="eval_dataset_fineweb2hq_arb_q07_hash5")
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--quality-score-field", default="quality_score")
    args = parser.parse_args()

    ds = load_dataset(
        args.dataset_path,
        name=args.dataset_name,
        split=args.dataset_split,
        streaming=True,
    )

    rows = []
    seen_hashes = set()
    scanned = 0

    pbar = tqdm(ds, desc="Collecting eval split")
    for row in pbar:
        scanned += 1

        if not row_ok(
            row,
            text_field=args.text_field,
            quality_score_field=args.quality_score_field,
            min_quality_score=args.min_quality_score,
            eval_percent=args.eval_percent,
            hash_salt=args.hash_salt,
        ):
            if scanned % 1000 == 0:
                pbar.set_postfix(scanned=scanned, kept=len(rows))
            continue

        text = row[args.text_field]
        normalized = " ".join(text.split())
        text_hash = hashlib.sha256(normalized.encode("utf-8")).hexdigest()

        if text_hash in seen_hashes:
            if scanned % 1000 == 0:
                pbar.set_postfix(scanned=scanned, kept=len(rows))
            continue

        seen_hashes.add(text_hash)

        rows.append(
            {
                "text": text,
                "quality_score": row.get(args.quality_score_field),
                "text_hash": text_hash,
                "split_bucket": stable_text_bucket(text, salt=args.hash_salt),
            }
        )

        pbar.set_postfix(scanned=scanned, kept=len(rows))

        if len(rows) >= args.max_eval_docs:
            break
        
    out = Dataset.from_list(rows)
    out_dir = Path(args.output_dir)
    out.save_to_disk(str(out_dir))

    print(f"Saved eval dataset to: {out_dir}")
    print(f"Eval docs: {len(out)}")
    print(f"Scanned raw rows: {scanned}")
    print(f"Hash salt: {args.hash_salt}")
    print(f"Eval percent: {args.eval_percent}")


if __name__ == "__main__":
    main()