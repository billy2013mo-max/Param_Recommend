#!/usr/bin/env python3
"""Token-length profile for the blind-test SFT datasets.

Uses the real Qwen3 tokenizer + the chat template the training would apply,
matching how the offline pipeline profiles real datasets.  Outputs length
distribution statistics needed to build the blind GPU queue.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from transformers import AutoTokenizer


def build_profile(sft_path: Path, tokenizer, template: str = "qwen3") -> dict:
    total = 0
    sample_tokens = []
    with open(sft_path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            messages = row["messages"]
            # Apply the chat template the way training does (assistant turn marked).
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            ids = tokenizer(text, add_special_tokens=False)["input_ids"]
            n = len(ids)
            total += n
            sample_tokens.append(n)
    sample_tokens.sort()
    n = len(sample_tokens)
    return {
        "schema": "sft_blind_text_profile/v1",
        "source": str(sft_path.name),
        "template": template,
        "tokenizer": tokenizer.name_or_path,
        "rows": n,
        "total_tokens": total,
        "mean_tokens_per_sample": total / n if n else None,
        "median_tokens_per_sample": statistics.median(sample_tokens) if n else None,
        "p90_tokens_per_sample": sample_tokens[int(n * 0.90)] if n else None,
        "p95_tokens_per_sample": sample_tokens[int(n * 0.95)] if n else None,
        "p99_tokens_per_sample": sample_tokens[int(n * 0.99)] if n else None,
        "max_tokens_per_sample": sample_tokens[-1] if n else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--tokenizer",
        default="/wanqing-models/Qwen3-8B",
        help="HF tokenizer directory",
    )
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    profile = build_profile(args.input, tokenizer)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(profile, f, ensure_ascii=False, indent=1)
    print(json.dumps(profile, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
