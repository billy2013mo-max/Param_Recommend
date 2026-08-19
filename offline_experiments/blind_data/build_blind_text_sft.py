#!/usr/bin/env python3
"""Build blind-test SFT datasets from downloaded business ASR/OCR text.

Turns the 0105 business text rows (title + ASR + OCR) into llamafactory
`messages` SFT rows.  Task: given the video's ASR and OCR text, generate the
video title.  Output keeps the real business text length distribution, which
is what the memory/throughput blind test needs.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

SYSTEM = (
    "你是短视频内容理解助手。根据给定视频的语音识别(ASR)文本和画面(OCR)文本，"
    "提炼并生成该视频的标题。标题要准确概括视频主题，简洁自然。"
)


def parse_text(text: str) -> dict[str, str]:
    """Parse the '# 标题 / # ASR文本 / # OCR文本' sections."""
    title = ""
    asr = ""
    ocr = ""
    current = ""
    for line in text.splitlines():
        if line.startswith("# 标题"):
            current = "title"
        elif line.startswith("# ASR文本"):
            current = "asr"
        elif line.startswith("# OCR文本"):
            current = "ocr"
        elif line.strip() and current:
            if current == "title":
                title = (title + line.strip()).strip()
            elif current == "asr":
                asr = (asr + line.strip()).strip()
            elif current == "ocr":
                ocr = (ocr + line.strip()).strip()
    return {"title": title, "asr": asr, "ocr": ocr}


def build_rows(input_path: Path, limit: int | None = None) -> list[dict]:
    rows = []
    with open(input_path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            raw = json.loads(line)
            text = str(raw.get("text") or "")
            parsed = parse_text(text)
            if not parsed["title"] or not parsed["asr"]:
                continue
            user_text = "视频内容文本：\n"
            if parsed["asr"]:
                user_text += f"【语音识别】{parsed['asr']}\n"
            if parsed["ocr"]:
                user_text += f"【画面文本】{parsed['ocr']}"
            rows.append(
                {
                    "messages": [
                        {"role": "system", "content": SYSTEM},
                        {"role": "user", "content": user_text.strip()},
                        {"role": "assistant", "content": parsed["title"]},
                    ],
                    "sample_id": f"blind-{Path(input_path).stem}-{i:06d}",
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    rows = build_rows(args.input, limit=args.limit)
    with open(args.output, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
