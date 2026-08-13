#!/usr/bin/env python3
"""Cache and freeze a bounded real-image pzfj38 canary dataset."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import tempfile
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from PIL import Image

from common import DATA_DIR, ROOT, sha256_file, sha256_json, write_json, write_jsonl


SCHEMA = "sft_h800_vl_canary_media/v1"
SOURCE = (
    ROOT.parent
    / "real_business_validation_20260730"
    / "datasets"
    / "dataset-pzfj38-1774860803"
    / "113"
    / "publish"
    / "extracted"
    / "dataset-pzfj38-1774860803-V113.jsonl"
)
OUTPUT_DIR = DATA_DIR / "packing_vl_canary_v1"
OUTPUT_DATA = OUTPUT_DIR / "vl_pzfj38_canary_v1.jsonl"
MEDIA_DIR = OUTPUT_DIR / "media"
OUTPUT_MANIFEST = ROOT / "artifacts" / "h800_vl_canary_media_manifest_v1.json"
MAX_IMAGE_BYTES = 64 * 1024 * 1024


def _url_sha256(url: str) -> str:
    return hashlib.sha256(url.encode("utf-8")).hexdigest()


def _extension(url: str) -> str:
    suffix = Path(urllib.parse.urlparse(url).path).suffix.lower()
    return suffix if suffix in {".png", ".jpg", ".jpeg", ".webp", ".bmp"} else ".img"


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _decode_metadata(payload: bytes) -> dict[str, Any]:
    with Image.open(io.BytesIO(payload)) as image:
        image.verify()
    with Image.open(io.BytesIO(payload)) as image:
        width, height = image.size
        return {
            "width": int(width),
            "height": int(height),
            "format": str(image.format or "unknown"),
            "mode": str(image.mode),
        }


def _download(url: str, retries: int = 3) -> tuple[bytes, dict[str, Any]]:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    parsed = urllib.parse.urlsplit(url)
    request_url = urllib.parse.urlunsplit(
        (
            parsed.scheme,
            parsed.netloc,
            urllib.parse.quote(parsed.path, safe="/%:@+~"),
            urllib.parse.quote(parsed.query, safe="=&%:/?+@~"),
            "",
        )
    )
    last_error: BaseException | None = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(
                request_url,
                headers={"User-Agent": "Param-Recommend-VL-Canary/1.0"},
            )
            with opener.open(request, timeout=60) as response:
                payload = response.read(MAX_IMAGE_BYTES + 1)
                if len(payload) > MAX_IMAGE_BYTES:
                    raise ValueError(f"image exceeds {MAX_IMAGE_BYTES} bytes")
                metadata = {
                    "http_status": int(getattr(response, "status", 200)),
                    "content_type": str(response.headers.get("Content-Type") or ""),
                }
            metadata.update(_decode_metadata(payload))
            return payload, metadata
        except BaseException as error:
            last_error = error
            if attempt + 1 < retries:
                time.sleep(1.0 * (attempt + 1))
    raise RuntimeError(f"failed to cache media after {retries} attempts: {type(last_error).__name__}: {last_error}")


def _cache_one(url: str) -> dict[str, Any]:
    url_digest = _url_sha256(url)
    destination = MEDIA_DIR / f"{url_digest}{_extension(url)}"
    if destination.is_file():
        payload = destination.read_bytes()
        decode = _decode_metadata(payload)
        http = {"http_status": None, "content_type": None, "cache_reused": True}
    else:
        payload, http = _download(url)
        _atomic_write(destination, payload)
        decode = {
            key: http.pop(key) for key in ("width", "height", "format", "mode")
        }
        http["cache_reused"] = False
    return {
        "url_sha256": url_digest,
        "content_sha256": hashlib.sha256(payload).hexdigest(),
        "path": str(destination.resolve()),
        "relative_path": str(destination.resolve().relative_to(ROOT.resolve())),
        "bytes": len(payload),
        **decode,
        **http,
    }


def prepare(rows: int, workers: int) -> dict[str, Any]:
    if rows != 128:
        raise ValueError("v1 freezes exactly 128 source rows")
    if not SOURCE.is_file():
        raise FileNotFoundError(SOURCE)
    selected: list[dict[str, Any]] = []
    urls: list[str] = []
    with SOURCE.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if len(selected) >= rows:
                break
            row = json.loads(line)
            images = row.get("images")
            messages = row.get("messages")
            if (
                not isinstance(images, list)
                or len(images) != 2
                or not all(isinstance(url, str) and url.startswith("http://") for url in images)
                or not isinstance(messages, list)
                or sum(str(message.get("content") or "").count("<image>") for message in messages) != 2
            ):
                raise ValueError(f"source row {line_number} violates the two-image canary contract")
            selected.append(row)
            urls.extend(images)
    if len(selected) != rows:
        raise ValueError(f"source has only {len(selected)} eligible rows")

    unique_urls = list(dict.fromkeys(urls))
    cached: dict[str, dict[str, Any]] = {}
    MEDIA_DIR.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_cache_one, url): url for url in unique_urls}
        for completed, future in enumerate(as_completed(futures), start=1):
            url = futures[future]
            cached[url] = future.result()
            if completed % 32 == 0 or completed == len(futures):
                print(f"cached {completed}/{len(futures)} images", flush=True)

    derived = []
    for row in selected:
        derived.append(
            {
                "messages": row["messages"],
                "images": [cached[url]["path"] for url in row["images"]],
            }
        )
    write_jsonl(OUTPUT_DATA, derived)
    media_rows = sorted(cached.values(), key=lambda item: item["url_sha256"])
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "source": {
            "path": str(SOURCE.resolve()),
            "sha256": sha256_file(SOURCE),
            "selection": "first_128_rows_in_source_order_for_software_media_canary_only",
        },
        "derived": {
            "path": str(OUTPUT_DATA.resolve()),
            "sha256": sha256_file(OUTPUT_DATA),
            "rows": len(derived),
            "images_per_row": 2,
            "unique_images": len(media_rows),
            "total_image_bytes": sum(int(row["bytes"]) for row in media_rows),
            "all_decoded": len(media_rows) == len(unique_urls),
            "raw_urls_excluded_from_manifest": True,
        },
        "media": media_rows,
        "usage": "semantic_canary_only_excluded_from_fit_and_acceptance",
    }
    report["report_sha256"] = sha256_json(report)
    write_json(OUTPUT_MANIFEST, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=128)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    report = prepare(args.rows, args.workers)
    print(
        json.dumps(
            {
                "data": report["derived"],
                "manifest": str(OUTPUT_MANIFEST),
                "report_sha256": report["report_sha256"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
