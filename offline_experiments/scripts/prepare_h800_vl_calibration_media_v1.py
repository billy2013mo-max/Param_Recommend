#!/usr/bin/env python3
"""Cache and freeze 1,000 two-image pzfj38 calibration rows."""

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


SCHEMA = "sft_h800_vl_calibration_media/v1"
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
OUTPUT_DIR = DATA_DIR / "vl_calibration_v1"
OUTPUT_DATA = OUTPUT_DIR / "vl_pzfj38_calibration_v1.jsonl"
MEDIA_DIR = OUTPUT_DIR / "media"
CANARY_MEDIA_DIR = DATA_DIR / "packing_vl_canary_v1" / "media"
OUTPUT_MANIFEST = ROOT / "artifacts" / "h800_vl_calibration_media_manifest_v1.json"
ROWS = 1000
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


def _decode(payload: bytes) -> dict[str, Any]:
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
                headers={"User-Agent": "Param-Recommend-VL-Calibration/1.0"},
            )
            with opener.open(request, timeout=60) as response:
                payload = response.read(MAX_IMAGE_BYTES + 1)
                if len(payload) > MAX_IMAGE_BYTES:
                    raise ValueError(f"image exceeds {MAX_IMAGE_BYTES} bytes")
                http = {
                    "http_status": int(getattr(response, "status", 200)),
                    "content_type": str(response.headers.get("Content-Type") or ""),
                }
            return payload, {**http, **_decode(payload)}
        except BaseException as error:
            last_error = error
            if attempt + 1 < retries:
                time.sleep(float(attempt + 1))
    raise RuntimeError(
        f"failed to cache media after {retries} attempts: "
        f"{type(last_error).__name__}: {last_error}"
    )


def _existing_path(digest: str, preferred: Path) -> Path | None:
    if preferred.is_file():
        return preferred
    matches = sorted(CANARY_MEDIA_DIR.glob(f"{digest}.*"))
    return matches[0] if len(matches) == 1 else None


def _cache_one(url: str) -> dict[str, Any]:
    digest = _url_sha256(url)
    preferred = MEDIA_DIR / f"{digest}{_extension(url)}"
    existing = _existing_path(digest, preferred)
    if existing is not None:
        payload = existing.read_bytes()
        decode = _decode(payload)
        http = {
            "http_status": None,
            "content_type": None,
            "cache_reused": True,
            "cache_source": "calibration" if existing.parent == MEDIA_DIR else "semantic_canary",
        }
        destination = existing
    else:
        payload, downloaded = _download(url)
        _atomic_write(preferred, payload)
        decode = {key: downloaded.pop(key) for key in ("width", "height", "format", "mode")}
        http = {
            **downloaded,
            "cache_reused": False,
            "cache_source": "downloaded_for_calibration",
        }
        destination = preferred
    return {
        "url_sha256": digest,
        "content_sha256": hashlib.sha256(payload).hexdigest(),
        "path": str(destination.resolve()),
        "relative_path": str(destination.resolve().relative_to(ROOT.resolve())),
        "bytes": len(payload),
        **decode,
        **http,
    }


def prepare(workers: int) -> dict[str, Any]:
    if not SOURCE.is_file():
        raise FileNotFoundError(SOURCE)
    selected: list[dict[str, Any]] = []
    urls: list[str] = []
    with SOURCE.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if len(selected) >= ROWS:
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
                raise ValueError(f"source row {line_number} violates the two-image contract")
            selected.append(row)
            urls.extend(images)
    if len(selected) != ROWS:
        raise ValueError(f"source has only {len(selected)} eligible rows")

    unique_urls = list(dict.fromkeys(urls))
    MEDIA_DIR.mkdir(parents=True, exist_ok=True)
    cached: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_cache_one, url): url for url in unique_urls}
        for completed, future in enumerate(as_completed(futures), start=1):
            url = futures[future]
            cached[url] = future.result()
            if completed % 100 == 0 or completed == len(futures):
                print(f"cached {completed}/{len(futures)} unique images", flush=True)

    derived = [
        {
            "messages": row["messages"],
            "images": [cached[url]["path"] for url in row["images"]],
        }
        for row in selected
    ]
    write_jsonl(OUTPUT_DATA, derived)
    media = sorted(cached.values(), key=lambda item: item["url_sha256"])
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "source": {
            "path": str(SOURCE.resolve()),
            "sha256": sha256_file(SOURCE),
            "selection": "first_1000_rows_in_source_order_calibration_fit_only",
        },
        "derived": {
            "path": str(OUTPUT_DATA.resolve()),
            "sha256": sha256_file(OUTPUT_DATA),
            "rows": len(derived),
            "images_per_row": 2,
            "unique_images": len(media),
            "total_image_bytes": sum(int(row["bytes"]) for row in media),
            "all_decoded": len(media) == len(unique_urls),
            "raw_urls_excluded_from_manifest": True,
            "cache_reused_images": sum(row["cache_reused"] is True for row in media),
        },
        "media": media,
        "usage": "calibration_fit_only_never_prospective_acceptance",
    }
    report["report_sha256"] = sha256_json(report)
    write_json(OUTPUT_MANIFEST, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=24)
    args = parser.parse_args()
    report = prepare(args.workers)
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
