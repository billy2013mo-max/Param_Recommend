#!/usr/bin/env python3
"""Extract, cache, decode and freeze the selected business image/video datasets."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import stat
import tempfile
import time
import urllib.parse
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any

import cv2
from PIL import Image

from common import ARTIFACT_DIR, DATA_DIR, percentile, sha256_file, sha256_json, write_json, write_jsonl


SCHEMA = "sft_h800_vl_media_business_dataset/v1"
DATA_ROOT = DATA_DIR / "vl_media_business_v1"
RAW_ROOT = DATA_ROOT / "raw"
EXTRACTED_ROOT = DATA_ROOT / "extracted"
DERIVED_ROOT = DATA_ROOT / "derived"
MEDIA_ROOT = DATA_ROOT / "media"
MANIFEST_PATH = ARTIFACT_DIR / "h800_vl_media_business_download_manifest_v1.json"
IMAGE_CAP_BYTES = 64 * 1024 * 1024
VIDEO_CAP_BYTES = 512 * 1024 * 1024
TOTAL_VIDEO_CAP_BYTES = 4 * 1024 * 1024 * 1024

SOURCES: dict[str, dict[str, Any]] = {
    "pzfj38_v113": {
        "dataset_id": "dataset-pzfj38-1774860803",
        "version": 113,
        "modality": "image",
        "object_key": "datasets/dataset-pzfj38-1774860803/113/publish/dataset-pzfj38-1774860803-V113.zip",
        "raw": RAW_ROOT / "dataset-pzfj38-1774860803/113/publish/dataset-pzfj38-1774860803-V113.zip",
        "size": 1_469_857,
        "sha256": "d67bc5424a857993b60d6aa18d7f9cdb0648aa733b0078d0270e42cf5b2c0d57",
    },
    "qype19_v7": {
        "dataset_id": "dataset-qype19-1770794488",
        "version": 7,
        "modality": "image",
        "object_key": "datasets/dataset-qype19-1770794488/7/publish/dataset-qype19-1770794488-V7.zip",
        "raw": RAW_ROOT / "dataset-qype19-1770794488/7/publish/dataset-qype19-1770794488-V7.zip",
        "size": 92_150,
        "sha256": "ca5a8c32be87553bb0c7e8250e36e2d38ef1e25a274c0cd9c0da00ff81e6b7aa",
    },
    "zltbjg_v2": {
        "dataset_id": "dataset-zltbjg-1780992949",
        "version": 2,
        "modality": "video",
        "object_key": "datasets/dataset-zltbjg-1780992949/2/publish/dataset-zltbjg-1780992949-V2.jsonl",
        "raw": RAW_ROOT / "dataset-zltbjg-1780992949/2/publish/dataset-zltbjg-1780992949-V2.jsonl",
        "size": 1_341_992,
        "sha256": "37ea8448c0903e543e2772824ac7735d9b737a7cc38d8f9f3be61d6288062fc6",
    },
}


def _validate_sources() -> None:
    for source in SOURCES.values():
        path = Path(source["raw"])
        if not path.is_file():
            raise FileNotFoundError(f"Run download_vl_media_business_v1.sh first: {path}")
        if path.stat().st_size != source["size"]:
            raise ValueError(f"Unexpected size for {path}")
        if sha256_file(path) != source["sha256"]:
            raise ValueError(f"Unexpected SHA-256 for {path}")


def _safe_extract(archive: Path, destination: Path) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    members: list[zipfile.ZipInfo]
    with zipfile.ZipFile(archive) as source:
        members = source.infolist()
        if len(members) != 1:
            raise ValueError(f"Expected one file in {archive}, got {len(members)}")
        for member in members:
            member_path = PurePosixPath(member.filename)
            if member_path.is_absolute() or ".." in member_path.parts:
                raise ValueError(f"Unsafe archive member: {member.filename}")
            mode = member.external_attr >> 16
            if stat.S_ISLNK(mode):
                raise ValueError(f"Refusing symlink archive member: {member.filename}")
            target = (destination / member.filename).resolve()
            if root not in target.parents and target != root:
                raise ValueError(f"Archive member escapes destination: {member.filename}")
        source.extractall(destination)
    extracted = destination / members[0].filename
    if not extracted.is_file():
        raise FileNotFoundError(extracted)
    return extracted


def _request_url(url: str) -> str:
    parsed = urllib.parse.urlsplit(url)
    return urllib.parse.urlunsplit(
        (
            parsed.scheme,
            parsed.netloc,
            urllib.parse.quote(parsed.path, safe="/%:@+~"),
            urllib.parse.quote(parsed.query, safe="=&%:/?+@~"),
            "",
        )
    )


def _suffix(url: str, modality: str) -> str:
    value = Path(urllib.parse.urlsplit(url).path).suffix.lower()
    allowed = {".png", ".jpg", ".jpeg", ".webp", ".bmp"} if modality == "image" else {".mp4", ".webm", ".mov", ".mkv"}
    return value if value in allowed else (".img" if modality == "image" else ".video")


def _download(url: str, destination: Path, byte_cap: int, retries: int = 3) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    last_error: BaseException | None = None
    for attempt in range(retries):
        descriptor, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".part", dir=destination.parent)
        temporary = Path(temporary_name)
        downloaded = 0
        try:
            with os.fdopen(descriptor, "wb") as output:
                request = urllib.request.Request(
                    _request_url(url),
                    headers={"User-Agent": "Param-Recommend-VL-Media/1.0"},
                )
                with opener.open(request, timeout=120) as response:
                    content_type = str(response.headers.get("Content-Type") or "").split(";", 1)[0]
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        downloaded += len(chunk)
                        if downloaded > byte_cap:
                            raise ValueError(f"media exceeds per-object byte cap {byte_cap}")
                        output.write(chunk)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, destination)
            return {"content_type": content_type, "http_status": int(getattr(response, "status", 200))}
        except BaseException as error:
            last_error = error
            try:
                os.close(descriptor)
            except OSError:
                pass
            temporary.unlink(missing_ok=True)
            if attempt + 1 < retries:
                time.sleep(float(attempt + 1))
    raise RuntimeError(f"Failed to download media: {type(last_error).__name__}: {last_error}")


def _decode_image(path: Path) -> dict[str, Any]:
    with Image.open(path) as image:
        image.verify()
    with Image.open(path) as image:
        width, height = image.size
        return {
            "decode_ok": True,
            "format": str(image.format or "unknown"),
            "mode": str(image.mode),
            "width": int(width),
            "height": int(height),
            "pixels": int(width * height),
        }


def _decode_video(path: Path) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"OpenCV cannot open {path}")
    try:
        width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
        height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        duration = float(frame_count / fps) if fps > 0 else 0.0
        decoded_positions: list[int] = []
        for position in sorted({0, max(frame_count // 2, 0), max(frame_count - 1, 0)}):
            capture.set(cv2.CAP_PROP_POS_FRAMES, position)
            ok, frame = capture.read()
            if ok and frame is not None and frame.size > 0:
                decoded_positions.append(position)
        if width <= 0 or height <= 0 or fps <= 0 or frame_count <= 0 or not decoded_positions:
            raise ValueError(f"Invalid video metadata for {path}")
        return {
            "decode_ok": True,
            "format": "video",
            "width": width,
            "height": height,
            "pixels_per_frame": width * height,
            "fps": fps,
            "frame_count": frame_count,
            "duration_seconds": duration,
            "decoded_probe_frames": decoded_positions,
        }
    finally:
        capture.release()


def _cache_one(url: str, modality: str) -> dict[str, Any]:
    url_digest = hashlib.sha256(url.encode("utf-8")).hexdigest()
    destination = MEDIA_ROOT / modality / f"{url_digest}{_suffix(url, modality)}"
    cache_reused = destination.is_file()
    http: dict[str, Any] = {"http_status": None, "content_type": None}
    if not cache_reused:
        http = _download(url, destination, IMAGE_CAP_BYTES if modality == "image" else VIDEO_CAP_BYTES)
    decoded = _decode_image(destination) if modality == "image" else _decode_video(destination)
    return {
        "url_sha256": url_digest,
        "content_sha256": sha256_file(destination),
        "path": str(destination.resolve()),
        "relative_path": str(destination.resolve().relative_to(DATA_ROOT.parent.resolve())),
        "bytes": destination.stat().st_size,
        "cache_reused": cache_reused,
        **http,
        **decoded,
    }


def _cache_all(urls: list[str], modality: str, workers: int) -> dict[str, dict[str, Any]]:
    unique = list(dict.fromkeys(urls))
    cached: dict[str, dict[str, Any]] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_cache_one, url, modality): url for url in unique}
        for completed, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            url = futures[future]
            cached[url] = future.result()
            if completed % 100 == 0 or completed == len(futures):
                print(f"cached {completed}/{len(futures)} unique {modality} objects", flush=True)
    return cached


def _read_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON on {path}:{line_number}") from error
            if not isinstance(row, dict):
                raise ValueError(f"Expected object on {path}:{line_number}")
            rows.append(row)
    return rows


def _distribution(values: list[float | int]) -> dict[str, float | int]:
    return {
        "minimum": min(values),
        "p50": percentile(values, 50),
        "p90": percentile(values, 90),
        "p95": percentile(values, 95),
        "maximum": max(values),
    }


def _row_digest(row: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(row, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def _prepare_qype(source_path: Path, workers: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = _read_rows(source_path)
    urls = [url for row in rows for url in (row.get("images") or [])]
    if len(rows) != 2011 or any(len(row.get("images") or []) != 2 for row in rows):
        raise ValueError("qype19 V7 must contain 2,011 two-image rows")
    placeholders = [sum(str(message.get("content") or "").count("<image>") for message in row.get("messages") or []) for row in rows]
    if any(count != 2 for count in placeholders):
        raise ValueError("qype19 image placeholders do not match two-image rows")
    cached = _cache_all(urls, "image", workers)
    derived = [{"messages": row["messages"], "images": [cached[url]["path"] for url in row["images"]]} for row in rows]
    output = DERIVED_ROOT / "qype19_v7_images_local.jsonl"
    write_jsonl(output, derived)
    media = sorted(cached.values(), key=lambda item: item["url_sha256"])
    profile = [
        {
            "row_index": index,
            "source_row_sha256": _row_digest(row),
            "image_count": 2,
            "total_image_pixels": sum(cached[url]["pixels"] for url in row["images"]),
            "maximum_image_pixels": max(cached[url]["pixels"] for url in row["images"]),
            "image_bytes": sum(cached[url]["bytes"] for url in row["images"]),
        }
        for index, row in enumerate(rows)
    ]
    profile_path = DERIVED_ROOT / "qype19_v7_image_workload_profile.jsonl"
    write_jsonl(profile_path, profile)
    summary = {
        "rows": len(rows),
        "media_references": len(urls),
        "unique_media": len(media),
        "total_media_bytes": sum(item["bytes"] for item in media),
        "all_media_decoded": all(item["decode_ok"] for item in media),
        "formats": dict(Counter(item["format"] for item in media)),
        "width": _distribution([item["width"] for item in media]),
        "height": _distribution([item["height"] for item in media]),
        "pixels": _distribution([item["pixels"] for item in media]),
        "row_total_image_pixels": _distribution([item["total_image_pixels"] for item in profile]),
        "schema_repair": None,
        "derived_path": str(output.resolve()),
        "derived_sha256": sha256_file(output),
        "profile_path": str(profile_path.resolve()),
        "profile_sha256": sha256_file(profile_path),
    }
    return summary, media


def _prepare_video(source_path: Path, workers: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = _read_rows(source_path)
    urls = [url for row in rows for url in (row.get("videos") or [])]
    if len(rows) != 100 or any(len(row.get("videos") or []) != 1 for row in rows):
        raise ValueError("zltbjg V2 must contain 100 one-video rows")
    original_placeholders = sum(str(row.get("prompt") or "").count("<video>") for row in rows)
    cached = _cache_all(urls, "video", workers)
    media = sorted(cached.values(), key=lambda item: item["url_sha256"])
    if sum(item["bytes"] for item in media) > TOTAL_VIDEO_CAP_BYTES:
        raise ValueError("Downloaded video corpus exceeds the campaign total byte cap")
    derived = []
    for row in rows:
        prompt = str(row.get("prompt") or "")
        if "<video>" not in prompt:
            prompt = f"<video>\n{prompt}"
        derived.append({"prompt": prompt, "response": row.get("response"), "videos": [cached[row["videos"][0]]["path"]]})
    output = DERIVED_ROOT / "zltbjg_v2_videos_local.jsonl"
    write_jsonl(output, derived)
    profile = [
        {
            "row_index": index,
            "source_row_sha256": _row_digest(row),
            "video_count": 1,
            "video_bytes": cached[row["videos"][0]]["bytes"],
            "width": cached[row["videos"][0]]["width"],
            "height": cached[row["videos"][0]]["height"],
            "pixels_per_frame": cached[row["videos"][0]]["pixels_per_frame"],
            "fps": cached[row["videos"][0]]["fps"],
            "frame_count": cached[row["videos"][0]]["frame_count"],
            "duration_seconds": cached[row["videos"][0]]["duration_seconds"],
        }
        for index, row in enumerate(rows)
    ]
    profile_path = DERIVED_ROOT / "zltbjg_v2_video_workload_profile.jsonl"
    write_jsonl(profile_path, profile)
    summary = {
        "rows": len(rows),
        "media_references": len(urls),
        "unique_media": len(media),
        "total_media_bytes": sum(item["bytes"] for item in media),
        "all_media_decoded": all(item["decode_ok"] for item in media),
        "width": _distribution([item["width"] for item in media]),
        "height": _distribution([item["height"] for item in media]),
        "pixels_per_frame": _distribution([item["pixels_per_frame"] for item in media]),
        "fps": _distribution([item["fps"] for item in media]),
        "frame_count": _distribution([item["frame_count"] for item in media]),
        "duration_seconds": _distribution([item["duration_seconds"] for item in media]),
        "schema_repair": {
            "original_video_placeholders": original_placeholders,
            "derived_video_placeholders": sum(item["prompt"].count("<video>") for item in derived),
            "policy": "prepend_one_video_placeholder_when_missing_preserve_raw_source",
        },
        "derived_path": str(output.resolve()),
        "derived_sha256": sha256_file(output),
        "profile_path": str(profile_path.resolve()),
        "profile_sha256": sha256_file(profile_path),
    }
    return summary, media


def prepare(image_workers: int, video_workers: int) -> dict[str, Any]:
    _validate_sources()
    pzfj38_jsonl = _safe_extract(Path(SOURCES["pzfj38_v113"]["raw"]), EXTRACTED_ROOT / "pzfj38_v113")
    qype19_jsonl = _safe_extract(Path(SOURCES["qype19_v7"]["raw"]), EXTRACTED_ROOT / "qype19_v7")
    qype_summary, qype_media = _prepare_qype(qype19_jsonl, image_workers)
    video_summary, video_media = _prepare_video(Path(SOURCES["zltbjg_v2"]["raw"]), video_workers)

    existing_pz_manifest = ARTIFACT_DIR / "h800_vl_calibration_media_manifest_v1.json"
    existing_pz = json.loads(existing_pz_manifest.read_text(encoding="utf-8")) if existing_pz_manifest.is_file() else None
    pz_rows = _read_rows(pzfj38_jsonl)
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "selection_policy": {
            "pzfj38_v113": "existing calibration-fit image source; archive reacquired and existing 1,000-row media cache reused",
            "qype19_v7": "source-isolated image validation; all 2,011 rows and all unique images",
            "zltbjg_v2": "video mechanism calibration; all 100 rows and all unique videos",
            "raw_urls_excluded_from_manifest": True,
        },
        "screened_but_not_selected": {
            "dataset-7ou19q-1784268835-v1": {
                "rows": 1000,
                "modality": "image",
                "unique_media_references": 952,
                "accessible_media_references_at_screen": 0,
                "schema_fields": ["prompt", "images", "model_output_contents"],
                "reason": "all media references inaccessible from the training environment and output-field semantics require an explicit adapter contract",
            },
            "dataset-ga4yhz-1785379675-v1": {
                "rows": 281,
                "modality": "image",
                "unique_media_references": 215,
                "accessible_media_references_at_screen": 192,
                "schema_fields": ["prompt", "images"],
                "reason": "no supervised response field and 23 unique media references inaccessible at screen time",
            },
        },
        "blobstore": {
            "bucket": "infra-ai-infra-storage",
            "endpoint_environment": "idc",
            "download_script": str((Path(__file__).parent / "download_vl_media_business_v1.sh").resolve()),
            "objects": [
                {
                    "dataset_id": source["dataset_id"],
                    "version": source["version"],
                    "modality": source["modality"],
                    "object_key": source["object_key"],
                    "local_path": str(Path(source["raw"]).resolve()),
                    "size_bytes": source["size"],
                    "sha256": source["sha256"],
                }
                for source in SOURCES.values()
            ],
        },
        "datasets": {
            "pzfj38_v113": {
                "rows": len(pz_rows),
                "extracted_path": str(pzfj38_jsonl.resolve()),
                "extracted_sha256": sha256_file(pzfj38_jsonl),
                "existing_media_cache_manifest": str(existing_pz_manifest.resolve()) if existing_pz else None,
                "existing_cached_rows": existing_pz.get("derived", {}).get("rows") if existing_pz else 0,
                "existing_unique_images": existing_pz.get("derived", {}).get("unique_images") if existing_pz else 0,
            },
            "qype19_v7": qype_summary,
            "zltbjg_v2": video_summary,
        },
        "media": {
            "qype19_v7": qype_media,
            "zltbjg_v2": video_media,
        },
        "limits": {
            "per_image_bytes": IMAGE_CAP_BYTES,
            "per_video_bytes": VIDEO_CAP_BYTES,
            "total_video_bytes": TOTAL_VIDEO_CAP_BYTES,
        },
    }
    report["report_sha256"] = sha256_json(report)
    write_json(MANIFEST_PATH, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-workers", type=int, default=24)
    parser.add_argument("--video-workers", type=int, default=8)
    args = parser.parse_args()
    report = prepare(args.image_workers, args.video_workers)
    print(
        json.dumps(
            {
                "manifest": str(MANIFEST_PATH.resolve()),
                "report_sha256": report["report_sha256"],
                "datasets": {name: {key: value for key, value in data.items() if key in {"rows", "unique_media", "total_media_bytes", "all_media_decoded"}} for name, data in report["datasets"].items()},
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
