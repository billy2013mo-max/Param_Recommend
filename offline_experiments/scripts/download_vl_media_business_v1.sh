#!/usr/bin/env bash
# Download the frozen image/video dataset objects required by the VL campaign.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." >/dev/null 2>&1 && pwd)"
SOURCE_DOWNLOADER="${VL_BLOB_DOWNLOADER:-/wanqing-develop/luowenjing/tianmu-video-intelligence-model-factory/model_factory_evq/blob_download.sh}"
DESTINATION_ROOT="${VL_MEDIA_DATA_ROOT:-${PROJECT_ROOT}/offline_experiments/data/vl_media_business_v1/raw}"
BUCKET="${VL_BLOB_BUCKET:-infra-ai-infra-storage}"
ENVIRONMENT="${VL_BLOB_ENVIRONMENT:-idc}"

if [[ ! -f "$SOURCE_DOWNLOADER" ]]; then
  echo "BlobStore downloader not found: $SOURCE_DOWNLOADER" >&2
  exit 2
fi

download_and_verify() {
  local object_key="$1"
  local relative_path="$2"
  local expected_size="$3"
  local expected_sha256="$4"
  local destination="${DESTINATION_ROOT}/${relative_path}"
  local actual_size
  local actual_sha256

  BS3_BUCKET="$BUCKET" bash "$SOURCE_DOWNLOADER" \
    "$ENVIRONMENT" download-key "$object_key" "$destination"

  actual_size="$(wc -c < "$destination" | tr -d ' ')"
  if [[ "$actual_size" != "$expected_size" ]]; then
    echo "Size mismatch for $object_key: expected=$expected_size actual=$actual_size" >&2
    exit 3
  fi
  actual_sha256="$(sha256sum "$destination" | awk '{print $1}')"
  if [[ "$actual_sha256" != "$expected_sha256" ]]; then
    echo "SHA-256 mismatch for $object_key" >&2
    exit 4
  fi
  echo "Verified: $object_key ($actual_size bytes, sha256=$actual_sha256)"
}

download_and_verify \
  "datasets/dataset-pzfj38-1774860803/113/publish/dataset-pzfj38-1774860803-V113.zip" \
  "dataset-pzfj38-1774860803/113/publish/dataset-pzfj38-1774860803-V113.zip" \
  "1469857" \
  "d67bc5424a857993b60d6aa18d7f9cdb0648aa733b0078d0270e42cf5b2c0d57"

download_and_verify \
  "datasets/dataset-qype19-1770794488/7/publish/dataset-qype19-1770794488-V7.zip" \
  "dataset-qype19-1770794488/7/publish/dataset-qype19-1770794488-V7.zip" \
  "92150" \
  "ca5a8c32be87553bb0c7e8250e36e2d38ef1e25a274c0cd9c0da00ff81e6b7aa"

download_and_verify \
  "datasets/dataset-zltbjg-1780992949/2/publish/dataset-zltbjg-1780992949-V2.jsonl" \
  "dataset-zltbjg-1780992949/2/publish/dataset-zltbjg-1780992949-V2.jsonl" \
  "1341992" \
  "37ea8448c0903e543e2772824ac7735d9b737a7cc38d8f9f3be61d6288062fc6"

echo "BlobStore dataset download completed: $DESTINATION_ROOT"
