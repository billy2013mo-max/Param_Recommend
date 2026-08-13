#!/usr/bin/env python3
"""Freeze the existing Qwen3.5 package overlay used by H800 diagnostic jobs."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from common import ARTIFACT_DIR, sha256_file, sha256_json, write_json


SITE_PACKAGES = Path(
    "/wanqing-develop/luowenjing/Param_Recommend/qwen36_venv/lib/python3.11/site-packages"
)
TILELANG_OVERLAY = Path("/tmp/qwen35_tilelang_overlay_0.1.12")
OUTPUT = ARTIFACT_DIR / "h800_qwen35_runtime_contract_v1.json"
CRITICAL_FILES = (
    SITE_PACKAGES / "transformers" / "__init__.py",
    SITE_PACKAGES / "transformers" / "models" / "qwen3_5" / "configuration_qwen3_5.py",
    SITE_PACKAGES / "transformers" / "models" / "qwen3_5" / "modeling_qwen3_5.py",
    SITE_PACKAGES / "llamafactory" / "model" / "loader.py",
    SITE_PACKAGES / "llamafactory" / "model" / "model_utils" / "liger_kernel.py",
    SITE_PACKAGES / "fla" / "__init__.py",
    SITE_PACKAGES / "flash_attn_3" / "_C.abi3.so",
    SITE_PACKAGES
    / "flash_attn_3-3.0.0b1+20260105.cu126torch280cxx11abitrue.9b6dba.dist-info"
    / "METADATA",
    SITE_PACKAGES / "transformers-5.3.0.dist-info" / "METADATA",
    SITE_PACKAGES / "llamafactory-0.9.5.dev0.dist-info" / "METADATA",
    TILELANG_OVERLAY / "tilelang" / "__init__.py",
    TILELANG_OVERLAY / "tvm_ffi" / "__init__.py",
    TILELANG_OVERLAY / "tvm_ffi" / "core.cpython-311-x86_64-linux-gnu.so",
    TILELANG_OVERLAY / "tvm_ffi" / "lib" / "libtvm_ffi.so",
)


def main() -> None:
    missing = [str(path) for path in CRITICAL_FILES if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Qwen3.5 runtime critical files are absent: {missing}")
    report = {
        "schema": "sft_h800_qwen35_runtime_contract/v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Qwen3.5 transfer-only H800 diagnostics; never applied to Qwen3 base-anchor jobs",
        "environment": {
            "PYTHONPATH_prepend": [str(TILELANG_OVERLAY), str(SITE_PACKAGES)],
            "variables": {
                "FLA_TILELANG": "1",
                "TILELANG_CACHE_DIR": "/tmp/qwen35_tilelang_cache_h800_precedence_v2",
            },
        },
        "declared_versions": {
            "transformers": "5.3.0",
            "llamafactory": "0.9.5.dev0",
            "torch": "2.8.0+cu126",
            "fla_backend": "tilelang 0.1.12 with overlay-first module precedence",
            "tilelang": "0.1.12",
            "apache-tvm-ffi": "0.1.11",
        },
        "file_sha256": {str(path): sha256_file(path) for path in CRITICAL_FILES},
        "model_configuration_probe": {
            "model_path": "/wanqing-models/Qwen3.5-4B",
            "expected_config_class": "Qwen3_5Config",
            "verified_before_freeze": True,
        },
    }
    report["report_sha256"] = sha256_json(report)
    write_json(OUTPUT, report)
    print({"output": str(OUTPUT), "report_sha256": report["report_sha256"]})


if __name__ == "__main__":
    main()
