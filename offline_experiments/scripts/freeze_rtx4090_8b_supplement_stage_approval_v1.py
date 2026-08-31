#!/usr/bin/env python3
"""Freeze a v2 approval design for one stage of the RTX 4090 8B supplement.

Why this exists
---------------
``validate_setup.py`` freezes a v1 design (``file_sha256`` + ``matrix_summary``
only).  The shared approval gate in ``run_job.verify_approval`` now requires the
v2 shape: ``allowed_job_ids``, ``execution_order``, ``queue_binding`` (+ its
sha), ``runtime_identity`` / ``runtime_fingerprint_sha256``, ``runtime_patch``
and ``provenance_binding``.  ``validate_setup.py`` has never emitted
``allowed_job_ids``, so the campaign-pipeline path cannot launch as-is.

``freeze_rtx4090_generalization_live_approval.py`` does emit v2, but it expects
to be copied to a campaign root that also carries ``scripts/``,
``frozen_predictions_before_holdout.json`` and ``EXPERIMENT_DESIGN.md``.  This
campaign has none of those, so this is the same freeze targeted at this
supplement's actual file set.

Approval is per stage because the throughput and scaling queues carry no
``job_id`` until memory results materialise them -- they cannot be whitelisted
in advance.

Usage
-----
    python3 freeze_rtx4090_8b_supplement_stage_approval_v1.py \
        --campaign campaigns/rtx4090_20260830_8b \
        --stage memory --promote
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from approval_gate import build_provenance_binding, build_queue_binding
from common import read_json, read_jsonl, sha256_file, sha256_json, write_json
from promote_approval_candidate import promote_candidate
from run_job import live_runtime_identity, live_runtime_patch

STAGE_QUEUES = {
    "memory": "matrix/memory_boundary_families.jsonl",
    "throughput": "matrix/throughput_requests.jsonl",
    "scaling": "matrix/strong_scaling_requests.jsonl",
}

AUTHORIZATION = (
    "用户于 2026-08-31 在看到冻结设计 SHA 后明确批准本次补数：给 RTX 4090 补 "
    "Qwen3-8B LoRA 的显存与吞吐数据，把 4090 侧模型规模从三个扩到四个，并让与 "
    "H800 的跨卡锚点增加一个。动机是双卡联合 V3 显存模型在 4090 侧折外误拒率 "
    "24.3%（H800 侧 4.06%），诊断为数据覆盖不足（29 个 source、3 个模型规模），"
    "该诊断是假设，需重拟合后用数字检验。执行前已冻结全网格预测（29 格预测可放行 / "
    "61 格预测拒绝）以便预测-实测对照。仅在空闲的 GPU 0,1,2,3 上执行，不得终止或"
    "干扰他人进程。"
)


def freeze(campaign_root: Path, stage: str, frozen_predictions: Path) -> Path:
    if stage not in STAGE_QUEUES:
        raise SystemExit(f"未知阶段 {stage!r}，可选：{sorted(STAGE_QUEUES)}")
    root = campaign_root.resolve()
    queue = (root / STAGE_QUEUES[stage]).resolve()
    rows = read_jsonl(queue)
    if not rows:
        raise SystemExit(f"队列为空：{queue}")
    missing_ids = [i for i, row in enumerate(rows) if not row.get("job_id")]
    if missing_ids:
        raise SystemExit(
            f"{stage} 队列有 {len(missing_ids)} 行没有 job_id，无法预先白名单。"
            " throughput / scaling 要等 memory 结果物化之后才有 job_id。"
        )

    experiment = read_json(root / "config" / "experiment.json")
    campaign_id = str(experiment["campaign_id"])
    scope = experiment["training_scope"]

    queue_binding = build_queue_binding(queue, rows, root)
    provenance_binding = build_provenance_binding(root)
    runtime_identity = live_runtime_identity()
    runtime_patch = live_runtime_patch()
    ids = [str(row["job_id"]) for row in rows]

    manifest_paths = [
        queue,
        root / "config" / "experiment.json",
        root / "config" / "models.json",
        root / "config" / "hardware.json",
        root / "artifacts" / "provenance.json",
        root / "artifacts" / "model_inventory.json",
        root / "artifacts" / "dataset_analysis.json",
        root / "artifacts" / "preprocessing_validation.json",
        root / "artifacts" / "smoke_validation.json",
    ]
    absent = [str(p) for p in manifest_paths if not p.is_file()]
    if absent:
        raise SystemExit(f"冻结所需文件缺失：{absent}")
    manifest = {
        str(p.relative_to(root)): sha256_file(p) for p in manifest_paths
    }

    stage_binding = {
        "allowed_job_ids": ids,
        "queue_path": queue_binding["path"],
        "queue_sha256": queue_binding["sha256"],
        "ordered_job_ids": queue_binding["ordered_job_ids"],
        "ordered_job_payload_sha256": queue_binding[
            "ordered_job_payload_sha256"
        ],
        "job_payload_sha256": queue_binding["job_payload_sha256"],
    }
    design = {
        "schema_version": 1,
        "training_started": False,
        "design_purpose": f"{campaign_id}:{stage}",
        "file_sha256": manifest,
        "execution_order": [stage],
        "allowed_job_ids": ids,
        "authorized_gpu_ids": list(scope["gpu_ids"]),
        "max_gpu_count": int(scope["max_gpu_count"]),
        "runtime_identity": runtime_identity,
        "runtime_fingerprint_sha256": sha256_json(runtime_identity),
        "runtime_patch": runtime_patch,
        "provenance_binding": provenance_binding,
        "queue_binding": queue_binding,
        # The generic validator retains this historical field name.  It is an
        # exact mirror of the bound queue, not a semantic claim about screening.
        "throughput_screen_delta": stage_binding,
        "predict_then_measure": {
            "frozen_predictions_path": str(frozen_predictions),
            "frozen_predictions_sha256": sha256_file(frozen_predictions),
            "note": (
                "本阶段的预测在占卡之前已冻结，运行后须与实测逐格对照，"
                "而不是事后解释结果。"
            ),
        },
        "runtime_cohort_id": experiment["fixed_runtime"]["runtime_cohort_id"],
    }
    candidate = root / "artifacts" / f"approval_design_{stage}_candidate.json"
    write_json(candidate, design)
    return candidate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument(
        "--frozen-predictions",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "artifacts"
        / "rtx4090_qwen3_8b_lora_supplement_frozen_predictions_v1.json",
    )
    parser.add_argument("--promote", action="store_true")
    args = parser.parse_args()

    candidate = freeze(
        args.campaign, args.stage, args.frozen_predictions.resolve()
    )
    digest = sha256_file(candidate)
    report = promote_candidate(
        candidate_path=candidate,
        expected_candidate_sha256=digest,
        project_root=args.campaign.resolve(),
        authorization=AUTHORIZATION,
        approved_by="user",
        promote=args.promote,
    )
    print(
        json.dumps(
            {
                "stage": args.stage,
                "candidate": str(candidate),
                "candidate_sha256": digest,
                "job_ids": len(read_json(candidate)["allowed_job_ids"]),
                "promotion": report,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
