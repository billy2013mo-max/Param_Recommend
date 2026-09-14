#!/usr/bin/env python3
"""汇总全部 VL 显存观测，建统一建模表。

四个来源
--------
  1. vl_vision_activation_supplement_v1   72 行  本轮补数，patch 跨 12.6×，gc×mbs 全交叉
  2. qwen35_vl_supplement_v1              56 行  跨模型规模（11 个模型、4 个族）
  3. hybrid_vl_prospective_acceptance_v3  27 行  V3 前瞻验收，来源隔离
  4. frozen_vl_combined_formal_v1         18 行  最早的图片臂

patch 怎么补齐
--------------
来源 2/4 的画像没有 `raw_patch_units_total` 字段。但六份带该字段的画像上
`raw_patch_units / visual_tokens` 精确等于 4.000 = spatial_merge_size²，
所以缺失的 patch 按几何关系算：

    raw_patch_units = visual_tokens × spatial_merge_size²

这是处理器的确定性几何（merge 把 2×2 的 patch 合成 1 个视觉 token），不是拟合。
算出来的行标 `patch_source = derived_from_merge_geometry`，与直接读到的区分开。

两个显存指标，分工不同
----------------------
  `allocated`  张量真实占用。中心模型学这个——reserved 里含分配器碎片，
               把碎片学进模型会得出「开 GC 更耗显存」这类错误结论
               （本轮实测：开 GC 的碎片 3.36 GiB vs 关 GC 的 0.37 GiB，
                拿 reserved 比就会看到假的方向翻转）。
  `reserved`   分配器保留量，OOM 由它触发。准入上界必须覆盖它。

所以两个都留，用途分开记。

训练/评测怎么分
---------------
按**来源**分，不做随机划分。随机划分会把同一 (模型, 形状, 机制) 的重复行拆到
两边，评测集就不是分布外的了，指标会虚高。来源 3 是当初就设计成来源隔离的
前瞻验收集，且它的 27 行吞吐在 V3 里一次都没被预测过，最适合当评测集。
"""
from __future__ import annotations

import glob
import json
import os
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
ART = REPO / "artifacts"
OUT = ART / "h800_vl_memory_table_v1.json"

INVENTORY = ART / "h800_qwen35_vl_supplement_model_inventory_v1.json"
# 外推验证批补了 qwen2p5_vl_32b（从未测过，不在基础清单里）。增量清单是基础
# 清单的超集（11 个 + 1），存在就优先用它——表里每行的几何字段全部按 model_id
# 从清单取，缺条目会让该模型的表行几何为空，准入算权重就没了依据。
_INVENTORY_WITH_PROBE = ART / "h800_vl_extrap_probe_model_inventory_v1.json"
if _INVENTORY_WITH_PROBE.is_file():
    INVENTORY = _INVENTORY_WITH_PROBE
# H800 单卡容量。OOM 行只知道「真实需求 > 这个数」，是右删失下界。
CAPACITY_BYTES = 150142189568

# 通用来源注册表：(队列文件名, 表里的来源名)。
# 新活动跑完后只需在这里加一行，不要再复制一整段纳入代码——逐批手写已经导致
# 四次「跑完才发现数据没进表」。纳入口径统一：success 与 oom 分开处理，
# OOM 记成右删失下界，纯新增不取代任何行。
GENERIC_SOURCES = (
    # ("h800_vl_final_gaps_v1.jsonl", "vl_final_gaps"),
    #   ^ 2026-09-13 作废，不得纳入。该批 142 格作业本身跑成功，但工作量标签是错的：
    #     prepare 脚本按「最短档」取模板行拿数据集字段、却按目标长度另取画像，
    #     于是作业读单图短文本数据、标四图长文本画像。序列长度漂移门禁抓到 122 行
    #     偏离 >20%，最严重的 97 行记录 2032 而实测 672（+157%）。
    #     与 V3 那次「画像错绑、序列长度低记 2.9 倍」是同一类错误。
    #     修正后的重跑见 h800_vl_final_gaps_r2_v1.jsonl。
    ("h800_vl_final_gaps_r2_v1.jsonl", "vl_final_gaps_r2"),
)


def read_json(path: str | Path):
    with open(path) as handle:
        return json.load(handle)


def memory_of(job_id: str) -> dict[str, int] | None:
    """取该作业最新一次尝试所有 rank 的显存峰值（跨 rank 取 max）。"""
    attempts = glob.glob(str(REPO / "results" / job_id / "attempts" / "*"))
    if not attempts:
        return None
    newest = max(attempts, key=os.path.getmtime)
    reserved, allocated, vision = [], [], []
    for path in glob.glob(os.path.join(newest, "metrics", "summary.rank*.json")):
        summary = read_json(path)
        for key, bucket in (("max_reserved", reserved), ("max_allocated", allocated)):
            if isinstance(summary.get(key), (int, float)):
                bucket.append(int(summary[key]))
        probe = summary.get("vision_phase_memory_probe") or {}
        if isinstance(probe.get("max_reserved_during_vision"), (int, float)):
            vision.append(int(probe["max_reserved_during_vision"]))
    if not reserved:
        return None
    return {
        "reserved": max(reserved),
        "allocated": max(allocated) if allocated else None,
        "vision_phase_reserved": max(vision) if vision else None,
    }


def observed_sequence_length(job_id: str) -> float | None:
    """从 batch_shape_evidence 取实测的平均逻辑序列长度。

    这是判定「作业到底跑的是哪一档画像」的唯一可信依据。画像文件名和
    media_tier 标签都可能与作业实际参数不符——V3 那 27 行就是标签写 low、
    绑定 low 档画像，实际按 high 档像素跑。
    """
    attempts = glob.glob(str(REPO / "results" / job_id / "attempts" / "*"))
    if not attempts:
        return None
    newest = max(attempts, key=os.path.getmtime)
    lengths: list[int] = []
    for path in glob.glob(os.path.join(newest, "metrics", "summary.rank*.json")):
        evidence = read_json(path).get("batch_shape_evidence") or {}
        for microbatch in evidence.get("measured_microbatches") or []:
            lengths.extend(microbatch.get("logical_sequence_lengths") or [])
    return sum(lengths) / len(lengths) if lengths else None


def resolve_profile(path: str, job_id: str, declared_max_pixels: int | None) -> tuple[str, str]:
    """挑出与作业实际运行相符的那份画像。

    优先按作业自报的 `image_max_pixels` 与画像 binding 对齐；对不上时，用实测
    序列长度在 low/high 两档里选更接近的那一档，并把这次改判记录下来。

    这条存在的原因：V3 的 27 行作业参数是 image_max_pixels=589824（high 档），
    却绑了 low 档画像（binding 写 200704），导致序列长度被低记 2.9 倍。当时没有
    像素档门禁，错误一路带到了验收指标里。
    """
    candidates = [path]
    for a, b in ((".low.", ".high."), (".high.", ".low.")):
        if a in path:
            candidates.append(path.replace(a, b))

    # 第一优先：作业自报的像素档与画像 binding 一致。
    if declared_max_pixels is not None:
        for candidate in candidates:
            if not os.path.isfile(candidate):
                continue
            binding = read_json(candidate).get("processor_binding") or {}
            if binding.get("image_max_pixels") == declared_max_pixels:
                note = "matched_declared_max_pixels" if candidate == path else (
                    "reassigned_by_declared_max_pixels"
                )
                return candidate, note

    # 退路：拿实测序列长度选最近的一档。
    observed = observed_sequence_length(job_id)
    if observed is not None and len(candidates) > 1:
        best, best_gap, note = path, None, "kept_declared_profile"
        for candidate in candidates:
            if not os.path.isfile(candidate):
                continue
            summary = read_json(candidate).get("summary") or {}
            block = summary.get("total_tokens")
            if not isinstance(block, dict) or block.get("mean") is None:
                continue
            gap = abs(observed - float(block["mean"]))
            if best_gap is None or gap < best_gap:
                best, best_gap = candidate, gap
        if best != path:
            note = "reassigned_by_observed_sequence_length"
        return best, note

    return path, "kept_declared_profile"


def profile_features(path: str) -> dict[str, Any] | None:
    """读画像的工作量均值；patch 缺失时按 merge 几何补出来。"""
    try:
        document = read_json(path)
    except (OSError, json.JSONDecodeError):
        return None
    summary = document.get("summary") or {}
    binding = document.get("processor_binding") or {}

    def mean(key: str) -> float | None:
        block = summary.get(key)
        if isinstance(block, dict) and block.get("mean") is not None:
            return float(block["mean"])
        return None

    visual = mean("visual_tokens_total")
    patch = mean("raw_patch_units_total")
    patch_source = "profile_field"
    if patch is None and visual is not None:
        merge = int(binding.get("spatial_merge_size") or 0)
        if merge <= 0:
            return None
        # merge 把 merge×merge 个 patch 合成一个视觉 token，比值恒为 merge²，
        # 在六份带该字段的画像上精确成立（4.000）。
        patch = visual * merge * merge
        patch_source = "derived_from_merge_geometry"
    return {
        "text_tokens": mean("text_tokens"),
        "visual_tokens": visual,
        "total_tokens": mean("total_tokens"),
        "raw_patch_units": patch,
        "patch_source": patch_source,
        "pixel_values_elements": mean("pixel_values_elements_total"),
        "images_per_sample": mean("images_per_sample"),
        "spatial_merge_size": binding.get("spatial_merge_size"),
        "image_max_pixels": binding.get("image_max_pixels"),
    }


def model_features() -> dict[str, dict[str, Any]]:
    features = {}
    for entry in read_json(INVENTORY)["models"]:
        components = entry.get("component_parameter_estimates") or {}
        geometry = entry.get("vision_geometry") or {}
        features[entry["id"]] = {
            "family": entry.get("family"),
            "hidden_size": entry.get("hidden_size"),
            "num_hidden_layers": entry.get("num_hidden_layers"),
            "total_parameters": entry.get("actual_parameters"),
            "language_parameters": components.get("language_model_or_other"),
            "vision_tower_parameters": components.get("vision_tower"),
            "vision_depth": geometry.get("depth"),
            "vision_hidden_size": geometry.get("hidden_size"),
            "is_moe": entry.get("family") == "qwen3_vl_moe",
        }
    return features


def row(source: str, job_id: str, model_id: str, config: dict, workload: dict,
        memory: dict, models: dict, profile_note: str = "kept_declared_profile",
        profile_path: str = "") -> dict[str, Any]:
    return {
        "job_id": job_id,
        "source": source,
        "model_id": model_id,
        "profile_resolution": profile_note,
        "profile_used": os.path.basename(profile_path),
        **{k: config.get(k) for k in ("gpu_count", "zero_stage", "gc", "mbs", "media_tier")},
        **workload,
        "reserved_bytes": memory["reserved"],
        "allocated_bytes": memory["allocated"],
        "vision_phase_reserved_bytes": memory["vision_phase_reserved"],
        **(models.get(model_id) or {}),
    }


def collect() -> list[dict[str, Any]]:
    models = model_features()
    rows: list[dict[str, Any]] = []

    # 来源 1：本轮补数
    for job in (json.loads(line) for line in
                open(REPO / "matrix" / "h800_vl_vision_activation_supplement_v1.jsonl")):
        memory = memory_of(job["job_id"])
        resolved, note = resolve_profile(job["dataset_profile_path"], job["job_id"],
                                         job.get("image_max_pixels"))
        workload = profile_features(resolved)
        if not (memory and workload):
            continue
        rows.append(row("vision_activation_v1", job["job_id"], job["model_id"],
                        {"gpu_count": job["gpu_count"], "zero_stage": job["zero_stage"],
                         "gc": bool(job["gc"]), "mbs": int(job["mbs"]),
                         "media_tier": job["media_tier"]},
                        workload, memory, models, note, resolved))

    # 来源 2：qwen35_vl 补充实验
    profiles = {(p["model_id"], p["tier"]): p["path"]
                for p in read_json(ART / "h800_qwen35_vl_supplement_processor_profiles_manifest_v1.json")["profiles"]}
    zero_map = {"none": 0, "zero1": 1, "zero2": 2, "zero3": 3}
    for record in read_json(ART / "h800_qwen35_vl_supplement_results_v1.json")["rows"]:
        if not record.get("media_tier") or record["classification"] != "success":
            continue
        path = profiles.get((record["model_id"], record["media_tier"]))
        memory = memory_of(record["job_id"])
        note = "kept_declared_profile"
        if path:
            path, note = resolve_profile(path, record["job_id"], None)
        workload = profile_features(path) if path else None
        if not (memory and workload):
            continue
        rows.append(row("qwen35_vl_supplement", record["job_id"], record["model_id"],
                        {"gpu_count": record["gpu_count"],
                         "zero_stage": zero_map[record["zero"]],
                         "gc": bool(record["gc"]), "mbs": int(record["mbs"]),
                         "media_tier": record["media_tier"]},
                        workload, memory, models, note, path or ""))

    # 来源 3：V3 前瞻验收（评测集）
    frozen = {p["request_id"]: p for p in
              read_json(ART / "h800_hybrid_vl_prospective_frozen_predictions_v3.json")["vl_image"]["predictions"]}
    for record in read_json(ART / "h800_hybrid_vl_prospective_acceptance_report_v3.json")["rows"]:
        if record["track"] != "vl_image" or record["classification"] != "success":
            continue
        prediction = frozen[record["job_id"]]
        configuration = prediction["configuration"]
        memory_bytes = memory_of(record["job_id"])
        resolved, note = resolve_profile(
            prediction["dataset_profile"]["path"], record["job_id"],
            configuration.get("image_max_pixels"),
        )
        workload = profile_features(resolved)
        if not (memory_bytes and workload):
            continue
        rows.append(row("prospective_v3", record["job_id"], record["model_id"],
                        {"gpu_count": configuration["gpu_count"],
                         "zero_stage": configuration["zero_stage"],
                         "gc": bool(configuration["gradient_checkpointing"]),
                         "mbs": int(configuration["physical_mbs"]),
                         "media_tier": "low"},
                        workload, memory_bytes, models, note, resolved))

    # 来源 5：视觉占比扫描（C5，108 格）—— 把 0.4–0.8 的空隙填上
    for job in (json.loads(line) for line in
                open(REPO / "matrix" / "h800_vl_visual_share_sweep_v1.jsonl")):
        memory = memory_of(job["job_id"])
        resolved, note = resolve_profile(job["dataset_profile_path"], job["job_id"],
                                         job.get("image_max_pixels"))
        workload = profile_features(resolved)
        if not (memory and workload):
            continue
        rows.append(row("visual_share_sweep", job["job_id"], job["model_id"],
                        {"gpu_count": job["gpu_count"], "zero_stage": job["zero_stage"],
                         "gc": bool(job["gc"]), "mbs": int(job["mbs"]),
                         "media_tier": job["media_tier"]},
                        workload, memory, models, note, resolved))

    # 来源 6：4.57.1 重跑（120 格）—— 取代来源 1/5 里 Qwen2.5-VL / Qwen3-VL 的
    # 那 120 行（它们误跑在 5.3.0 上，同配置显存差 2.6 倍）。
    superseded: set[str] = set()
    for job in (json.loads(line) for line in
                open(REPO / "matrix" / "h800_vl_tf4571_rerun_v1.jsonl")):
        superseded.add(str(job["supersedes_job_id"]))
        memory = memory_of(job["job_id"])
        resolved, note = resolve_profile(job["dataset_profile_path"], job["job_id"],
                                         job.get("image_max_pixels"))
        workload = profile_features(resolved)
        if not (memory and workload):
            continue
        rows.append(row("vl_tf4571_rerun", job["job_id"], job["model_id"],
                        {"gpu_count": job["gpu_count"], "zero_stage": job["zero_stage"],
                         "gc": bool(job["gc"]), "mbs": int(job["mbs"]),
                         "media_tier": job["media_tier"]},
                        workload, memory, models, note, resolved))
    # 把被取代的 5.3.0 行剔除：它们的运行时不是这两个模型的生产版本。
    rows = [r for r in rows if r["job_id"] not in superseded]

    # 来源 7：多卡 × ZeRO 分片扫描（40 格）—— 唯一让同一画像在 1/2/4 卡都跑过的
    # 数据，卡数效应因此可以从模型规模里分离出来。
    for job in (json.loads(line) for line in
                open(REPO / "matrix" / "h800_vl_multigpu_sweep_v1.jsonl")):
        memory = memory_of(job["job_id"])
        resolved, note = resolve_profile(job["dataset_profile_path"], job["job_id"],
                                         job.get("image_max_pixels"))
        workload = profile_features(resolved)
        if not (memory and workload):
            continue
        rows.append(row("vl_multigpu_sweep", job["job_id"], job["model_id"],
                        {"gpu_count": job["gpu_count"], "zero_stage": job["zero_stage"],
                         "gc": bool(job["gc"]), "mbs": int(job["mbs"]),
                         "media_tier": job["media_tier"]},
                        workload, memory, models, note, resolved))

    # 来源 8：多卡路由填厚（288 格）—— 让 16 个多卡路由各自有足够样本，
    # 支撑分路由建模。
    for job in (json.loads(line) for line in
                open(REPO / "matrix" / "h800_vl_route_fill_v1.jsonl")):
        memory = memory_of(job["job_id"])
        resolved, note = resolve_profile(job["dataset_profile_path"], job["job_id"],
                                         job.get("image_max_pixels"))
        workload = profile_features(resolved)
        if not (memory and workload):
            continue
        rows.append(row("vl_route_fill", job["job_id"], job["model_id"],
                        {"gpu_count": job["gpu_count"], "zero_stage": job["zero_stage"],
                         "gc": bool(job["gc"]), "mbs": int(job["mbs"]),
                         "media_tier": job["media_tier"]},
                        workload, memory, models, note, resolved))

    # 来源 9：OOM 边界 + 微批轴（v1 有效 37 格 + v2 全部 39 格）。
    # 这是唯一带 OOM 样本的来源——27 个 cuda_oom_confirmed 的右删失下界，
    # 准入上界靠它才定得出来。v1 里 39 格 incomplete_metrics 由 v2 取代
    # （根因是 ga 字段照抄 mbs=1 的 64，与运行时 target_gbs/(卡×mbs) 不符）。
    superseded_by_v2: set[str] = set()
    for job in (json.loads(line) for line in
                open(REPO / "matrix" / "h800_vl_oom_boundary_v2.jsonl")):
        if job.get("supersedes_job_id"):
            superseded_by_v2.add(str(job["supersedes_job_id"]))

    for queue_name in ("h800_vl_oom_boundary_v1.jsonl", "h800_vl_oom_boundary_v2.jsonl"):
        for job in (json.loads(line) for line in open(REPO / "matrix" / queue_name)):
            if str(job["job_id"]) in superseded_by_v2:
                continue
            status_paths = glob.glob(str(REPO / "results" / str(job["job_id"])
                       / "attempts" / "*" / "status.json"))
            if not status_paths:
                continue
            classification = read_json(
              Path(max(status_paths, key=os.path.getmtime))).get("classification")
            if classification not in ("success", "oom"):
                continue
            resolved, note = resolve_profile(job["dataset_profile_path"], job["job_id"],
                                             job.get("image_max_pixels"))
            workload = profile_features(resolved)
            if not workload:
                continue
            if classification == "oom":
                # OOM 行没有可信峰值（崩溃时的读数不是真实需求量），
                # 只记它是右删失下界：真实需求 > 卡容量。
                rows.append({
                    **{k: job.get(k) for k in ("gpu_count", "zero_stage", "mbs")},
                    "job_id": job["job_id"], "source": "vl_oom_boundary",
                    "model_id": job["model_id"], "gc": bool(job["gc"]),
                    "media_tier": job.get("media_tier"),
                    "classification": "oom",
                    "profile_resolution": note,
                    "profile_used": os.path.basename(resolved),
                    **workload,
                    "reserved_bytes": None, "allocated_bytes": None,
                    "vision_phase_reserved_bytes": None,
                    "right_censored_lower_bound_bytes": CAPACITY_BYTES,
                    **(models.get(job["model_id"]) or {}),
                })
                continue
            memory = memory_of(job["job_id"])
            if not memory:
                continue
            entry = row("vl_oom_boundary", job["job_id"], job["model_id"],
                        {"gpu_count": job["gpu_count"], "zero_stage": job["zero_stage"],
                         "gc": bool(job["gc"]), "mbs": int(job["mbs"]),
                         "media_tier": job.get("media_tier")},
                        workload, memory, models, note, resolved)
            entry["classification"] = "success"
            rows.append(entry)

    # 来源 4：最早的图片臂
    for job in (json.loads(line) for line in
                open(REPO / "matrix" / "h800_frozen_vl_combined_formal_v1.jsonl")):
        if job.get("arm_id") != "real_image":
            continue
        memory = memory_of(job["job_id"])
        resolved, note = resolve_profile(job["dataset_profile_path"], job["job_id"],
                                         job.get("image_max_pixels"))
        workload = profile_features(resolved)
        if not (memory and workload):
            continue
        rows.append(row("frozen_vl_formal", job["job_id"], job["model_id"],
                        {"gpu_count": job.get("gpu_count"), "zero_stage": job.get("zero_stage"),
                         "gc": bool(job.get("gc")), "mbs": int(job.get("mbs")),
                         "media_tier": job.get("media_tier")},
                        workload, memory, models, note, resolved))

    # 来源 10：多卡 4.57.1 重跑（430 格）—— 取代来源 7/8/9 里 Qwen2.5-VL / Qwen3-VL
    # 的多卡行。那些行的 worker 跑在 5.3.0 上：torchrun 的 shebang 在 2026-07-28
    # 被改成指向 qwen36_venv，多卡作业因此不论模型族都拿到 5.3.0，而这两个模型
    # 生产用 4.57.1，同配置显存差 2.6 倍。
    #
    # 这一批同时覆盖成功与 OOM 两种结局，所以按来源 9 的方式分开处理：
    # 4.57.1 比 5.3.0 省显存，原先在 5.3.0 下炸的格子换到 4.57.1 未必炸，
    # OOM 边界是重新找出来的，不是照搬。
    rerun_queue = REPO / "matrix" / "h800_vl_multigpu_tf4571_rerun_v1.jsonl"
    superseded_by_rerun: set[str] = set()
    if rerun_queue.is_file():
        for job in (json.loads(line) for line in open(rerun_queue)):
            if job.get("supersedes_job_id"):
                superseded_by_rerun.add(str(job["supersedes_job_id"]))
            status_paths = glob.glob(str(REPO / "results" / str(job["job_id"])
                                         / "attempts" / "*" / "status.json"))
            if not status_paths:
                continue
            classification = read_json(
                Path(max(status_paths, key=os.path.getmtime))).get("classification")
            if classification not in ("success", "oom"):
                continue
            resolved, note = resolve_profile(job["dataset_profile_path"], job["job_id"],
                                             job.get("image_max_pixels"))
            workload = profile_features(resolved)
            if not workload:
                continue
            if classification == "oom":
                rows.append({
                    **{k: job.get(k) for k in ("gpu_count", "zero_stage", "mbs")},
                    "job_id": job["job_id"], "source": "vl_multigpu_tf4571_rerun",
                    "model_id": job["model_id"], "gc": bool(job["gc"]),
                    "media_tier": job.get("media_tier"),
                    "classification": "oom",
                    "profile_resolution": note,
                    "profile_used": os.path.basename(resolved),
                    **workload,
                    "reserved_bytes": None, "allocated_bytes": None,
                    "vision_phase_reserved_bytes": None,
                    "right_censored_lower_bound_bytes": CAPACITY_BYTES,
                    **(models.get(job["model_id"]) or {}),
                })
                continue
            memory = memory_of(job["job_id"])
            if not memory:
                continue
            entry = row("vl_multigpu_tf4571_rerun", job["job_id"], job["model_id"],
                        {"gpu_count": job["gpu_count"], "zero_stage": job["zero_stage"],
                         "gc": bool(job["gc"]), "mbs": int(job["mbs"]),
                         "media_tier": job.get("media_tier")},
                        workload, memory, models, note, resolved)
            entry["classification"] = "success"
            rows.append(entry)

    # 来源 11：跨模型外推验证批（74 格）。7 个模型跑同一条配置（2/4 卡 ZeRO-3
    # 检查点开），把参数量维度铺开，用来检验「同一条线内结构相同、只有参数量
    # 不同 -> 系数可外推」这个假设。
    #
    # 这批是纯新增，不取代任何已有行：它测的路由要么此前没有数据，要么只有
    # 2-4 行。同样覆盖成功与 OOM 两种结局，按来源 9/10 的方式分开处理。
    probe_queue = REPO / "matrix" / "h800_vl_extrap_probe_v1.jsonl"
    if probe_queue.is_file():
        for job in (json.loads(line) for line in open(probe_queue)):
            status_paths = glob.glob(str(REPO / "results" / str(job["job_id"])
                                         / "attempts" / "*" / "status.json"))
            if not status_paths:
                continue
            classification = read_json(
                Path(max(status_paths, key=os.path.getmtime))).get("classification")
            if classification not in ("success", "oom"):
                continue
            resolved, note = resolve_profile(job["dataset_profile_path"], job["job_id"],
                                             job.get("image_max_pixels"))
            workload = profile_features(resolved)
            if not workload:
                continue
            if classification == "oom":
                rows.append({
                    **{k: job.get(k) for k in ("gpu_count", "zero_stage", "mbs")},
                    "job_id": job["job_id"], "source": "vl_extrap_probe",
                    "model_id": job["model_id"], "gc": bool(job["gc"]),
                    "media_tier": job.get("media_tier"),
                    "classification": "oom",
                    "profile_resolution": note,
                    "profile_used": os.path.basename(resolved),
                    **workload,
                    "reserved_bytes": None, "allocated_bytes": None,
                    "vision_phase_reserved_bytes": None,
                    "right_censored_lower_bound_bytes": CAPACITY_BYTES,
                    **(models.get(job["model_id"]) or {}),
                })
                continue
            memory = memory_of(job["job_id"])
            if not memory:
                continue
            entry = row("vl_extrap_probe", job["job_id"], job["model_id"],
                        {"gpu_count": job["gpu_count"], "zero_stage": job["zero_stage"],
                         "gc": bool(job["gc"]), "mbs": int(job["mbs"]),
                         "media_tier": job.get("media_tier")},
                        workload, memory, models, note, resolved)
            entry["classification"] = "success"
            rows.append(entry)

    # 来源 12：跨模型外推验证批第二轮（142 格）。三层：把 10 条 ZeRO-3 路由补到
    # 生产门槛（10 行）、给这些路由补 OOM 右删失下界、并开出关检查点轴。
    #
    # 与来源 11 同样是纯新增，不取代任何行。关检查点层有 27 格 OOM——点位按
    # 「关检查点后激活系数放大 8.1 倍」估的，而该倍数实测范围 2.5-17.2 很不稳，
    # 估乐观了。OOM 行本身是准入模型要的右删失下界，不是浪费。
    probe_r2_queue = REPO / "matrix" / "h800_vl_extrap_probe_r2_v1.jsonl"
    if probe_r2_queue.is_file():
        for job in (json.loads(line) for line in open(probe_r2_queue)):
            status_paths = glob.glob(str(REPO / "results" / str(job["job_id"])
                                         / "attempts" / "*" / "status.json"))
            if not status_paths:
                continue
            classification = read_json(
                Path(max(status_paths, key=os.path.getmtime))).get("classification")
            if classification not in ("success", "oom"):
                continue
            resolved, note = resolve_profile(job["dataset_profile_path"], job["job_id"],
                                             job.get("image_max_pixels"))
            workload = profile_features(resolved)
            if not workload:
                continue
            if classification == "oom":
                rows.append({
                    **{k: job.get(k) for k in ("gpu_count", "zero_stage", "mbs")},
                    "job_id": job["job_id"], "source": "vl_extrap_probe_r2",
                    "model_id": job["model_id"], "gc": bool(job["gc"]),
                    "media_tier": job.get("media_tier"),
                    "classification": "oom",
                    "profile_resolution": note,
                    "profile_used": os.path.basename(resolved),
                    **workload,
                    "reserved_bytes": None, "allocated_bytes": None,
                    "vision_phase_reserved_bytes": None,
                    "right_censored_lower_bound_bytes": CAPACITY_BYTES,
                    **(models.get(job["model_id"]) or {}),
                })
                continue
            memory = memory_of(job["job_id"])
            if not memory:
                continue
            entry = row("vl_extrap_probe_r2", job["job_id"], job["model_id"],
                        {"gpu_count": job["gpu_count"], "zero_stage": job["zero_stage"],
                         "gc": bool(job["gc"]), "mbs": int(job["mbs"]),
                         "media_tier": job.get("media_tier")},
                        workload, memory, models, note, resolved)
            entry["classification"] = "success"
            rows.append(entry)

    # 来源 13：关检查点补数（31 格）。第二轮开出了关检查点轴，但 92 格里 27 格
    # OOM、其余分散在 12 条路由上都没攒够准入门槛的 10 行，导致 6 个新模型里
    # 4 个在这条轴上完全无法作答。
    #
    # 本批点位改用每条路由自身的实测 (P,A) 选，不再用第二轮那个 8.1 倍先验——
    # 实测放大倍数中位其实是 16.6（范围 6.8-29.9），低估一倍正是第二轮 OOM 偏多
    # 的原因。本批 31 格零 OOM。
    gc_fill_queue = REPO / "matrix" / "h800_vl_gc_off_fill_v1.jsonl"
    if gc_fill_queue.is_file():
        for job in (json.loads(line) for line in open(gc_fill_queue)):
            status_paths = glob.glob(str(REPO / "results" / str(job["job_id"])
                                         / "attempts" / "*" / "status.json"))
            if not status_paths:
                continue
            classification = read_json(
                Path(max(status_paths, key=os.path.getmtime))).get("classification")
            if classification not in ("success", "oom"):
                continue
            resolved, note = resolve_profile(job["dataset_profile_path"], job["job_id"],
                                             job.get("image_max_pixels"))
            workload = profile_features(resolved)
            if not workload:
                continue
            if classification == "oom":
                rows.append({
                    **{k: job.get(k) for k in ("gpu_count", "zero_stage", "mbs")},
                    "job_id": job["job_id"], "source": "vl_gc_off_fill",
                    "model_id": job["model_id"], "gc": bool(job["gc"]),
                    "media_tier": job.get("media_tier"),
                    "classification": "oom",
                    "profile_resolution": note,
                    "profile_used": os.path.basename(resolved),
                    **workload,
                    "reserved_bytes": None, "allocated_bytes": None,
                    "vision_phase_reserved_bytes": None,
                    "right_censored_lower_bound_bytes": CAPACITY_BYTES,
                    **(models.get(job["model_id"]) or {}),
                })
                continue
            memory = memory_of(job["job_id"])
            if not memory:
                continue
            entry = row("vl_gc_off_fill", job["job_id"], job["model_id"],
                        {"gpu_count": job["gpu_count"], "zero_stage": job["zero_stage"],
                         "gc": bool(job["gc"]), "mbs": int(job["mbs"]),
                         "media_tier": job.get("media_tier")},
                        workload, memory, models, note, resolved)
            entry["classification"] = "success"
            rows.append(entry)

    # 通用来源：注册式纳入。前面来源 1-13 是逐批手写的，每开一个新活动都得复制
    # 一段几乎相同的代码——这个坑已经踩了四次（来源 11/12/13 以及本次），每次都是
    # 「跑完才发现数据进不了表、等于白跑」。这里改成注册表：新活动只要在
    # GENERIC_SOURCES 里加一行 (队列文件, 来源名) 即可，不再复制代码。
    #
    # 纳入口径与来源 11-13 完全一致：success 与 oom 两种结局分开处理，OOM 行记成
    # 右删失下界。纯新增，不取代任何已有行。
    for queue_name, source_name in GENERIC_SOURCES:
        queue_path = REPO / "matrix" / queue_name
        if not queue_path.is_file():
            continue
        for job in (json.loads(line) for line in open(queue_path)):
            status_paths = glob.glob(str(REPO / "results" / str(job["job_id"])
                                         / "attempts" / "*" / "status.json"))
            if not status_paths:
                continue
            classification = read_json(
                Path(max(status_paths, key=os.path.getmtime))).get("classification")
            if classification not in ("success", "oom"):
                continue
            resolved, note = resolve_profile(job["dataset_profile_path"], job["job_id"],
                                             job.get("image_max_pixels"))
            workload = profile_features(resolved)
            if not workload:
                continue
            if classification == "oom":
                rows.append({
                    **{k: job.get(k) for k in ("gpu_count", "zero_stage", "mbs")},
                    "job_id": job["job_id"], "source": source_name,
                    "model_id": job["model_id"], "gc": bool(job["gc"]),
                    "media_tier": job.get("media_tier"),
                    "classification": "oom",
                    "profile_resolution": note,
                    "profile_used": os.path.basename(resolved),
                    **workload,
                    "reserved_bytes": None, "allocated_bytes": None,
                    "vision_phase_reserved_bytes": None,
                    "right_censored_lower_bound_bytes": CAPACITY_BYTES,
                    **(models.get(job["model_id"]) or {}),
                })
                continue
            memory = memory_of(job["job_id"])
            if not memory:
                continue
            entry = row(source_name, job["job_id"], job["model_id"],
                        {"gpu_count": job["gpu_count"], "zero_stage": job["zero_stage"],
                         "gc": bool(job["gc"]), "mbs": int(job["mbs"]),
                         "media_tier": job.get("media_tier")},
                        workload, memory, models, note, resolved)
            entry["classification"] = "success"
            rows.append(entry)

    # 剔除被重跑取代的 5.3.0 行。即使重跑尚未产出结果也要剔——那些行的运行时
    # 不是这两个模型的生产版本，留着只会让人误用。
    rows = [r for r in rows if r["job_id"] not in superseded_by_rerun]

    # 剔除 tiny 像素档（50176）的行：该档的画像与训练实测系统性对不上。
    #
    # 2026-09-13 用 6 格探针验过，三个模型全部超 20% 漂移门禁：
    #   Qwen3-VL-8B      画像 119 / 实测 171  (-30%)
    #   Qwen3.5-27B      画像 114 / 实测 161  (-29%)
    #   Qwen2.5-VL-32B   画像 128 / 实测 166  (-23%)
    # 这一轮数据集与画像的绑定已经修对（同为 f1/short），偏差依旧，所以不是标签
    # 错配，而是训练管线在低像素档下的行为与 processor 单独调用不一致：真 processor
    # 在 max_pixels=50176 下给 45 个视觉 token，训练实测约 90。原因未查清。
    #
    # 该档原本是为「让 8B/27B/32B 的关检查点轴够到准入门槛」而造的，结论是此路不通。
    rows = [r for r in rows if r.get("image_max_pixels") != 50176]
    return rows


def main() -> None:
    rows = collect()
    from collections import Counter

    by_source = Counter(r["source"] for r in rows)
    by_patch_source = Counter(r["patch_source"] for r in rows)
    by_resolution = Counter(r["profile_resolution"] for r in rows)

    # 校验：每行记录的序列长度必须与实测吻合（20% 内），否则画像仍然选错了。
    drift = []
    for r in rows:
        observed = observed_sequence_length(r["job_id"])
        if observed and r.get("total_tokens"):
            gap = abs(r["total_tokens"] - observed) / observed
            if gap > 0.20:
                drift.append({"job_id": r["job_id"], "recorded": r["total_tokens"],
                              "observed": round(observed, 1), "gap": round(gap, 3)})
    payload = {
        "schema": "sft_h800_vl_memory_table/v1",
        "usage": "modeling_table_only_no_model_fitted_here",
        "rows_total": len(rows),
        "by_source": dict(by_source),
        "by_patch_source": dict(by_patch_source),
        "by_profile_resolution": dict(by_resolution),
        "sequence_length_drift": {
            "rows_over_20pct": len(drift),
            "detail": drift[:10],
            "gate": "每行记录的 total_tokens 必须与实测平均序列长度相差 20% 以内",
        },
        "metric_policy": {
            "center_target": "allocated_bytes",
            "center_reason": "reserved 含分配器碎片；开 GC 碎片 3.36 GiB vs 关 GC 0.37 GiB，"
                             "用 reserved 会看到假的方向翻转",
            "admission_target": "reserved_bytes",
            "admission_reason": "OOM 由 reserved 触发，上界必须覆盖它",
        },
        "split_policy": {
            "train": ["vision_activation_v1", "qwen35_vl_supplement", "frozen_vl_formal",
                      "visual_share_sweep", "vl_tf4571_rerun", "vl_multigpu_sweep",
                      "vl_route_fill", "vl_oom_boundary",
                      "vl_multigpu_tf4571_rerun", "vl_extrap_probe", "vl_extrap_probe_r2", "vl_gc_off_fill", "vl_final_gaps_r2"],
            "test": ["prospective_v3"],
            "reason": "按来源分而非随机分；随机分会把同一 (模型,形状,机制) 的重复行"
                      "拆到两边，评测集就不是分布外的了",
        },
        "rows": rows,
    }
    with OUT.open("w") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=1)

    print(f"总计 {len(rows)} 行")
    for source, count in by_source.items():
        print(f"  {source:<26}{count}")
    print(f"\npatch 来源：{dict(by_patch_source)}")
    print(f"画像归属：{dict(by_resolution)}")
    print(f"序列长度与实测偏离 >20% 的行：{len(drift)}")
    for d in drift[:5]:
        print(f"   {d['job_id']}  记录 {d['recorded']:.0f} vs 实测 {d['observed']:.0f}")
    train = [r for r in rows if r["source"] != "prospective_v3"]
    test = [r for r in rows if r["source"] == "prospective_v3"]
    print(f"\n训练集 {len(train)} 行，评测集 {len(test)} 行")
    print(f"\n→ {OUT}")


if __name__ == "__main__":
    main()
