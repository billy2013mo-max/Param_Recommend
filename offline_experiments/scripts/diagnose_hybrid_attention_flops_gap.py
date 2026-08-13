"""Quantify how much of the Qwen3.5 step-time error the geometry fix closes.

Read-only diagnostic.  It recomputes the analytic FLOPs term with
:mod:`structural_linear_work` and reports, per measured point, the ratio

    eta = observed_step_seconds / predicted_step_seconds

alongside the FLOPs correction factor.  The question it answers is narrow: of
the gap between the frozen predictions and the measurements, how much is the
historical ``linear_applications_per_pass`` underestimate, and how much is left
over for a genuine efficiency term?

This fits nothing and writes no artifact.  A FLOPs correction of ``x`` does not
imply a step-time correction of ``x`` -- the models are log-linear in many
features and only a refit gives the true response.  The ``residual_eta`` column
is therefore an upper bound on what a hybrid-attention efficiency term would
still have to explain, not a prediction.

Usage:
    python3 scripts/diagnose_hybrid_attention_flops_gap.py
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from structural_linear_work import structural_linear_work  # noqa: E402

ROOT = SCRIPTS.parent
REPO = ROOT.parent
MODEL_ROOT = Path("/wanqing-models")

QWEN35_CAMPAIGN = REPO / "offline_experiments_qwen35_tilelang_20260729"
QWEN35_REPLAY = (
    QWEN35_CAMPAIGN / "artifacts" / "qwen35_followup_frozen_model_replay.json"
)
CANARY_REPLAY = (
    ROOT / "artifacts" / "qwen25_qwen35_frozen_model_replay_2026-07-29.json"
)

# The replay artifacts keep token counts but drop
# ``computed_attention_token_pairs``, which the attention FLOPs term needs.  It
# only exists in the per-rank metrics the harness wrote, so pairs are read from
# there.  Both campaigns that produced the rows below are searched.
RESULT_ROOTS = (
    QWEN35_CAMPAIGN / "results",
    REPO / "offline_experiments_qgen_20260729" / "results",
    ROOT / "results",
)

MODEL_DIRS = {
    "qwen3p5_4b": "Qwen3.5-4B",
    "qwen2p5_14b": "Qwen2.5-14B",
    "qwen2p5_32b": "Qwen2.5-32B",
    "qwen3_4b": "Qwen3-4B",
    "qwen3_8b": "Qwen3-8B",
    "qwen3_14b": "Qwen3-14B",
    "qwen3_0p6b": "Qwen3-0.6B",
    "qwen3_1p7b": "Qwen3-1.7B",
}

HEADS = (
    "v5_structured_cross_card_single_head",
    "v4b_full_factor_two_head_rejected",
    "v4_full_factor_single_head",
)

# LoRA rank/target fixed across every campaign in this project.
LORA_RANK = 16


def _historical_linear_applications(geom: Mapping[str, Any]) -> int:
    hidden = geom["hidden_size"]
    heads = geom["num_attention_heads"]
    kv_heads = geom.get("num_key_value_heads") or heads
    head_dim = geom.get("head_dim") or hidden // heads
    kv_width = kv_heads * head_dim
    return geom["num_hidden_layers"] * (
        2 * hidden * hidden
        + 2 * hidden * kv_width
        + 3 * hidden * geom["intermediate_size"]
    ) + geom["vocab_size"] * hidden


def _text_geometry(config: Mapping[str, Any]) -> Mapping[str, Any]:
    if config.get("hidden_size") is not None:
        return config
    return config.get("text_config") or config


def _adapter_parameters(geom: Mapping[str, Any]) -> int:
    """Mirror h800_theory_basis._model_geometry for target=all, rank=16."""
    hidden = geom["hidden_size"]
    heads = geom["num_attention_heads"]
    kv_heads = geom.get("num_key_value_heads") or heads
    head_dim = geom.get("head_dim") or hidden // heads
    kv_width = kv_heads * head_dim
    return (
        LORA_RANK
        * geom["num_hidden_layers"]
        * (9 * hidden + 2 * kv_width + 3 * geom["intermediate_size"])
    )


def _total_flops(
    linear_base: int,
    adapter: int,
    computed_tokens: float,
    attention_pairs: float,
    hidden: int,
    layers: int,
    train_type: str,
    gradient_checkpointing: bool,
) -> float:
    """Reproduce h800_theory_basis.performance_basis FLOPs accounting."""
    if train_type == "full":
        linear_flops = 6 * linear_base * computed_tokens
    else:
        linear_flops = (
            4 * linear_base * computed_tokens + 6 * adapter * computed_tokens
        )
    attention_flops = 6 * layers * hidden * attention_pairs
    total = linear_flops + attention_flops
    if gradient_checkpointing:
        recompute_linear = 2 * linear_base * computed_tokens
        if train_type == "lora":
            recompute_linear += 2 * adapter * computed_tokens
        total += recompute_linear + 2 * layers * hidden * attention_pairs
    return total


def _load_config(model_id: str) -> Mapping[str, Any] | None:
    name = MODEL_DIRS.get(model_id)
    if name is None:
        return None
    path = MODEL_ROOT / name / "config.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def _job_dir(job_id: str) -> Path | None:
    for root in RESULT_ROOTS:
        candidate = root / job_id
        if candidate.is_dir():
            return candidate
    return None


def _attention_pairs_per_step(job_id: str, steps: int) -> float | None:
    """Read ``computed_attention_token_pairs`` from the newest terminal attempt.

    Mirrors ``evaluate_qwen25_qwen35_holdout._aggregate_success``: attempts are
    ordered by ``finished_unix`` and token counters are summed across ranks
    (only wall-clock is a max-over-ranks quantity).  Several mbs=1 jobs have an
    earlier failed attempt, so picking the newest matters.
    """
    job_dir = _job_dir(job_id)
    if job_dir is None:
        return None
    best_finished = float("-inf")
    best_total: float | None = None
    for attempt in (job_dir / "attempts").glob("*"):
        summaries = sorted((attempt / "metrics").glob("summary.rank*.json"))
        if not summaries:
            continue
        total = 0.0
        finished = float("-inf")
        complete = True
        for path in summaries:
            payload = json.loads(path.read_text())
            totals = payload.get("measured_totals") or {}
            pairs = totals.get("computed_attention_token_pairs")
            if pairs is None or not payload.get("measured_steps"):
                complete = False
                break
            total += float(pairs)
            finished = max(finished, float(payload.get("finished_unix") or 0.0))
        if complete and finished >= best_finished:
            best_finished = finished
            best_total = total
    if best_total is None:
        return None
    return best_total / steps


def _rows_from_replay(path: Path, cohort_key: str) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    report = json.loads(path.read_text())
    replay = report.get("throughput_replay") or {}
    section = replay.get(cohort_key) or {}
    by_job: dict[str, dict[str, Any]] = {}
    for head in HEADS:
        for row in (section.get(head) or {}).get("detailed_predictions") or []:
            job_id = row["job_id"]
            entry = by_job.setdefault(
                job_id,
                {
                    "job_id": job_id,
                    "configuration": row["configuration"],
                    "observed": row["observed"],
                    "predicted_step": {},
                },
            )
            entry["predicted_step"][head] = row["predicted"]["step_seconds"]
    return list(by_job.values())


def _analyze(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    skipped: list[str] = []
    for row in rows:
        config = row["configuration"]
        model_id = config["model_id"]
        raw = _load_config(model_id)
        if raw is None:
            skipped.append(row["job_id"])
            continue
        geom = _text_geometry(raw)
        observed = row["observed"]
        steps = observed["measured_steps"]
        computed_tokens = observed["computed_tokens"] / steps
        pairs = _attention_pairs_per_step(row["job_id"], steps)
        if pairs is None:
            skipped.append(row["job_id"])
            continue

        historical = _historical_linear_applications(geom)
        corrected = structural_linear_work(raw)["linear_applications_per_pass"]
        adapter = (
            _adapter_parameters(geom) if config["train_type"] == "lora" else 0
        )
        shared = dict(
            adapter=adapter,
            computed_tokens=computed_tokens,
            attention_pairs=pairs,
            hidden=geom["hidden_size"],
            layers=geom["num_hidden_layers"],
            train_type=config["train_type"],
            gradient_checkpointing=config["gradient_checkpointing"],
        )
        flops_old = _total_flops(linear_base=historical, **shared)
        flops_new = _total_flops(linear_base=corrected, **shared)

        entry = {
            "job_id": row["job_id"],
            "model_id": model_id,
            "train_type": config["train_type"],
            "gpu_count": config["gpu_count"],
            "zero_stage": config["zero_stage"],
            "gc": config["gradient_checkpointing"],
            "mbs": config["mbs"],
            "cutoff_len": config["cutoff_len"],
            "observed_step": observed["step_seconds"],
            "flops_ratio": flops_new / flops_old,
            "eta": {},
            "residual_eta": {},
        }
        for head, predicted in row["predicted_step"].items():
            eta = observed["step_seconds"] / predicted
            entry["eta"][head] = eta
            entry["residual_eta"][head] = eta / (flops_new / flops_old)
        out.append(entry)
    if skipped:
        # Never let a missing counter read as full coverage.
        print(
            f"  [skipped {len(skipped)} of {len(rows)} rows: no attention-pair "
            f"counter or no local config] {', '.join(sorted(skipped)[:4])}"
            + (" ..." if len(skipped) > 4 else "")
        )
    return out


def _summarize(rows: list[dict[str, Any]], label: str) -> None:
    if not rows:
        print(f"\n{label}: no rows")
        return
    print(f"\n{'=' * 108}\n{label}  ({len(rows)} points)\n{'=' * 108}")
    ratios = {row["flops_ratio"] for row in rows}
    print(
        "FLOPs correction factor: "
        + ", ".join(f"{value:.4f}" for value in sorted(ratios))
    )
    header = (
        f"{'train':5s} {'gpu':>3s} {'zero':>5s} {'gc':>2s} {'mbs':>3s} "
        f"{'obs_step':>9s} {'flops':>6s}"
    )
    for head in HEADS:
        tag = head.split("_")[0]
        header += f" {tag + '_eta':>8s} {tag + '_res':>8s}"
    print(header)
    print("-" * 108)
    for row in sorted(
        rows, key=lambda item: (item["train_type"], item["gpu_count"], item["mbs"])
    ):
        line = (
            f"{row['train_type']:5s} {row['gpu_count']:>3d} "
            f"{str(row['zero_stage']):>5s} {'Y' if row['gc'] else 'N':>2s} "
            f"{row['mbs']:>3d} {row['observed_step']:>9.3f} "
            f"{row['flops_ratio']:>6.4f}"
        )
        for head in HEADS:
            eta = row["eta"].get(head)
            res = row["residual_eta"].get(head)
            line += (
                f" {eta:>8.3f} {res:>8.3f}"
                if eta is not None
                else f" {'-':>8s} {'-':>8s}"
            )
        print(line)
    print("-" * 108)
    for head in HEADS:
        etas = [row["eta"][head] for row in rows if head in row["eta"]]
        residuals = [
            row["residual_eta"][head] for row in rows if head in row["residual_eta"]
        ]
        if not etas:
            continue
        print(
            f"{head:38s} eta mean {statistics.fmean(etas):6.3f} "
            f"median {statistics.median(etas):6.3f} "
            f"[{min(etas):.3f}, {max(etas):.3f}]   "
            f"residual mean {statistics.fmean(residuals):6.3f} "
            f"median {statistics.median(residuals):6.3f} "
            f"[{min(residuals):.3f}, {max(residuals):.3f}]"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json", type=Path, help="optional path to dump the per-point rows"
    )
    args = parser.parse_args()

    followup = _analyze(_rows_from_replay(QWEN35_REPLAY, "raw_success_only"))
    _summarize(followup, "Qwen3.5-4B followup (formal, cutoff=512)")

    canary_qwen35 = _analyze(
        _rows_from_replay(CANARY_REPLAY, "qwen35_canary_steps_2_to_5_diagnostic")
    )
    _summarize(
        [row for row in canary_qwen35 if row["model_id"] == "qwen3p5_4b"],
        "Qwen3.5-4B canary (steps 2-5, TileLang)",
    )
    _summarize(
        [row for row in canary_qwen35 if row["model_id"] != "qwen3p5_4b"],
        "Qwen2.5 controls, same TileLang environment",
    )

    controls = _analyze(
        _rows_from_replay(CANARY_REPLAY, "primary_qwen25_formal_success_only")
    )
    _summarize(controls, "Qwen2.5 formal baseline (no TileLang overlay)")

    print(
        "\nReading the columns: 'eta' is observed/predicted step time, so >1 means"
        "\nthe frozen model predicts too fast.  'res' divides eta by the FLOPs"
        "\ncorrection -- what a hybrid-attention efficiency term would still have to"
        "\nexplain if step time scaled linearly with FLOPs.  It does not, so treat"
        "\n'res' as an upper bound and confirm against a refit.  Models whose"
        "\nFLOPs ratio is 1.0000 must keep their eta unchanged: that is the"
        "\nno-regression check for the dense family."
    )

    if args.json:
        payload = {
            "qwen35_followup": followup,
            "qwen35_canary": [
                row for row in canary_qwen35 if row["model_id"] == "qwen3p5_4b"
            ],
            "qwen25_tilelang_controls": [
                row for row in canary_qwen35 if row["model_id"] != "qwen3p5_4b"
            ],
            "qwen25_formal_controls": controls,
        }
        args.json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
