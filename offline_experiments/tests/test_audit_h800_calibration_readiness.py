from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import audit_h800_calibration_readiness as audit_module  # noqa: E402
from audit_h800_calibration_readiness import (  # noqa: E402
    NonH800ObservationError,
    audit,
    write_report,
)
from export_h800_observations import export_observations  # noqa: E402
from tests.test_export_h800_observations import EvidenceProject  # noqa: E402


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n"
            for item in rows
        ),
        encoding="utf-8",
    )


def synthetic_row(
    index: int,
    *,
    role: str,
    split_unit_id: str,
    outcome: str = "success",
    mbs: int = 2,
    cutoff_len: int = 4096,
    parameter_count: int = 8_000_000_000,
    gpu_count: int = 2,
    zero: str = "zero3",
    train_type: str = "full",
    packing: bool = False,
    evidence_class: str = "unpacked_boundary_and_efficiency",
    pair_id: str | None = None,
    pair_index: int | None = None,
    treatment: str | None = None,
    repeat: int = 0,
) -> dict:
    job = {
        "job_id": f"job-{index}",
        "model_id": f"model-{parameter_count}",
        "dataset_id": f"data-{cutoff_len}",
        "train_type": train_type,
        "gpu_count": gpu_count,
        "zero": zero,
        "gc": True,
        "mbs": mbs,
        "cutoff_len": cutoff_len,
        "packing": packing,
        "repeat": repeat,
        "calibration_evidence_class": evidence_class,
        "calibration_partition": {
            "role": role,
            "split_unit_id": split_unit_id,
            "policy": "model-length-disjoint-v1",
        },
    }
    if pair_id is not None:
        job["packing_pair"] = {
            "pair_id": pair_id,
            "order": "ABBA",
            "sequence_index": pair_index,
            "treatment": treatment,
        }
    success = outcome == "success"
    usable = outcome in {"success", "oom"}
    return {
        "schema": "sft_efficiency_observation/v2",
        "observation_id": f"observation-{index}",
        "attempt": {"execution_attempt_id": f"{index:020x}"},
        "hardware": {
            "gpu_family": "H800",
            "gpu_type": "NVIDIA H800 140GB HBM3",
            "gpu_id": "local_h800_140g",
            "profile": {
                "gpu_id": "local_h800_140g",
                "name_reported_by_driver": "NVIDIA H800",
            },
        },
        "configuration": {
            "job": job,
            "calibration_partition": job["calibration_partition"],
            "environment": {"ENABLE_CCE": "1", "FA3_VARIANT": "default"},
        },
        "outcome": {
            "class": outcome,
            "usable_for_feasibility_calibration": usable,
            "usable_for_throughput_calibration": success,
        },
        "fingerprint": {
            "quality": "complete",
            "calibration_evidence_eligible": True,
            "computed_execution_manifest_sha256": f"{index + 10_000:064x}",
            "runtime_mechanism_fingerprint_sha256": "a" * 64,
            "runtime_config": {
                "payload": {
                    "bf16": True,
                    "flash_attn": "fa3",
                    "gradient_checkpointing": True,
                    "torch_compile": False,
                    "optim": "adamw_torch_fused",
                    "use_reentrant_gc": True,
                }
            },
            "runtime_model_inventory": {
                "logical_parameter_elements": parameter_count,
            },
        },
    }


def packing_pair_rows(
    start: int,
    *,
    pair_id: str,
    role: str,
    split_unit_id: str,
    zero_stage: int,
    cutoff_len: int,
    packed_oom: bool = False,
) -> list[dict]:
    gpu_count = 1 if zero_stage == 0 else 2
    zero = "none" if zero_stage == 0 else "zero3"
    rows = []
    treatments = ("unpacked", "packed", "packed", "unpacked")
    for offset, treatment in enumerate(treatments):
        packing = treatment == "packed"
        outcome = "oom" if packed_oom and offset == 1 else "success"
        rows.append(
            synthetic_row(
                start + offset,
                role=role,
                split_unit_id=split_unit_id,
                outcome=outcome,
                mbs=1 if packing else 2,
                cutoff_len=cutoff_len,
                parameter_count=4_000_000_000 + cutoff_len,
                gpu_count=gpu_count,
                zero=zero,
                packing=packing,
                evidence_class="packing_paired_only",
                pair_id=pair_id,
                pair_index=offset,
                treatment=treatment,
                repeat=0 if offset < 2 else 1,
            )
        )
    return rows


def complete_packing_study_rows() -> list[dict]:
    rows: list[dict] = []
    index = 1
    # Both roles of every mode/stage dependency have complete in-domain
    # unpacked boundaries through a fully evidenced MBS=16 success.
    for zero_stage in (0, 3):
        for role, suffix in (("calibration", "cal"), ("holdout", "hold")):
            rows.append(
                synthetic_row(
                    index,
                    role=role,
                    split_unit_id=f"boundary-z{zero_stage}-{suffix}",
                    mbs=16,
                    cutoff_len=512 if zero_stage == 0 else 4096,
                    parameter_count=2_000_000_000 + zero_stage,
                    gpu_count=1 if zero_stage == 0 else 2,
                    zero="none" if zero_stage == 0 else "zero3",
                )
            )
            index += 1
    pair_specs = (
        ("pair-cal-z0", "calibration", "pair-cal-z0", 0, 8192),
        ("pair-cal-z3", "calibration", "pair-cal-z3", 3, 4096),
        ("pair-hold-z0", "holdout", "pair-hold-z0", 0, 2048),
        ("pair-hold-z3", "holdout", "pair-hold-z3", 3, 4096),
    )
    for pair_id, role, unit, zero_stage, cutoff_len in pair_specs:
        rows.extend(
            packing_pair_rows(
                index,
                pair_id=pair_id,
                role=role,
                split_unit_id=unit,
                zero_stage=zero_stage,
                cutoff_len=cutoff_len,
            )
        )
        index += 4
    return rows


class H800CalibrationReadinessTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def audit_rows(self, rows: list[dict], name: str = "observations.jsonl") -> dict:
        path = self.root / name
        write_rows(path, rows)
        return audit(path)

    def test_packing_selector_uses_separate_pair_acceptance(self) -> None:
        self.assertEqual(
            audit_module._required_calibration_variation({"packing": True}),
            (),
        )
        self.assertIn(
            "micro_batch_sizes",
            audit_module._required_calibration_variation({"packing": False}),
        )

    def test_complete_v2_mbs16_success_completes_supported_domain_boundary(self) -> None:
        project = EvidenceProject(self.root / "mbs16-complete")
        project.add_attempt("mbs16-success", mbs=16)

        report = self.audit_rows(export_observations(project.root))

        boundary = report["selectors"][0]["role_boundary_evidence"]["calibration"]
        self.assertTrue(boundary["complete"])
        self.assertEqual(
            boundary["completion_mode"],
            "supported_domain_fully_feasible_at_mbs16",
        )
        self.assertEqual(boundary["oom_rows"], 0)

    def test_success_below_mbs16_without_oom_does_not_complete_boundary(self) -> None:
        project = EvidenceProject(self.root / "mbs8-incomplete")
        project.add_attempt("mbs8-success", mbs=8)

        report = self.audit_rows(export_observations(project.root))

        selector = report["selectors"][0]
        boundary = selector["role_boundary_evidence"]["calibration"]
        self.assertFalse(boundary["complete"])
        self.assertIn(
            "calibration_supported_mbs_boundary_incomplete",
            selector["identifiability_blockers"],
        )

    def test_success_oom_bracket_completes_without_out_of_domain_probe(self) -> None:
        rows = [
            synthetic_row(
                1,
                role="calibration",
                split_unit_id="cal-a",
                outcome="success",
                mbs=4,
            ),
            synthetic_row(
                2,
                role="calibration",
                split_unit_id="cal-b",
                outcome="oom",
                mbs=8,
            ),
        ]
        with mock.patch.object(
            audit_module, "validate_canonical_observation", return_value=[]
        ):
            report = self.audit_rows(rows)

        boundary = report["selectors"][0]["role_boundary_evidence"]["calibration"]
        self.assertTrue(boundary["complete"])
        self.assertEqual(boundary["completion_mode"], "success_oom_bracket")
        self.assertEqual(boundary["outside_supported_mbs"], [])

    def test_mbs32_success_is_rejected_not_used_as_domain_completion(self) -> None:
        row = synthetic_row(
            1,
            role="calibration",
            split_unit_id="cal-outside",
            mbs=32,
        )
        with mock.patch.object(
            audit_module, "validate_canonical_observation", return_value=[]
        ):
            report = self.audit_rows([row])

        selector = report["selectors"][0]
        boundary = selector["role_boundary_evidence"]["calibration"]
        self.assertFalse(boundary["complete"])
        self.assertEqual(boundary["outside_supported_mbs"], [32])
        self.assertIn(
            "calibration_supported_mbs_boundary_incomplete",
            selector["identifiability_blockers"],
        )

    def test_complete_abba_packing_study_uses_unpacked_mode_stage_boundaries(self) -> None:
        rows = complete_packing_study_rows()
        with mock.patch.object(
            audit_module, "validate_canonical_observation", return_value=[]
        ):
            report = self.audit_rows(rows)

        self.assertEqual(len(report["packing_selectors"]), 1)
        packing = report["packing_selectors"][0]
        self.assertTrue(packing["ready_for_profile_candidate"])
        self.assertEqual(packing["complete_pairs"], {"calibration": 2, "holdout": 2})
        self.assertEqual(packing["packed_oom_rows"], 0)
        self.assertTrue(
            all(row["complete"] for row in packing["corresponding_unpacked_boundaries"])
        )
        self.assertNotIn("packing_pair_evidence_incomplete", report["publication_blockers"])

    def test_incomplete_abba_or_new_packed_oom_blocks_packing(self) -> None:
        rows = complete_packing_study_rows()
        # Remove the final A repeat from one calibration pair.
        rows = [
            row
            for row in rows
            if not (
                ((row["configuration"]["job"].get("packing_pair") or {}).get("pair_id"))
                == "pair-cal-z0"
                and ((row["configuration"]["job"].get("packing_pair") or {}).get("sequence_index"))
                == 3
            )
        ]
        # A packed OOM in the other calibration pair is never accepted as a
        # packing recommendation datapoint.
        for row in rows:
            metadata = row["configuration"]["job"].get("packing_pair") or {}
            if metadata.get("pair_id") == "pair-cal-z3" and metadata.get(
                "sequence_index"
            ) == 1:
                row["outcome"]["class"] = "oom"
                row["outcome"]["usable_for_throughput_calibration"] = False
        with mock.patch.object(
            audit_module, "validate_canonical_observation", return_value=[]
        ):
            report = self.audit_rows(rows)

        packing = report["packing_selectors"][0]
        self.assertFalse(packing["ready_for_profile_candidate"])
        self.assertEqual(packing["packed_oom_rows"], 1)
        self.assertIn("packing_introduced_oom", packing["calibration_blockers"])
        self.assertIn(
            "incomplete_or_invalid_abba_pairs", packing["calibration_blockers"]
        )
        self.assertIn("packing_pair_evidence_incomplete", report["publication_blockers"])

    def test_missing_corresponding_unpacked_holdout_boundary_blocks_packing(self) -> None:
        rows = complete_packing_study_rows()
        rows = [
            row
            for row in rows
            if not (
                row["configuration"]["job"].get("calibration_evidence_class")
                == "unpacked_boundary_and_efficiency"
                and row["configuration"]["job"]["zero"] == "zero3"
                and row["configuration"]["calibration_partition"]["role"]
                == "holdout"
            )
        ]
        with mock.patch.object(
            audit_module, "validate_canonical_observation", return_value=[]
        ):
            report = self.audit_rows(rows)

        packing = report["packing_selectors"][0]
        self.assertFalse(packing["ready_for_profile_candidate"])
        self.assertIn(
            "corresponding_unpacked_boundary_incomplete",
            packing["holdout_blockers"],
        )

    def test_approved_exact_h800_row_is_counted_but_not_published(self) -> None:
        project = EvidenceProject(self.root / "approved")
        project.add_attempt("approved-h800")
        rows = export_observations(project.root)

        report = self.audit_rows(rows)

        self.assertEqual(report["counts"]["observations"], 1)
        self.assertEqual(report["counts"]["complete_fingerprint"], 1)
        self.assertEqual(report["counts"]["complete_approved_exact_h800"], 1)
        self.assertEqual(report["counts"]["complete_feasibility_usable"], 1)
        self.assertEqual(report["counts"]["complete_throughput_usable"], 1)
        self.assertEqual(len(report["selectors"]), 1)
        self.assertEqual(report["selectors"][0]["roles"]["calibration"], 1)
        self.assertFalse(report["calibration_publishable"])
        self.assertFalse(report["coefficients_were_fit"])
        self.assertIn(
            "approved_holdout_acceptance_report_not_supplied",
            report["publication_blockers"],
        )

    def test_legacy_rows_never_count_as_calibration_evidence(self) -> None:
        project = EvidenceProject(self.root / "legacy")
        project.add_legacy_result()

        report = self.audit_rows(export_observations(project.root))

        self.assertEqual(report["counts"]["observations"], 1)
        self.assertEqual(report["counts"]["complete_fingerprint"], 0)
        self.assertEqual(report["counts"]["complete_approved_exact_h800"], 0)
        self.assertFalse(report["calibration_publishable"])
        self.assertIn(
            "no_complete_run_bound_execution_fingerprints",
            report["publication_blockers"],
        )

    def test_4090_identity_fails_before_embedded_evidence_is_considered(self) -> None:
        project = EvidenceProject(self.root / "wrong-family")
        project.add_attempt("h800-source-row")
        row = export_observations(project.root)[0]
        row["hardware"]["gpu_family"] = "RTX4090"
        row["hardware"]["gpu_type"] = "NVIDIA GeForce RTX 4090"

        with self.assertRaises(NonH800ObservationError):
            self.audit_rows([row])

    def test_random_job_id_containing_4090_is_not_hardware_identity(self) -> None:
        project = EvidenceProject(self.root / "h800-random-id")
        project.add_attempt("mem-a4090b-is-still-h800")

        report = self.audit_rows(export_observations(project.root))

        self.assertEqual(report["counts"]["complete_approved_exact_h800"], 1)

    def test_duplicate_observation_id_is_rejected(self) -> None:
        project = EvidenceProject(self.root / "duplicate-observation")
        project.add_attempt("one-attempt")
        row = export_observations(project.root)[0]

        with self.assertRaisesRegex(ValueError, "duplicate observation_id"):
            self.audit_rows([row, copy.deepcopy(row)])

    def test_duplicate_attempt_id_across_jobs_is_rejected(self) -> None:
        project = EvidenceProject(self.root / "duplicate-attempt")
        shared_attempt = EvidenceProject.attempt_id("shared-attempt")
        project.add_attempt("first-job", attempt_id=shared_attempt)
        project.add_attempt("second-job", attempt_id=shared_attempt)
        rows = export_observations(project.root)

        self.assertEqual(len(rows), 2)
        self.assertNotEqual(rows[0]["observation_id"], rows[1]["observation_id"])
        with self.assertRaisesRegex(ValueError, "duplicate execution attempt"):
            self.audit_rows(rows)

    def test_duplicate_execution_fingerprint_is_rejected_defense_in_depth(self) -> None:
        project = EvidenceProject(self.root / "duplicate-fingerprint")
        project.add_attempt("first-job")
        project.add_attempt("second-job")
        rows = export_observations(project.root)
        self.assertNotEqual(
            rows[0]["attempt"]["execution_attempt_id"],
            rows[1]["attempt"]["execution_attempt_id"],
        )
        rows[1]["fingerprint"]["computed_execution_manifest_sha256"] = rows[0][
            "fingerprint"
        ]["computed_execution_manifest_sha256"]

        # A content-addressed final manifest normally binds the attempt ID, so a
        # duplicate fingerprint cannot be produced without corrupting embedded
        # evidence.  Stub only that earlier validation layer to exercise the
        # audit's independent duplicate-fingerprint guard.
        with mock.patch.object(
            audit_module,
            "validate_canonical_observation",
            return_value=[],
        ):
            with self.assertRaisesRegex(ValueError, "duplicate execution fingerprint"):
                self.audit_rows(rows)

    def test_calibration_holdout_split_unit_overlap_is_reported(self) -> None:
        project = EvidenceProject(self.root / "split-leakage")
        project.add_attempt(
            "calibration-job",
            role="calibration",
            split_unit_id="qwen3_8b-long_4096",
        )
        project.add_attempt(
            "holdout-job",
            role="holdout",
            split_unit_id="qwen3_8b-long_4096",
        )
        rows = export_observations(project.root)

        report = self.audit_rows(rows)

        self.assertEqual(len(report["selectors"]), 1)
        selector = report["selectors"][0]
        self.assertEqual(selector["roles"]["calibration"], 1)
        self.assertEqual(selector["roles"]["holdout"], 1)
        self.assertIn(
            "calibration_holdout_split_units_overlap",
            selector["holdout_blockers"],
        )
        self.assertEqual(
            report["partition_audit"]["overlap"], ["qwen3_8b-long_4096"]
        )
        self.assertIn(
            "global_calibration_holdout_partition_inconsistent",
            report["publication_blockers"],
        )
        self.assertFalse(selector["ready_for_profile_candidate"])
        self.assertFalse(report["all_selectors_ready_for_profile_candidate"])

    def test_report_write_is_atomic_and_strict_json(self) -> None:
        output = self.root / "nested" / "report.json"
        report = {"schema": "test", "calibration_publishable": False}

        write_report(output, report)

        self.assertEqual(json.loads(output.read_text(encoding="utf-8")), report)
        self.assertFalse(list(output.parent.glob("*.tmp")))


if __name__ == "__main__":
    unittest.main()
