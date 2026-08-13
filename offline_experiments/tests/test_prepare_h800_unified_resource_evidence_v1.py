from __future__ import annotations

import sys
import unittest
from collections import Counter, defaultdict
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from common import read_json, read_jsonl, sha256_file, sha256_json
from freeze_h800_unified_resource_evidence_v1 import (
    _validate_campaign_design,
    _validate_queue,
)
from prepare_h800_unified_resource_evidence_v1 import (
    COMMON_ANCHOR,
    CONTROLLED_IDS,
    CONTROLLED_MANIFEST,
    CONTROLLED_PROFILE_DIR,
    DESIGN,
    EXPERIMENT_CONFIG,
    FIT_SOURCES,
    GPU_IDS,
    MECHANISMS,
    ORDER_SEEDS,
    PACKING_PROFILES,
    PACKING_SETTINGS,
    QUEUE,
    _models,
    build_design,
    build_jobs,
    materialize_controlled_profiles,
)
from run_job import validate_job


class UnifiedResourceEvidenceDesignTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.controlled = materialize_controlled_profiles()
        cls.jobs = build_jobs(_models())

    def test_exact_campaign_size_and_gpu_work(self) -> None:
        roles = Counter(job["evidence_role"] for job in self.jobs)
        self.assertEqual(
            roles,
            {
                "unified_nonpacking_matched_anchor_fit": 64,
                "unified_nonpacking_boundary_probe_fit": 64,
                "critical_same_max_order_variance_diagnostic": 20,
                "unified_packing_matched_formal_fit": 72,
            },
        )
        self.assertEqual(len(self.jobs), 220)
        self.assertEqual(sum(int(job["gpu_count"]) for job in self.jobs), 600)

    def test_nonpacking_matrix_is_balanced_and_anchor_is_shared(self) -> None:
        rows = [
            job
            for job in self.jobs
            if job["evidence_role"].startswith("unified_nonpacking_")
        ]
        self.assertEqual(len(MECHANISMS), 16)
        self.assertEqual(len(FIT_SOURCES), 4)
        cells: dict[tuple[str, str], list[dict]] = defaultdict(list)
        for job in rows:
            cells[(job["split_unit_id"], job["mechanism_id"])].append(job)
        self.assertEqual(len(cells), 64)
        for key, pair in cells.items():
            self.assertEqual(len(pair), 2, key)
            roles = {job["evidence_role"] for job in pair}
            self.assertEqual(
                roles,
                {
                    "unified_nonpacking_matched_anchor_fit",
                    "unified_nonpacking_boundary_probe_fit",
                },
            )
            anchor = next(job for job in pair if "anchor" in job["evidence_role"])
            self.assertEqual(anchor["model_id"], COMMON_ANCHOR["model_id"])
            self.assertEqual(anchor["mbs"], COMMON_ANCHOR["mbs"])
            self.assertEqual(anchor["cutoff_len"], COMMON_ANCHOR["cutoff_len"])

    def test_critical_runs_are_four_scenarios_not_twenty_fit_units(self) -> None:
        rows = [
            job
            for job in self.jobs
            if job["evidence_role"]
            == "critical_same_max_order_variance_diagnostic"
        ]
        self.assertEqual({job["split_unit_id"] for job in rows}, set(CONTROLLED_IDS))
        for dataset_id in CONTROLLED_IDS:
            group = [job for job in rows if job["split_unit_id"] == dataset_id]
            self.assertEqual(len(group), len(ORDER_SEEDS))
            self.assertEqual({job["seed"] for job in group}, set(ORDER_SEEDS))
            physical = {
                (
                    job["model_id"],
                    job["gpu_count"],
                    job["zero_stage"],
                    job["gc"],
                    job["mbs"],
                    job["cutoff_len"],
                )
                for job in group
            }
            self.assertEqual(len(physical), 1)
        maxima = {
            max(
                int(row["total_tokens"])
                for row in read_jsonl(
                    CONTROLLED_PROFILE_DIR
                    / f"{dataset_id}.qwen3_nothink.jsonl"
                )
            )
            for dataset_id in CONTROLLED_IDS
        }
        self.assertEqual(len(maxima), 1)

    def test_packing_matrix_has_matched_pairs_and_real_repeats(self) -> None:
        rows = [
            job
            for job in self.jobs
            if job["evidence_role"] == "unified_packing_matched_formal_fit"
        ]
        groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
        for job in rows:
            profile_id = job["scenario_id"].split("__", 1)[0]
            groups[(profile_id, job["mechanism_id"])].append(job)
        self.assertEqual(len(groups), len(PACKING_PROFILES) * len(PACKING_SETTINGS))
        for key, group in groups.items():
            self.assertEqual(len(group), 6, key)
            for packing in (False, True):
                arm = [job for job in group if job["packing"] is packing]
                self.assertEqual(len(arm), 3, (key, packing))
                self.assertEqual({job["repeat"] for job in arm}, {0, 1, 2})
                self.assertEqual(len({job["seed"] for job in arm}), 3)
                self.assertEqual(
                    len(
                        {
                            (
                                job["model_id"],
                                job["gpu_count"],
                                job["zero_stage"],
                                job["gc"],
                                job["mbs"],
                                job["cutoff_len"],
                                job["target_gbs"],
                            )
                            for job in arm
                        }
                    ),
                    1,
                )
            unpacked = next(job for job in group if not job["packing"])
            packed = next(job for job in group if job["packing"])
            self.assertEqual(unpacked["target_gbs"], packed["target_gbs"])
            self.assertLessEqual(
                packed["packing_contract"]["expected_sample_gbs_relative_error"],
                0.10,
            )

    def test_every_job_passes_the_real_launcher_contract(self) -> None:
        for job in self.jobs:
            with self.subTest(job=job["job_id"]):
                validate_job(job)
                self.assertFalse(job["offload"])
                if int(job["gpu_count"]) == 1:
                    self.assertEqual(job["zero"], "none")
                else:
                    self.assertIn(job["zero"], {"zero2", "zero3"})

    def test_ids_payload_and_generated_files_are_deterministic(self) -> None:
        rebuilt = build_jobs(_models())
        self.assertEqual(self.jobs, rebuilt)
        self.assertEqual(len({job["job_id"] for job in self.jobs}), 220)
        self.assertEqual(read_jsonl(QUEUE), self.jobs)
        self.assertTrue(DESIGN.is_file())
        self.assertTrue(EXPERIMENT_CONFIG.is_file())

    def test_design_is_self_hashed_and_preserves_censoring_contract(self) -> None:
        design = build_design(self.jobs, self.controlled)
        unsigned = dict(design)
        report_sha = unsigned.pop("report_sha256")
        self.assertEqual(report_sha, sha256_json(unsigned))
        self.assertFalse(design["gpu_training_started"])
        self.assertFalse(design["execution_authorized"])
        self.assertFalse(design["publication_allowed"])
        self.assertTrue(design["fit_contract"]["single_shared_model"])
        self.assertTrue(design["fit_contract"]["oom_rows_are_right_censored_lower_bounds"])
        self.assertTrue(
            design["fit_contract"]["critical_seed_rows_collapse_to_four_profile_scenarios"]
        )

    def test_controlled_manifest_and_staged_eight_gpu_scope_are_bound(self) -> None:
        manifest = read_json(CONTROLLED_MANIFEST)
        for binding in manifest["outputs"]:
            path = Path(binding["path"])
            self.assertTrue(path.is_file())
            self.assertEqual(sha256_file(path), binding["sha256"])
        experiment = read_json(EXPERIMENT_CONFIG)
        scope = experiment["training_scope"]
        self.assertEqual(scope["gpu_ids"], list(GPU_IDS))
        self.assertEqual(scope["exclusive_node_gpu_ids"], list(GPU_IDS))
        self.assertEqual(scope["max_gpu_count"], 4)
        self.assertEqual(scope["global_batch_sizes"], [64, 128, 256])

    def test_freezer_accepts_only_the_exact_sealed_queue(self) -> None:
        selected, full = _validate_queue(QUEUE)
        self.assertEqual(selected, self.jobs)
        self.assertEqual(full, self.jobs)
        campaign = _validate_campaign_design(full)
        self.assertTrue(campaign["fit_contract"]["single_shared_model"])


if __name__ == "__main__":
    unittest.main()
