from __future__ import annotations

from collections import Counter
from pathlib import Path
import sys
import unittest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from prepare_h800_admission_negative_evidence_v1 import (  # noqa: E402
    GROUP_A,
    GROUP_B,
    SOURCES,
    TARGET_GBS,
    _align8,
    _data_path,
    _models,
    _profile_path,
    build_design,
    build_jobs,
)


class AdmissionNegativeEvidenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.jobs = build_jobs(_models())

    def test_job_totals_match_the_published_plan(self) -> None:
        groups = Counter(job["experiment_group"] for job in self.jobs)
        self.assertEqual(groups["group_a_missing_failures"], len(GROUP_A) * 3 * 2)
        self.assertEqual(
            groups["group_b_single_campaign_failures"], len(GROUP_B) * 3 * 2
        )
        self.assertEqual(len(self.jobs), (len(GROUP_A) + len(GROUP_B)) * 3 * 2)

    def test_every_mechanism_gets_both_a_high_and_a_low_probe(self) -> None:
        # A high-only design can leave a mechanism with failures but no safe run,
        # which is just as unfittable as having no failures.
        for mechanism in (*GROUP_A, *GROUP_B):
            roles = {
                job["probe_role"]
                for job in self.jobs
                if job["mechanism_id"] == mechanism["mechanism_id"]
            }
            self.assertEqual(roles, {"probe_high", "probe_low"}, mechanism["mechanism_id"])

    def test_every_mechanism_covers_all_three_new_sources(self) -> None:
        for mechanism in (*GROUP_A, *GROUP_B):
            sources = {
                job["split_unit_id"]
                for job in self.jobs
                if job["mechanism_id"] == mechanism["mechanism_id"]
            }
            self.assertEqual(sources, set(SOURCES), mechanism["mechanism_id"])

    def test_sources_are_disjoint_from_the_first_batch(self) -> None:
        batch_one = {
            "lora_s2_src04_live_punishment",
            "lora_s2_src12_flood_multilabel",
            "lora_s2_src18_marketing_antifraud",
            "lora_src04_account_risk_short",
            "lora_src07_flood_event_classify",
            "lora_src12_marketing_antifraud_long",
        }
        self.assertFalse(set(SOURCES) & batch_one)

    def test_high_probe_is_never_lighter_than_the_low_probe(self) -> None:
        for mechanism in (*GROUP_A, *GROUP_B):
            high, low = mechanism["high"], mechanism["low"]
            self.assertGreaterEqual(
                high["mbs"] * high["cutoff_len"],
                low["mbs"] * low["cutoff_len"],
                mechanism["mechanism_id"],
            )

    def test_gbs_contract_holds_for_every_job(self) -> None:
        for job in self.jobs:
            self.assertEqual(
                job["gpu_count"] * job["mbs"] * job["gradient_accumulation_steps"],
                TARGET_GBS,
                job["job_id"],
            )

    def test_single_card_jobs_declare_no_deepspeed(self) -> None:
        single = [job for job in self.jobs if job["gpu_count"] == 1]
        self.assertTrue(single)
        for job in single:
            self.assertEqual(job["zero"], "none", job["job_id"])
        for job in self.jobs:
            if job["gpu_count"] > 1:
                self.assertIn(job["zero"], {"zero2", "zero3"}, job["job_id"])

    def test_every_job_satisfies_the_launcher_contract(self) -> None:
        from run_job import validate_job

        for job in self.jobs:
            with self.subTest(job=job["job_id"]):
                validate_job(job)

    def test_jobs_use_the_concrete_scheduler_path(self) -> None:
        for job in self.jobs:
            self.assertEqual(job["kind"], "throughput", job["job_id"])
            self.assertNotIn("mbs_candidates", job)
            self.assertEqual(
                job["evidence_role"], "memory_admission_negative_evidence"
            )

    def test_no_two_jobs_describe_the_same_physical_run(self) -> None:
        physical = Counter(
            (
                job["mechanism_id"],
                job["split_unit_id"],
                job["model_id"],
                job["mbs"],
                job["cutoff_len"],
            )
            for job in self.jobs
        )
        self.assertEqual([k for k, v in physical.items() if v > 1], [])

    def test_effective_sequence_follows_the_m1_length_policy(self) -> None:
        for job in self.jobs:
            self.assertEqual(
                job["aligned_effective_sequence"],
                _align8(min(job["cutoff_len"], job["raw_profile_max"])),
            )
            self.assertEqual(job["aligned_effective_sequence"] % 8, 0)

    def test_bound_inputs_exist_and_are_hashed(self) -> None:
        for source_id in SOURCES:
            self.assertTrue(_data_path(source_id).exists(), source_id)
            self.assertTrue(_profile_path(source_id).exists(), source_id)
        for job in self.jobs:
            self.assertEqual(len(job["data_sha256"]), 64)
            self.assertEqual(len(job["dataset_profile_sha256"]), 64)

    def test_sources_have_a_dense_tail(self) -> None:
        # p95 close to max is what makes a source pushable to OOM under the
        # min(cutoff, profile_max) policy; a sparse tail cannot raise pressure.
        for source_id, spec in SOURCES.items():
            self.assertGreaterEqual(
                spec["p95"] / spec["profile_max"], 0.9, source_id
            )

    def test_design_is_pre_gpu_and_self_hashing(self) -> None:
        design = build_design(self.jobs)
        self.assertFalse(design["gpu_training_started"])
        self.assertFalse(design["execution_authorized"])
        self.assertFalse(design["publication_allowed"])
        self.assertTrue(design["oom_is_expected_output"])
        self.assertTrue(
            design["interpretation"]["no_safety_threshold_is_relaxed_by_this_campaign"]
        )
        self.assertTrue(
            design["acceptance_additions_over_batch_one"][
                "unsafe_evidence_must_span_campaigns"
            ]
        )
        from common import sha256_json

        unsigned = dict(design)
        report_sha = unsigned.pop("report_sha256")
        self.assertEqual(report_sha, sha256_json(unsigned))


if __name__ == "__main__":
    unittest.main()
