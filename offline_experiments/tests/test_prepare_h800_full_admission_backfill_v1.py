from __future__ import annotations

from collections import Counter
from pathlib import Path
import sys
import unittest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from prepare_h800_full_admission_backfill_v1 import (  # noqa: E402
    BAND_SOURCES,
    MECHANISMS,
    SOURCES,
    SUPPORTED_MECHANISM,
    TARGET_GBS,
    _align8,
    _data_path,
    _effective_sequence,
    _models,
    _profile_path,
    build_design,
    build_jobs,
)


class FullAdmissionBackfillDesignTests(unittest.TestCase):
    def setUp(self) -> None:
        self.jobs = build_jobs(_models())

    def test_covers_the_nine_failing_mechanisms_plus_the_supported_one(self) -> None:
        self.assertEqual(len(MECHANISMS), 9)
        covered = {job["mechanism_id"] for job in self.jobs}
        expected = {row["mechanism_id"] for row in MECHANISMS} | {
            SUPPORTED_MECHANISM["mechanism_id"]
        }
        self.assertEqual(covered, expected)

    def test_job_totals_match_the_published_plan(self) -> None:
        groups = Counter(job["experiment_group"] for job in self.jobs)
        self.assertEqual(groups["backfill"], 36)
        self.assertEqual(groups["negative_boundary"], 6)
        self.assertEqual(groups["prospective_supported"], 4)
        self.assertEqual(len(self.jobs), 46)

    def test_every_backfill_mechanism_adds_exactly_two_business_sources(self) -> None:
        for mechanism in MECHANISMS:
            sources = {
                job["split_unit_id"]
                for job in self.jobs
                if job["mechanism_id"] == mechanism["mechanism_id"]
                and job["experiment_group"] == "backfill"
            }
            self.assertEqual(len(sources), 2, mechanism["mechanism_id"])

    def test_all_sources_are_business_data_never_the_public_corpora(self) -> None:
        public = {
            "short_512",
            "multiturn_2048",
            "multiturn_4096",
            "longtail_8192",
            "longcontext_16384",
            "longcontext_32768",
            "historical_public_connected_component_01",
        }
        used = {job["split_unit_id"] for job in self.jobs}
        self.assertFalse(used & public)
        self.assertTrue(used <= set(SOURCES))

    def test_campaign_is_full_finetuning_and_unpacked(self) -> None:
        self.assertTrue(all(job["train_type"] == "full" for job in self.jobs))
        self.assertFalse(any(job["packing"] for job in self.jobs))
        self.assertFalse(any(job["offload"] for job in self.jobs))

    def test_jobs_use_the_concrete_scheduler_path(self) -> None:
        # kind="memory_boundary" routes into the scheduler's adaptive family
        # sweep, which requires an `mbs_candidates` list these fixed probes do
        # not have; it raised KeyError on the first launch attempt.
        for job in self.jobs:
            self.assertEqual(job["kind"], "throughput", job["job_id"])
            self.assertNotIn("mbs_candidates", job)
            self.assertEqual(job["evidence_role"], "memory_admission_boundary")
            self.assertIsInstance(job["mbs"], int)

    def test_every_job_satisfies_the_launcher_contract(self) -> None:
        # run_job.validate_job is the gate that actually runs at launch time.
        # Checking it here turns a mid-campaign FatalLaunchError into a unit
        # test failure -- the "zero0" vs "none" mismatch killed the scheduler
        # 16 jobs in, after the first single-card job was reached.
        from run_job import validate_job

        for job in self.jobs:
            with self.subTest(job=job["job_id"]):
                validate_job(job)

    def test_single_card_jobs_declare_no_deepspeed(self) -> None:
        single = [job for job in self.jobs if job["gpu_count"] == 1]
        self.assertTrue(single)
        for job in single:
            self.assertEqual(job["zero"], "none", job["job_id"])
            self.assertEqual(job["zero_stage"], 0, job["job_id"])
        for job in self.jobs:
            if job["gpu_count"] > 1:
                self.assertIn(job["zero"], {"zero2", "zero3"}, job["job_id"])

    def test_gbs_contract_holds_for_every_job(self) -> None:
        for job in self.jobs:
            self.assertEqual(
                job["gpu_count"] * job["mbs"] * job["gradient_accumulation_steps"],
                TARGET_GBS,
                job["job_id"],
            )

    def test_effective_sequence_follows_the_m1_length_policy(self) -> None:
        for job in self.jobs:
            expected = _align8(
                min(job["cutoff_len"], job["raw_profile_max"])
            )
            self.assertEqual(job["aligned_effective_sequence"], expected)
            self.assertEqual(job["aligned_effective_sequence"] % 8, 0)

    def test_probe_low_is_never_heavier_than_probe_high(self) -> None:
        for mechanism in MECHANISMS:
            low = mechanism["probe_low"]
            high = mechanism["probe_high"]
            self.assertLessEqual(
                low["mbs"] * low["cutoff_len"],
                high["mbs"] * high["cutoff_len"],
                mechanism["mechanism_id"],
            )

    def test_mechanisms_lacking_negative_sources_get_an_extra_probe(self) -> None:
        for mechanism in MECHANISMS:
            extra = [
                job
                for job in self.jobs
                if job["mechanism_id"] == mechanism["mechanism_id"]
                and job["experiment_group"] == "negative_boundary"
            ]
            if mechanism.get("negative_boundary_short"):
                self.assertEqual(len(extra), 2, mechanism["mechanism_id"])
                self.assertEqual(
                    {job["split_unit_id"] for job in extra},
                    set(BAND_SOURCES["long"]),
                )
            else:
                self.assertEqual(extra, [])

    def test_job_ids_are_unique_and_deterministic(self) -> None:
        self.assertEqual(
            len({job["job_id"] for job in self.jobs}), len(self.jobs)
        )
        self.assertEqual(
            [job["job_id"] for job in self.jobs],
            [job["job_id"] for job in build_jobs(_models())],
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

    def test_extra_negative_probe_is_heavier_than_the_regular_high_probe(
        self,
    ) -> None:
        for mechanism in MECHANISMS:
            if not mechanism.get("negative_boundary_short"):
                continue
            if str(mechanism["band"]) != "long":
                continue
            regular = [
                job
                for job in self.jobs
                if job["mechanism_id"] == mechanism["mechanism_id"]
                and job["probe_role"] == "probe_high"
            ]
            extra = [
                job
                for job in self.jobs
                if job["mechanism_id"] == mechanism["mechanism_id"]
                and job["experiment_group"] == "negative_boundary"
            ]
            self.assertTrue(regular and extra)
            self.assertGreater(
                max(job["mbs"] for job in extra),
                max(job["mbs"] for job in regular),
                mechanism["mechanism_id"],
            )

    def test_bound_inputs_exist_and_are_hashed(self) -> None:
        for source_id in SOURCES:
            self.assertTrue(_data_path(source_id).exists(), source_id)
            self.assertTrue(_profile_path(source_id).exists(), source_id)
        for job in self.jobs:
            self.assertEqual(len(job["data_sha256"]), 64)
            self.assertEqual(len(job["dataset_profile_sha256"]), 64)

    def test_design_is_pre_gpu_and_self_hashing(self) -> None:
        design = build_design(self.jobs)
        self.assertFalse(design["gpu_training_started"])
        self.assertFalse(design["execution_authorized"])
        self.assertFalse(design["publication_allowed"])
        self.assertTrue(
            design["interpretation"]["no_safety_threshold_is_relaxed_by_this_campaign"]
        )
        unsigned = dict(design)
        report_sha = unsigned.pop("report_sha256")
        from common import sha256_json

        self.assertEqual(report_sha, sha256_json(unsigned))

    def test_design_records_why_new_data_is_needed(self) -> None:
        design = build_design(self.jobs)
        rationale = design["evidence_rationale"]
        self.assertEqual(rationale["historical_independent_sources_recounted"], 3)
        self.assertEqual(
            rationale["historical_independent_sources_as_currently_hardcoded"], 1
        )
        self.assertEqual(len(rationale["historical_upstream_corpora"]), 3)


if __name__ == "__main__":
    unittest.main()
