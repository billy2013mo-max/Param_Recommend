from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import check_length_alignment as cla


class AdmissionGateTests(unittest.TestCase):
    """The gate that V3 shipped without.

    V3's hybrid rows ran business data averaging ~805 tokens against cutoff
    8192, so "did not OOM" said nothing about whether the model's refusals at
    cutoff were right.  The gate refuses such a queue on the admission track.
    """

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def _exact_profile(self, name: str, *, tokens: int, rows: int = 4) -> Path:
        path = self.tmp / name
        with path.open("w", encoding="utf-8") as handle:
            for index in range(rows):
                handle.write(json.dumps({
                    "schema": "sft_h800_exact_text_profile/v1",
                    "sample_id": f"toy:{index}",
                    "total_tokens": tokens,
                    "target_tokens": tokens,
                    "label_tokens": 128,
                }) + "\n")
        return path

    def _summary_profile(self, name: str, **fields: object) -> Path:
        path = self.tmp / name
        path.write_text(json.dumps(fields), encoding="utf-8")
        return path

    def _job(self, profile: Path, *, cutoff: int, job_id: str = "j1") -> dict[str, object]:
        return {
            "job_id": job_id,
            "model_id": "qwen3p6_27b",
            "cutoff_len": cutoff,
            "dataset_profile_path": str(profile),
            "mbs": 1,
        }

    def test_exact_length_queue_passes_and_reports_binding(self) -> None:
        profile = self._exact_profile("exact_8192.jsonl", tokens=8192)
        binding = cla.require_exact_length_basis([self._job(profile, cutoff=8192)])
        self.assertTrue(binding["exact_length_basis"])
        self.assertEqual(binding["cutoff_lens"], [8192])
        self.assertEqual(binding["checked_jobs"], 1)

    def test_business_summary_profile_is_refused(self) -> None:
        """The exact V3 failure: max 11525 > cutoff 8192, but median is 751."""

        profile = self._summary_profile(
            "business.profile.json",
            max_tokens_per_sample=11_525,
            median_tokens_per_sample=751,
            mean_tokens_per_sample=1032,
        )
        with self.assertRaises(cla.LengthAlignmentError) as caught:
            cla.require_exact_length_basis([self._job(profile, cutoff=8192)])
        self.assertIn("only a maximum", str(caught.exception))

    def test_short_samples_are_refused_even_when_per_sample_exactness_is_known(self) -> None:
        profile = self._exact_profile("short.jsonl", tokens=805)
        violations = cla.audit_exact_length_basis([self._job(profile, cutoff=8192)])
        self.assertEqual(len(violations), 1)
        self.assertEqual(violations[0]["dataset_max_tokens"], 805)
        self.assertIn("need not reach the", violations[0]["reason"])

    def test_mixed_lengths_are_refused_although_the_longest_fills_cutoff(self) -> None:
        path = self.tmp / "mixed.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for tokens in (8192, 751, 900):
                handle.write(json.dumps({"total_tokens": tokens}) + "\n")
        violations = cla.audit_exact_length_basis([self._job(path, cutoff=8192)])
        self.assertEqual(len(violations), 1)
        self.assertEqual(violations[0]["dataset_min_tokens"], 751)

    def test_unaligned_cutoff_allows_one_padding_step(self) -> None:
        """cutoff 4095 is reachable: the collator pads 4088 up to 4096."""

        profile = self._exact_profile("pad.jsonl", tokens=4_088)
        binding = cla.require_exact_length_basis([self._job(profile, cutoff=4_095)])
        self.assertTrue(binding["exact_length_basis"])

    def test_missing_profile_is_refused(self) -> None:
        job = self._job(self.tmp / "absent.jsonl", cutoff=8192)
        violations = cla.audit_exact_length_basis([job])
        self.assertIn("unreadable", violations[0]["reason"])

    def test_nonpositive_cutoff_is_refused(self) -> None:
        profile = self._exact_profile("exact.jsonl", tokens=8192)
        violations = cla.audit_exact_length_basis([self._job(profile, cutoff=0)])
        self.assertIn("not positive", violations[0]["reason"])

    def test_every_offending_job_is_listed_not_just_the_first(self) -> None:
        short = self._exact_profile("s.jsonl", tokens=805)
        jobs = [
            self._job(short, cutoff=8192, job_id="a"),
            self._job(short, cutoff=4096, job_id="b"),
        ]
        with self.assertRaises(cla.LengthAlignmentError) as caught:
            cla.require_exact_length_basis(jobs)
        self.assertEqual(len(caught.exception.violations), 2)
        self.assertEqual(
            [v["job_id"] for v in caught.exception.violations], ["a", "b"]
        )


class RankingTrackTests(unittest.TestCase):
    """The ranking track keeps reporting rather than raising."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def test_long_tail_dataset_warns_without_raising(self) -> None:
        path = self.tmp / "business.profile.json"
        path.write_text(json.dumps({"max_tokens_per_sample": 11_525}), encoding="utf-8")
        warnings = cla.check_length_alignment([{
            "job_id": "j1",
            "model_id": "qwen3p6_27b",
            "cutoff_len": 8192,
            "dataset_profile_path": str(path),
            "mbs": 1,
        }])
        self.assertEqual(len(warnings), 1)
        self.assertEqual(warnings[0]["kind"], "long_tail_may_be_missed")

    def test_short_dataset_warns_that_center_overestimates(self) -> None:
        path = self.tmp / "vl.profile.json"
        path.write_text(json.dumps({"max_tokens_per_sample": 304}), encoding="utf-8")
        warnings = cla.check_length_alignment([{
            "job_id": "j1",
            "model_id": "qwen3_vl_4b",
            "cutoff_len": 8192,
            "dataset_profile_path": str(path),
            "mbs": 1,
        }])
        self.assertEqual(warnings[0]["kind"], "center_overestimates")


if __name__ == "__main__":
    unittest.main()
