"""Invariant tests for the workload contract.

These are the properties the plan requires to hold before any GPU calibration:
attention-pair accounting, token accounting, GA/GBS derivation and padding.  They
are stated as invariants rather than golden values so they keep holding as
coefficients change -- a golden number would only pin today's fit, while an
invariant pins the contract.

Each test says what would break in the model if the invariant failed, because
that is the reason the invariant is worth enforcing.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import candidate_generator as cg  # noqa: E402
from common import ARTIFACT_DIR  # noqa: E402
from structured_throughput_modeling import StaticDatasetProfiles  # noqa: E402

PROFILE_DIR = ARTIFACT_DIR / "dataset_profiles"
CAPACITY = 150_142_189_568
QWEN3_8B = 8_190_735_360

TEXT_DATASETS = ("short_512", "multiturn_4096", "longcontext_32768")
CUTOFF_BY_DATASET = {
    "short_512": 512,
    "multiturn_4096": 4096,
    "longcontext_32768": 32768,
}


def _profiles() -> StaticDatasetProfiles:
    if not PROFILE_DIR.is_dir():
        raise unittest.SkipTest("dataset profiles unavailable")
    return StaticDatasetProfiles(PROFILE_DIR)


class TestTokenAccountingInvariants(unittest.TestCase):
    """Effective, computed and padding tokens must stay mutually consistent."""

    def setUp(self) -> None:
        self.profiles = _profiles()

    def test_effective_tokens_never_exceed_computed_tokens(self) -> None:
        # Computed tokens include padding; effective ones do not.  If this
        # inverted, padding would be counted as useful work and the throughput
        # head would reward padding.
        for dataset_id in TEXT_DATASETS:
            for mbs in (1, 2, 4):
                profile = self.profiles.profile(
                    dataset_id,
                    cutoff_len=CUTOFF_BY_DATASET[dataset_id],
                    physical_mbs=mbs,
                    packing=False,
                )
                work = profile["work_per_physical_sequence"]
                self.assertLessEqual(
                    work["effective_tokens"],
                    work["computed_tokens"] + 1e-6,
                    f"{dataset_id} mbs={mbs}",
                )

    def test_computed_tokens_never_exceed_the_cutoff(self) -> None:
        # A physical sequence cannot compute more tokens than its capacity.
        for dataset_id in TEXT_DATASETS:
            cutoff = CUTOFF_BY_DATASET[dataset_id]
            profile = self.profiles.profile(
                dataset_id, cutoff_len=cutoff, physical_mbs=1, packing=False
            )
            self.assertLessEqual(
                profile["work_per_physical_sequence"]["computed_tokens"],
                cutoff + 1e-6,
            )

    def test_padding_utilization_is_a_ratio(self) -> None:
        for dataset_id in TEXT_DATASETS:
            for packing in (False, True):
                profile = self.profiles.profile(
                    dataset_id,
                    cutoff_len=CUTOFF_BY_DATASET[dataset_id],
                    physical_mbs=1,
                    packing=packing,
                )
                utilization = profile["padding_utilization"]
                self.assertGreater(utilization, 0.0)
                self.assertLessEqual(utilization, 1.0 + 1e-9)

    def test_larger_micro_batch_cannot_improve_padding_utilization(self) -> None:
        # Dynamic padding pads to the longest sample in the batch, so adding
        # samples can only keep or worsen utilization.  If this were violated,
        # the model would predict that bigger batches waste less compute.
        for dataset_id in TEXT_DATASETS:
            cutoff = CUTOFF_BY_DATASET[dataset_id]
            previous = None
            for mbs in (1, 2, 4):
                profile = self.profiles.profile(
                    dataset_id, cutoff_len=cutoff, physical_mbs=mbs, packing=False
                )
                current = profile["padding_utilization"]
                if previous is not None:
                    self.assertLessEqual(
                        current, previous + 1e-9, f"{dataset_id} mbs={mbs}"
                    )
                previous = current


class TestAttentionPairInvariants(unittest.TestCase):
    """Attention work must be quadratic-ish and never substituted by cutoff²."""

    def setUp(self) -> None:
        self.profiles = _profiles()

    def test_attention_pairs_are_positive(self) -> None:
        for dataset_id in TEXT_DATASETS:
            profile = self.profiles.profile(
                dataset_id,
                cutoff_len=CUTOFF_BY_DATASET[dataset_id],
                physical_mbs=1,
                packing=False,
            )
            self.assertGreater(
                profile["work_per_physical_sequence"][
                    "computed_attention_token_pairs"
                ],
                0.0,
            )

    def test_unpacked_pairs_do_not_use_the_flat_cutoff_square(self) -> None:
        # The plan forbids replacing per-sequence attention work with cutoff**2.
        # Real data is shorter than the cutoff, so the true pair count must be
        # strictly below that ceiling.
        for dataset_id in TEXT_DATASETS:
            cutoff = CUTOFF_BY_DATASET[dataset_id]
            profile = self.profiles.profile(
                dataset_id, cutoff_len=cutoff, physical_mbs=1, packing=False
            )
            pairs = profile["work_per_physical_sequence"][
                "computed_attention_token_pairs"
            ]
            self.assertLess(pairs, float(cutoff) ** 2)

    def test_packed_pairs_are_summed_per_subsequence_not_over_the_pack(self) -> None:
        # Cross-sample attention is isolated inside a pack, so pair count must be
        # far below (pack length)**2.  Treating a pack as one long sequence would
        # inflate attention work by roughly the number of samples per pack.
        for dataset_id in ("multiturn_4096",):
            cutoff = CUTOFF_BY_DATASET[dataset_id]
            packed = self.profiles.profile(
                dataset_id, cutoff_len=cutoff, physical_mbs=1, packing=True
            )
            pairs = packed["work_per_physical_sequence"][
                "computed_attention_token_pairs"
            ]
            self.assertLess(pairs, float(cutoff) ** 2)
            samples = packed["mean_samples_per_pack"]
            self.assertGreater(samples, 1.0)

    def test_attention_pairs_grow_with_cutoff(self) -> None:
        # Same dataset family, longer window: more tokens survive truncation, so
        # attention work must not shrink.
        short = self.profiles.profile(
            "longcontext_32768", cutoff_len=8192, physical_mbs=1, packing=False
        )
        long = self.profiles.profile(
            "longcontext_32768", cutoff_len=32768, physical_mbs=1, packing=False
        )
        self.assertLessEqual(
            short["work_per_physical_sequence"]["computed_attention_token_pairs"],
            long["work_per_physical_sequence"]["computed_attention_token_pairs"]
            + 1e-6,
        )


class TestPackingInvariants(unittest.TestCase):
    """Packing changes the workload shape; it must not change its semantics."""

    def setUp(self) -> None:
        self.profiles = _profiles()

    def test_packing_raises_logical_samples_per_physical_sequence(self) -> None:
        # This is the whole mechanism: one physical sequence carries several
        # logical samples.  Unpacked must stay at exactly one.
        unpacked = self.profiles.profile(
            "multiturn_4096", cutoff_len=4096, physical_mbs=1, packing=False
        )
        packed = self.profiles.profile(
            "multiturn_4096", cutoff_len=4096, physical_mbs=1, packing=True
        )
        self.assertAlmostEqual(
            unpacked["work_per_physical_sequence"]["logical_samples"], 1.0
        )
        self.assertGreater(
            packed["work_per_physical_sequence"]["logical_samples"], 1.0
        )

    def test_packing_beats_multi_sample_dynamic_padding_not_mbs_one(self) -> None:
        # Careful: at mbs=1 dynamic padding pads each sequence to its own length,
        # so unpacked utilization is exactly 1.0 and packing (which pads to the
        # cutoff) is marginally *below* it.  Packing's padding benefit therefore
        # only exists against mbs>1 baselines.  This is precisely why the frozen
        # policy separates the mbs=1 and mbs>1 benefit thresholds instead of
        # using one global figure -- asserting a blanket improvement here would
        # contradict the gate the policy actually implements.
        for dataset_id in ("multiturn_4096", "longcontext_32768"):
            cutoff = CUTOFF_BY_DATASET[dataset_id]
            at_one = self.profiles.profile(
                dataset_id, cutoff_len=cutoff, physical_mbs=1, packing=False
            )
            packed = self.profiles.profile(
                dataset_id, cutoff_len=cutoff, physical_mbs=1, packing=True
            )
            self.assertAlmostEqual(at_one["padding_utilization"], 1.0, places=6)
            for mbs in (2, 4):
                multi = self.profiles.profile(
                    dataset_id, cutoff_len=cutoff, physical_mbs=mbs, packing=False
                )
                self.assertGreater(
                    packed["padding_utilization"],
                    multi["padding_utilization"],
                    f"{dataset_id} vs mbs={mbs}",
                )

    def test_packing_reduces_physical_sequences_per_logical_sample(self) -> None:
        # This is packing's real, mbs-independent benefit: the same logical
        # samples ride in fewer physical sequences, so there are fewer
        # forward/backward passes.
        for dataset_id in ("multiturn_4096", "longcontext_32768"):
            cutoff = CUTOFF_BY_DATASET[dataset_id]
            unpacked = self.profiles.profile(
                dataset_id, cutoff_len=cutoff, physical_mbs=1, packing=False
            )
            packed = self.profiles.profile(
                dataset_id, cutoff_len=cutoff, physical_mbs=1, packing=True
            )
            self.assertGreater(
                packed["work_per_physical_sequence"]["logical_samples"],
                unpacked["work_per_physical_sequence"]["logical_samples"],
                dataset_id,
            )

    def test_packed_fill_ratio_is_reported_only_when_packing(self) -> None:
        unpacked = self.profiles.profile(
            "multiturn_4096", cutoff_len=4096, physical_mbs=1, packing=False
        )
        packed = self.profiles.profile(
            "multiturn_4096", cutoff_len=4096, physical_mbs=1, packing=True
        )
        self.assertIsNone(unpacked["packing_fill_ratio"])
        self.assertIsNotNone(packed["packing_fill_ratio"])

    def test_packing_does_not_inflate_effective_tokens_per_sample(self) -> None:
        # Packing rearranges samples; it must not create useful tokens.  Compare
        # per logical sample so the two shapes are commensurable.
        unpacked = self.profiles.profile(
            "multiturn_4096", cutoff_len=4096, physical_mbs=1, packing=False
        )
        packed = self.profiles.profile(
            "multiturn_4096", cutoff_len=4096, physical_mbs=1, packing=True
        )
        unpacked_work = unpacked["work_per_physical_sequence"]
        packed_work = packed["work_per_physical_sequence"]
        per_sample_unpacked = (
            unpacked_work["effective_tokens"] / unpacked_work["logical_samples"]
        )
        per_sample_packed = (
            packed_work["effective_tokens"] / packed_work["logical_samples"]
        )
        self.assertAlmostEqual(
            per_sample_unpacked, per_sample_packed, delta=0.02 * per_sample_unpacked
        )


class TestGbsAndGaInvariants(unittest.TestCase):
    """GBS is a user-visible training semantic and must be reproduced exactly."""

    def _scenario(self, **overrides):
        scenario = {
            "model_id": "qwen3_8b",
            "training_mode": "lora",
            "dataset_id": "short_512",
            "target_gbs": 64,
            "cutoff_len": 512,
            "actual_parameters": QWEN3_8B,
            "packing": False,
        }
        scenario.update(overrides)
        return scenario

    def test_ga_times_mbs_times_gpus_equals_target_gbs(self) -> None:
        for target_gbs in (32, 64, 128):
            out = cg.generate_candidates(
                self._scenario(target_gbs=target_gbs), capacity_bytes=CAPACITY
            )
            self.assertTrue(out["candidates"])
            for candidate in out["candidates"]:
                self.assertEqual(
                    candidate["gpu_count"]
                    * candidate["physical_mbs"]
                    * candidate["gradient_accumulation_steps"],
                    target_gbs,
                )

    def test_ga_is_always_at_least_one(self) -> None:
        # A GA below one would mean a fractional optimizer step.
        out = cg.generate_candidates(self._scenario(), capacity_bytes=CAPACITY)
        for candidate in out["candidates"]:
            self.assertGreaterEqual(candidate["gradient_accumulation_steps"], 1)

    def test_indivisible_shapes_are_rejected_not_rounded(self) -> None:
        # Rounding GA would silently change the user's GBS.
        out = cg.generate_candidates(
            self._scenario(target_gbs=32), capacity_bytes=CAPACITY
        )
        reasons = {row["reason"] for row in out["statically_rejected"]}
        self.assertIn("gbs_not_divisible_by_gpu_count_times_mbs", reasons)

    def test_packed_candidates_do_not_carry_an_unpacked_ga(self) -> None:
        # Packed GA depends on pack occupancy, which the frozen packing policy
        # owns; emitting an unpacked GA here would assert the wrong contract.
        out = cg.generate_candidates(
            self._scenario(packing=True), capacity_bytes=CAPACITY
        )
        self.assertTrue(out["candidates"])
        for candidate in out["candidates"]:
            self.assertNotIn("gradient_accumulation_steps", candidate)
            self.assertEqual(candidate["physical_mbs"], 1)

    def test_scenario_fields_are_invariant_across_generated_candidates(self) -> None:
        out = cg.generate_candidates(self._scenario(), capacity_bytes=CAPACITY)
        for field in ("target_gbs", "cutoff_len", "packing", "model_id"):
            values = {candidate[field] for candidate in out["candidates"]}
            self.assertEqual(len(values), 1, field)


class TestProfileDeterminism(unittest.TestCase):
    def test_same_request_yields_identical_work(self) -> None:
        # The recommendation path must be reproducible; a non-deterministic
        # profile would make a frozen prediction unverifiable.
        profiles = _profiles()
        first = profiles.profile(
            "multiturn_4096", cutoff_len=4096, physical_mbs=2, packing=True
        )
        second = StaticDatasetProfiles(PROFILE_DIR).profile(
            "multiturn_4096", cutoff_len=4096, physical_mbs=2, packing=True
        )
        self.assertEqual(
            first["work_per_physical_sequence"], second["work_per_physical_sequence"]
        )

    def test_profile_declares_it_is_pre_run_static(self) -> None:
        profiles = _profiles()
        profile = profiles.profile(
            "short_512", cutoff_len=512, physical_mbs=1, packing=False
        )
        self.assertTrue(profile["source_is_pre_run_static_profile"])


if __name__ == "__main__":
    unittest.main()
