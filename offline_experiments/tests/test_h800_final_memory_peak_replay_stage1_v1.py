from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from common import read_json, read_jsonl, sha256_file, sha256_json
from prepare_h800_final_memory_peak_replay_stage1_v1 import (
    DATA_SEED,
    DEFAULT_DATA_BUNDLE,
    DEFAULT_DESIGN,
    DEFAULT_PREDICTIONS,
    DEFAULT_QUEUE,
    EXPECTED_JOBS,
    EXPECTED_PAIRS,
    GPU_IDS,
    _inverse_permuted_replay_rows,
    _sampler_batches,
)


class FinalMemoryPeakReplayStage1Test(unittest.TestCase):
    def test_inverse_permutation_replays_exact_batch_on_every_rank(self) -> None:
        source = [
            {"system": "", "prompt": f"p{index}", "response": "r"} for index in range(8)
        ]
        peak_indices = [2, 7]
        replay = _inverse_permuted_replay_rows(source, peak_indices, replay_rows=32)
        emitted_source_indices = [
            int(replay[index]["sample_id"].rsplit(":", 1)[-1]) - 1
            for rank_batches in _sampler_batches(
                len(replay), mbs=2, world_size=2, data_seed=DATA_SEED
            )
            for batch in rank_batches
            for index in batch
        ]
        self.assertEqual({2, 7}, set(emitted_source_indices))
        for rank_batches in _sampler_batches(
            len(replay), mbs=2, world_size=2, data_seed=DATA_SEED
        ):
            for batch in rank_batches:
                actual = [
                    int(replay[index]["sample_id"].rsplit(":", 1)[-1]) - 1
                    for index in batch
                ]
                self.assertEqual(peak_indices, actual)

    def test_frozen_queue_and_bindings_are_exact(self) -> None:
        jobs = read_jsonl(DEFAULT_QUEUE)
        design = read_json(DEFAULT_DESIGN)
        predictions = read_json(DEFAULT_PREDICTIONS)
        data = read_json(DEFAULT_DATA_BUNDLE)
        self.assertEqual(EXPECTED_JOBS, len(jobs))
        self.assertEqual(EXPECTED_JOBS, len({row["job_id"] for row in jobs}))
        pairs = {row["pair_id"] for row in jobs}
        self.assertEqual(EXPECTED_PAIRS, len(pairs))
        for pair_id in pairs:
            modes = {row["replay_mode"] for row in jobs if row["pair_id"] == pair_id}
            self.assertEqual({"full_coverage", "peak_replay"}, modes)
        self.assertEqual(list(GPU_IDS), design["authorized_gpu_ids"])
        self.assertEqual(0, data["final_acceptance_sources_read"])
        self.assertEqual(0, predictions["outcomes_observed"])
        self.assertEqual(sha256_json(jobs), design["ordered_job_payload_sha256"])
        self.assertEqual(
            sha256_file(DEFAULT_QUEUE), design["bindings"]["queue"]["sha256"]
        )
        self.assertTrue(
            all(
                profile["replay_sampler_verification"][
                    "all_rank_microbatches_exact_peak_batch"
                ]
                for profile in data["profiles"]
            )
        )


if __name__ == "__main__":
    unittest.main()
