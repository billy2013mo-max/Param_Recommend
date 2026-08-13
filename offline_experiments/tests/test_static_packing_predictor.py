from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from static_packing_predictor import (  # noqa: E402
    PROJECT_ROOT,
    build_calibration_replay,
    build_decision,
    greedy_knapsack,
    load_policy,
)


POLICY_PATH = ROOT / "artifacts" / "static_packing_policy_v1.json"


class StaticPackingPredictorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = load_policy(POLICY_PATH)

    def _profile(self, directory: Path) -> Path:
        path = directory / "profile.jsonl"
        lengths = [500, 1000, 1500, 2000] * 250
        path.write_text(
            "".join(json.dumps({"total_tokens": length}) + "\n" for length in lengths),
            encoding="utf-8",
        )
        return path

    def _request(self, profile: Path, **overrides: object) -> dict:
        request = {
            "request_id": "case",
            "gpu_family": "H800",
            "modality": "text",
            "stage": "sft",
            "dtype": "bf16",
            "model_id": "qwen3_14b",
            "train_type": "lora",
            "profile_path": str(profile),
            "cutoff_len": 4096,
            "no_packing_mbs": 2,
            "gpu_count": 1,
            "target_gbs": 64,
            "preprocessing_num_workers": 8,
        }
        request.update(overrides)
        return request

    def test_greedy_knapsack_matches_largest_fitting_item_rule(self) -> None:
        self.assertEqual(
            greedy_knapsack([6, 5, 4, 3, 2], 10),
            [[6, 4], [5, 3, 2]],
        )

    def test_favorable_released_text_case_is_on(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            profile = self._profile(Path(temporary))
            decision = build_decision(
                self._request(profile),
                policy=self.policy,
                policy_path=POLICY_PATH,
                request_base=Path(temporary),
            )
        self.assertEqual(decision["recommendation"]["decision"], "on")
        self.assertTrue(decision["recommendation"]["packing"])
        self.assertTrue(
            decision["operational_contract"]["pre_recommendation_gpu_execution"]
            is False
        )

    def test_favorable_full_sft_case_is_held_off(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            profile = self._profile(Path(temporary))
            decision = build_decision(
                self._request(profile, train_type="full"),
                policy=self.policy,
                policy_path=POLICY_PATH,
                request_base=Path(temporary),
            )
        self.assertEqual(decision["recommendation"]["decision"], "hold_off")
        self.assertFalse(decision["recommendation"]["packing"])
        self.assertTrue(decision["recommendation"]["shadow_candidate_packing"])

    def test_short_sequence_large_mbs_fails_occupancy_gate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            profile = self._profile(Path(temporary))
            decision = build_decision(
                self._request(
                    profile,
                    cutoff_len=512,
                    no_packing_mbs=16,
                ),
                policy=self.policy,
                policy_path=POLICY_PATH,
                request_base=Path(temporary),
            )
        self.assertEqual(decision["recommendation"]["decision"], "off")
        self.assertIn(
            "cutoff_tokens_per_no_packing_mbs",
            decision["recommendation"]["reason_codes"],
        )

    def test_multimodal_case_is_structural_off_without_profile(self) -> None:
        request = self._request(Path("/not/read"), modality="vision_language")
        request.pop("profile_path")
        decision = build_decision(
            request,
            policy=self.policy,
            policy_path=POLICY_PATH,
            request_base=PROJECT_ROOT,
        )
        self.assertEqual(decision["recommendation"]["decision"], "off")
        self.assertIn(
            "multimodal_packing_not_supported",
            decision["recommendation"]["reason_codes"],
        )

    def test_policy_rejects_online_gpu_execution(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "policy.json"
            tampered = copy.deepcopy(self.policy)
            tampered["online_gpu_execution_allowed"] = True
            path.write_text(json.dumps(tampered), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "prohibit online GPU"):
                load_policy(path)

    def test_historical_calibration_replay_matches_all_ten_points(self) -> None:
        report = build_calibration_replay(
            policy=self.policy,
            policy_path=POLICY_PATH,
            stage_decisions_path=ROOT / "artifacts" / "stage_decisions.json",
            packing_requests_path=ROOT / "matrix" / "packing_pair_requests.jsonl",
            dataset_profile_dir=ROOT / "artifacts" / "dataset_profiles",
        )
        self.assertEqual(report["metrics"]["points"], 10)
        self.assertEqual(report["metrics"]["correct"], 10)
        self.assertEqual(
            report["evaluation_kind"], "calibration_resubstitution_not_holdout"
        )


if __name__ == "__main__":
    unittest.main()
