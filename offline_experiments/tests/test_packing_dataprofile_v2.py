import sys
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from prepare_packing_dataprofile_v2 import _candidate_cutoffs, _distribution, schema_document  # noqa: E402


class PackingDataProfileV2Test(unittest.TestCase):
    def test_schema_requires_cached_curve_and_online_no_read_contract(self) -> None:
        schema = schema_document()
        self.assertEqual(schema["$id"], "sft_packing_data_profile/v2")
        self.assertIn("packing_curve", schema["required"])
        contract = schema["properties"]["recommendation_contract"]["properties"]
        self.assertEqual(contract["raw_lengths_read"]["const"], False)
        self.assertEqual(contract["full_packer_run"]["const"], False)

    def test_candidate_cutoffs_include_nontruncating_event(self) -> None:
        values = _candidate_cutoffs(28_799)
        self.assertIn(29_184, values)
        self.assertIn(32_768, values)
        self.assertLessEqual(max(values), 40_960)

    def test_distribution_order(self) -> None:
        value = _distribution([1, 2, 2, 4, 10])
        self.assertLessEqual(value["mean"], value["p99"])
        self.assertLessEqual(value["p99"], value["maximum"])


if __name__ == "__main__":
    unittest.main()
