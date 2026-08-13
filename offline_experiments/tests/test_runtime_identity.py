from __future__ import annotations

import sys
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from run_job import live_runtime_identity  # noqa: E402


class RuntimeIdentityTests(unittest.TestCase):
    def test_job_identity_binds_deepspeed_version_and_patched_source(self) -> None:
        identity = live_runtime_identity()

        self.assertEqual(identity["python_executable"], sys.executable)
        self.assertIn("deepspeed", identity["packages"])
        self.assertIn("llamafactory", identity["packages"])
        self.assertIsNotNone(
            identity["framework_source_sha256"]["deepspeed_zero_partition_parameters"]
        )
        self.assertIsNotNone(identity["launcher_patch_sha256"])


if __name__ == "__main__":
    unittest.main()
