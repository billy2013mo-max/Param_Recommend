from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import rebuild_queue  # noqa: E402


class RebuildQueueTests(unittest.TestCase):
    def test_success_is_skipped_and_active_missing_job_is_not_duplicated(self) -> None:
        jobs = [{"job_id": "done"}, {"job_id": "live"}, {"job_id": "retry"}]
        with tempfile.TemporaryDirectory() as directory:
            results = Path(directory)
            status = results / "done" / "status.json"
            status.parent.mkdir(parents=True)
            status.write_text(json.dumps({"classification": "success"}), encoding="utf-8")
            pending, summary = rebuild_queue.partition_jobs(
                jobs,
                active={"live"},
                accepted={"success"},
                results_dir=results,
            )
        self.assertEqual([row["job_id"] for row in pending], ["retry"])
        self.assertEqual(summary["active_job_ids"], ["live"])
        self.assertEqual(
            summary["outcomes"],
            {"accepted:success": 1, "active": 1, "queued:missing": 1},
        )


if __name__ == "__main__":
    unittest.main()
