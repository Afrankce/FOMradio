import csv
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.monitor_matern_suite import Monitor, atomic, committed_history, process_matches, summarize


def fixture_history(directory, count):
    directory.mkdir(parents=True, exist_ok=True)
    rows = [dict(epoch=n, optimizer_step=5*n, train_loss=1/n, val_db_rmse=20-n/100,
                 val_db_mae=15-n/100, val_ssim=.7+n/10000) for n in range(1, count+1)]
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    atomic(directory / "metrics.csv", stream.getvalue())
    atomic(directory / "committed_epoch.json", dict(completed_epochs=count))
    atomic(directory / "config.json", dict(config_sha256="unit-test-fixture"))
    return rows


class MonitorTests(unittest.TestCase):
    def test_uncommitted_row_is_excluded(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture_history(root, 101)
            atomic(root / "committed_epoch.json", dict(completed_epochs=100))
            rows, digest = committed_history(root)
            self.assertEqual(len(rows), 100)
            self.assertEqual(len(digest), 64)
            self.assertEqual(summarize(rows, 100)["last"]["optimizer_step"], 500)
            with self.assertRaises(ValueError):
                summarize(rows, 200)

    def test_incomplete_history_is_not_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture_history(root, 99)
            atomic(root / "committed_epoch.json", dict(completed_epochs=100))
            with self.assertRaises(RuntimeError):
                committed_history(root)

    def test_milestone_backfill_and_idempotence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            jobs = {f"8x8/{v}": dict(phase="complete") for v in ("B2", "M1", "M2", "M3")}
            atomic(root / "suite_status.json", dict(status="complete", jobs=jobs))
            for name in jobs:
                fixture_history(root / name, 205)
            monitor = Monitor(root)
            with patch("scripts.monitor_matern_suite.subprocess.check_output", return_value="unit-test-gpu"), patch("builtins.print"):
                self.assertTrue(monitor.poll())
                paths = sorted((root / "monitoring/events").glob("*.json"))
                contents = {str(p): p.read_bytes() for p in paths}
                monitor.poll()
                self.assertEqual(contents, {str(p): p.read_bytes() for p in paths})
            for epoch in (100, 200):
                self.assertTrue((root / f"monitoring/comparisons/8x8_epoch_{epoch:04d}.md").exists())
            result = json.loads((root / "monitoring/8x8/M3/epoch_0100.json").read_text())
            self.assertEqual(result["last"]["val_db_rmse"], 19.)
            self.assertFalse((root / "monitoring/8x8/M3/epoch_0300.json").exists())

    def test_pid_identity_does_not_accept_unrelated_process(self):
        self.assertFalse(process_matches(os.getpid(), "/definitely-not-this-training-command"))
        self.assertFalse(process_matches(999999999, "/missing"))


if __name__ == "__main__":
    unittest.main()
