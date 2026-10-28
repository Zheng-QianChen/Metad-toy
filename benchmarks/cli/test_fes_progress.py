#!/usr/bin/env python3
"""Batch boundaries, kernel equivalence, progress output and cancellation."""
import csv
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
from metadtoy_cli.colvar import ColvarReader
from metadtoy_cli.hills import evaluate_bias
from metadtoy_cli.progress import ProgressReporter, TerminalProgress
from metadtoy_cli.reweight import WTOffset, create_reweighted_fes, wt_offset
from test_cli import native


class FesProgressTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)

    def tearDown(self):
        self.directory.cleanup()

    def samples(self, name="COLVAR"):
        path = self.root / name
        path.write_text("# step value log_weight\n" +
                        "".join("%d %.17g %.17g\n" % (2**53+index, (index%11)/10, np.sin(index)*20)
                                for index in range(101)))
        return path

    def test_batch_parser_int64_and_late_validation(self):
        path = self.samples()
        reader = ColvarReader(path)
        rows = list(reader.records())
        batches = list(reader.batches(7))
        self.assertEqual(sum(len(indices) for indices, columns in batches), len(rows))
        steps = np.concatenate([columns["step"] for indices, columns in batches])
        np.testing.assert_array_equal(steps, [row["step"] for index, row in rows])
        self.assertEqual(steps.dtype, np.dtype("int64"))
        with path.open("a") as stream:
            stream.write("9007199254741093 nan 0\n")
        with self.assertRaises(ValueError):
            list(ColvarReader(path).batches(7))

    def test_grid_factors_match_generic_evaluator(self):
        for dimension in (1, 2, 3):
            for periodic in (False, True):
                periods = tuple(2. if periodic and axis%2 == 0 else 0. for axis in range(dimension))
                history = native(self.root / "HILLS", sigma=(.17,)*dimension, period=periods,
                                 rows=[[3]+[-.25]*dimension+[.1], [8]+[.4]*dimension+[.2],
                                       [8]+[-.8]*dimension+[.01]])
                correction = WTOffset(history, [(axis+1, -1, 1) for axis in range(dimension)], bins=(15,))
                points = np.stack(np.meshgrid(*correction.coordinates, indexing="ij"), axis=-1).reshape(-1, dimension)
                correction.at(8, [0.]*dimension)
                reference = evaluate_bias(history, points, hill_batch=1)
                np.testing.assert_allclose(correction.values.ravel(), reference, atol=2e-16, rtol=2e-13)
                self.assertAlmostEqual(correction.offset, wt_offset(reference, .025, 9), places=14)

    def test_offset_batches_deposition_boundaries_and_large_steps(self):
        first = 2**53+100
        history = native(self.root / "HILLS", rows=[[first+2, .3, .1], [first+5, -.2, .2],
                                                   [first+5, .4, .03], [first+9, .5, .1]])
        steps = np.arange(first, first+12, dtype=np.int64)
        points = np.zeros((len(steps), 1))
        scalar = WTOffset(history, [(1, -1, 1)], bins=(64,))
        expected = [scalar.at(int(step), point) for step, point in zip(steps, points)]
        for size in (1, 4, 12):
            correction = WTOffset(history, [(1, -1, 1)], bins=(64,))
            result = np.concatenate([correction.batch_at(steps[start:start+size], points[start:start+size])
                                     for start in range(0, len(steps), size)])
            np.testing.assert_array_equal(result, expected)
            self.assertEqual(correction.hills, 4)
            self.assertEqual(correction.updates, 3)

    def test_pipeline_batch_equivalence_and_completion_after_publish(self):
        path = self.samples()
        expected = None
        for batch_size in (1, 7, 8192):
            events = []
            prefix = self.root / ("fes%d" % batch_size)
            def observe(event):
                if event["status"] == "done":
                    self.assertTrue(all(Path(str(prefix)+suffix).is_file()
                                        for suffix in (".csv", ".png", ".json", ".weights.csv")))
                events.append(event)
            outputs = create_reweighted_fes(path, prefix, thermal_energy=1, bins=(8,), bounds=[(1, .1, .9)],
                                            sample_batch=batch_size, progress=observe)
            with Path(outputs[0]).open() as stream:
                rows = list(csv.DictReader(stream))
            numbers = np.array([[float(row[column]) if row[column] else np.nan for column in ("fes", "probability_density", "count", "weight_ess")]
                                for row in rows])
            metadata = json.loads(Path(outputs[2]).read_text())
            if expected is None:
                expected = numbers, metadata["weight_ess"], Path(outputs[3]).read_bytes()
            np.testing.assert_allclose(numbers, expected[0], atol=1e-13, rtol=2e-13)
            self.assertAlmostEqual(metadata["weight_ess"], expected[1], places=11)
            self.assertEqual(Path(outputs[3]).read_bytes(), expected[2])
            phases = [event["phase"] for event in events]
            self.assertEqual(list(dict.fromkeys(phases)),
                             ["initialize", "hash_inputs", "scan", "weights", "export", "plot", "validate", "publish", "done"])

    def test_cancel_preserves_outputs_and_has_no_done(self):
        path = self.samples()
        prefix = self.root / "keep"
        old = Path(str(prefix)+".csv")
        old.write_text("previous result")
        events = []
        def cancel(event):
            events.append(event)
            if event["phase"] == "weights" and event["completed"] > 0:
                raise KeyboardInterrupt()
        with self.assertRaises(KeyboardInterrupt):
            create_reweighted_fes(path, prefix, thermal_energy=1, bins=(4,), sample_batch=7,
                                  progress=cancel, force=True)
        self.assertEqual(old.read_text(), "previous result")
        self.assertFalse(Path(str(prefix)+".json").exists())
        self.assertFalse(any(event["status"] == "done" for event in events))

    def test_progress_throttles_and_nonterminal_default_is_silent(self):
        now = [0.]
        events = []
        reporter = ProgressReporter(events.append, clock=lambda: now[0])
        reporter.begin("weights", 100, "frames")
        reporter.update(1)
        self.assertEqual(len(events), 1)
        now[0] = 1.
        reporter.update(25)
        self.assertEqual(events[-1]["eta_seconds"], 3.)
        stream = io.StringIO()
        renderer = TerminalProgress("auto", stream)
        renderer(events[-1])
        self.assertEqual(stream.getvalue(), "")
        renderer = TerminalProgress("always", stream)
        renderer(events[-1])
        self.assertIn("25.0%", stream.getvalue())

    def test_cli_progress_stderr_and_json_stdout(self):
        path = self.samples()
        base = [sys.executable, str(ROOT / "tools/metadtoy.py"), "fes", "reconstruct", str(path),
                "--kbt", "1", "--out", str(self.root / "cli"), "--force"]
        silent = subprocess.run(base, capture_output=True, text=True)
        self.assertEqual(silent.returncode, 0, silent.stderr)
        self.assertEqual(silent.stderr, "")
        self.assertEqual(len(json.loads(silent.stdout)["outputs"]), 4)
        visible = subprocess.run(base+["--progress", "always"], capture_output=True, text=True)
        self.assertEqual(visible.returncode, 0, visible.stderr)
        self.assertIn("100.0%", visible.stderr)
        self.assertIn("完成", visible.stderr)
        self.assertEqual(len(json.loads(visible.stdout)["outputs"]), 4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
