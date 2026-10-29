#!/usr/bin/env python3
"""Numerical and command-line checks. Run on the designated validation host."""
import csv
import ctypes
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
from metadtoy_cli.hills import HillsReader, evaluate_bias
from metadtoy_cli.analysis import create_preview, export_csv


def native(path, mode="HILLS", sigma=(0.3,), period=None, rows=None,
           version=3, kernel="SMOOTH_GAUSSIAN", wt=1):
    dimension = len(sigma)
    period = period or (0.0,) * dimension
    count = dimension if mode == "PBMETAD" else 1
    fields = ["step"] + ["s%d" % (i + 1) for i in range(dimension)] + ["height%d" % (i + 1) for i in range(count)]
    header = ["# METAD_TOY_BIAS %d %s %d" % (version, mode, dimension),
              "# PARAMETERS 0.1 0.025 9 %d%s" % (wt, " " + kernel if version == 3 else ""),
              "# SIGMA " + " ".join(map(str, sigma)),
              "# PERIOD " + " ".join(map(str, period)), "# FIELDS " + " ".join(fields)]
    data = rows if rows is not None else [[10] + [0.0] * dimension + [0.1] * count]
    path.write_text("\n".join(header + [" ".join(map(str, row)) for row in data]) + "\n")
    return HillsReader(path)


class CliTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)

    def tearDown(self):
        self.directory.cleanup()

    def test_legacy_int64_csv_and_variable_width(self):
        step = 2**53 + 7
        path = self.root / "legacy"
        path.write_text("%d 0 2 0.5\n%d 1 3 0.2\n" % (step, step + 1))
        reader = HillsReader(path)
        values = evaluate_bias(reader, [[0], [1]], hill_batch=1, point_batch=1)
        np.testing.assert_allclose(values, [2 + 3 * math.exp(-12.5), 3 + 2 * math.exp(-2)])
        destination = self.root / "raw.csv"
        export_csv(reader, destination)
        with destination.open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(rows[0]["step"], str(step))
        self.assertEqual(float(rows[1]["sigma1"]), 0.2)
        self.assertEqual(reader.summary()["last_step"], step + 1)

    def test_native_versions_and_periodic_invariance(self):
        for version in (1, 2, 3):
            reader = native(self.root / ("v%d" % version), version=version, period=(2.0,))
            result = evaluate_bias(reader, [[0.17], [2.17], [-1.83]])
            np.testing.assert_allclose(result, result[0], rtol=2e-14)

    def test_hills_analytic_and_batch_equivalence(self):
        reader = native(self.root / "hills", sigma=(0.3, 0.7),
                        rows=[[1, 0.1, -0.2, 0.3], [2, -0.4, 0.5, 0.8]])
        points = np.array([[0.0, 0.0], [0.2, -0.1], [2.0, 1.0]])
        expected = sum(record.heights[0] * np.exp(-0.5 * (((points - record.centers) / record.sigma)**2).sum(axis=1))
                       for record in reader.records())
        np.testing.assert_allclose(evaluate_bias(reader, points, hill_batch=1, point_batch=1), expected, rtol=2e-14)
        np.testing.assert_allclose(evaluate_bias(reader, points), expected, rtol=2e-14)
        one = evaluate_bias(reader, points, through_step=1)
        first = next(reader.records())
        np.testing.assert_allclose(one, first.heights[0] * np.exp(-0.5 * (((points - first.centers) / first.sigma)**2).sum(axis=1)))

    def test_pb_combination_cutoff_and_normalization(self):
        reader = native(self.root / "pb", mode="PBMETAD", sigma=(0.3, 0.7),
                        kernel="PLUMED_2_10", rows=[[1, 0, 0, 0.2, 0.8]])
        points = np.array([[0, 0], [2, 3], [0.3, 0.7]])
        quadratic = (points / [0.3, 0.7])**2
        components = np.where(quadratic < 12.5, np.exp(-0.5 * quadratic), 0.0) * [0.2, 0.8]
        minimum = components.min(axis=1)
        expected = minimum - 0.025 * np.log(np.exp(-(components - minimum[:, None]) / 0.025).mean(axis=1))
        np.testing.assert_allclose(evaluate_bias(reader, points), expected, atol=1e-15)
        np.testing.assert_allclose(evaluate_bias(reader, points, pb_normalization="sum"),
                                   expected - 0.025 * math.log(2), atol=1e-15)

    def test_preview_axis_order_and_wt_fes(self):
        reader = native(self.root / "hills", sigma=(0.3, 0.7), rows=[[1, 0.2, -0.1, 0.8]])
        outputs = create_preview(reader, self.root / "preview", axes=(2, 1), bins=(13, 17),
                                 bounds=((1, -1, 1), (2, -2, 2)), quantity="fes", method="hills")
        data = np.genfromtxt(outputs[0], delimiter=",", names=True)
        points = np.column_stack([data["s1"], data["s2"]])
        expected = -9 / 8 * evaluate_bias(reader, points)
        expected -= expected.min()
        np.testing.assert_allclose(data["fes"], expected, atol=2e-15)
        self.assertEqual(Path(outputs[1]).read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
        metadata = json.loads(Path(outputs[2]).read_text())
        self.assertEqual(metadata["axes"], [2, 1])
        self.assertEqual(len(data), 13 * 17)

    def test_highdim_requires_explicit_slice_and_rejects_pb_fes(self):
        reader = native(self.root / "hills", mode="PBMETAD", sigma=(0.2, 0.4, 0.8))
        with self.assertRaisesRegex(ValueError, "unplotted"):
            create_preview(reader, self.root / "missing")
        with self.assertRaisesRegex(ValueError, "joint FES"):
            create_preview(reader, self.root / "wrong", fixed=((3, 0),), quantity="fes", method="hills")
        outputs = create_preview(reader, self.root / "slice", fixed=((3, 0.25),), bins=(9,))
        metadata = json.loads(Path(outputs[2]).read_text())
        self.assertEqual(metadata["interpretation"], "slice")
        self.assertEqual(metadata["fixed"], {"3": 0.25})

    def test_invalid_records_leave_no_published_output(self):
        for number, text in enumerate(("1 0 nan .2\n", "1 0 2 0\n", "2 0 1 .2\n1 0 1 .2\n", "1 0 1 .2")):
            path = self.root / ("bad%d" % number)
            path.write_text(text)
            destination = self.root / ("bad%d.csv" % number)
            with self.assertRaises(ValueError):
                export_csv(HillsReader(path), destination)
            self.assertFalse(destination.exists())
            self.assertFalse(destination.with_suffix(".json").exists())

    def test_output_protection_and_unknown_formats(self):
        reader = native(self.root / "hills")
        destination = self.root / "raw.csv"
        destination.write_text("keep")
        with self.assertRaises(ValueError):
            export_csv(reader, destination)
        self.assertEqual(destination.read_text(), "keep")
        export_csv(reader, destination, force=True)
        source_csv = self.root / "input.csv"
        source_csv.write_text("1 0 0.1 0.2\n")
        with self.assertRaisesRegex(ValueError, "distinct"):
            export_csv(HillsReader(source_csv), source_csv, force=True)
        plumed = self.root / "plumed"
        plumed.write_text("#! FIELDS time s sigma height biasf\n0 0 .2 .1 9\n")
        with self.assertRaisesRegex(ValueError, "PLUMED"):
            HillsReader(plumed)

    def test_empty_and_non_wt(self):
        empty = native(self.root / "empty", rows=[])
        self.assertEqual(empty.summary()["depositions"], 0)
        with self.assertRaisesRegex(ValueError, "No depositions"):
            create_preview(empty, self.root / "empty-preview")
        normal = native(self.root / "normal", wt=0)
        with self.assertRaisesRegex(ValueError, "well-tempered"):
            create_preview(normal, self.root / "bad-fes", quantity="fes", method="hills")

    def test_stdlib_commands_and_cli_error_status(self):
        reader = native(self.root / "hills")
        executable = str(ROOT / "tools" / "metadtoy.py")
        result = subprocess.run([sys.executable, "-S", executable, "hills", "inspect", str(reader.path)],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["depositions"], 1)
        result = subprocess.run([sys.executable, "-S", executable, "hills", "export", str(reader.path),
                                 "--csv", str(self.root / "stdlib.csv")], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        result = subprocess.run([sys.executable, executable, "hills", "inspect", str(self.root / "absent")],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertNotIn("Traceback", result.stderr)

    @unittest.skipUnless(os.environ.get("METADTOY_BIAS_REFERENCE"), "production C++ probe not supplied")
    def test_production_cpp_crosscheck(self):
        library = ctypes.CDLL(os.environ["METADTOY_BIAS_REFERENCE"])
        evaluate = library.metadtoy_bias_reference
        pointer = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
        evaluate.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_double,
                             pointer, pointer, pointer, pointer, ctypes.c_int, pointer, ctypes.c_int, pointer]
        evaluate.restype = ctypes.c_int
        rng = np.random.default_rng(73517)
        for dimension in (1, 2, 3, 8, 32, 128):
            for mode, kernel in (("HILLS", "SMOOTH_GAUSSIAN"),
                                 ("PBMETAD", "SMOOTH_GAUSSIAN"), ("PBMETAD", "PLUMED_2_10")):
                sigma = np.linspace(0.3, 0.7, dimension)
                period = np.array([2.0 if axis % 2 else 0.0 for axis in range(dimension)])
                centers = rng.normal(size=(5, dimension))
                heights = rng.uniform(0.01, 0.4, size=(5, dimension if mode == "PBMETAD" else 1))
                rows = [[hill, *centers[hill], *heights[hill]] for hill in range(5)]
                reader = native(self.root / "probe", mode=mode, sigma=tuple(sigma),
                                period=tuple(period), rows=rows, kernel=kernel)
                points = rng.normal(size=(11, dimension))
                expected = np.zeros(len(points))
                result = evaluate(dimension, mode == "PBMETAD", kernel == "PLUMED_2_10", 0.025,
                                  sigma, period, centers, heights, len(centers), points, len(points), expected)
                self.assertEqual(result, 0)
                np.testing.assert_allclose(evaluate_bias(reader, points, hill_batch=2, point_batch=3),
                                           expected, rtol=2e-12, atol=2e-14)


if __name__ == "__main__":
    unittest.main(verbosity=2)
