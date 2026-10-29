#!/usr/bin/env python3
"""Numerical checks for FES weighting, c(t), projections and CLI defaults."""
import csv
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
from metadtoy_cli.analysis import create_preview
from metadtoy_cli.colvar import ColvarReader
from metadtoy_cli.hills import HillsReader, evaluate_bias
from metadtoy_cli.reweight import WTOffset, create_reweighted_fes, wt_offset
from test_cli import native


class ReweightTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)

    def tearDown(self):
        self.directory.cleanup()

    def write(self, name, text):
        path = self.root / name
        path.write_text(text, encoding="utf-8")
        return path

    def colvar(self, fields, rows, name="COLVAR"):
        return self.write(name, "#! FIELDS " + " ".join(fields) + "\n" +
                          "".join(" ".join(str(value) for value in row) + "\n" for row in rows))

    def outputs(self, path, **options):
        return create_reweighted_fes(path, self.root / "fes", **options)

    def read_output(self, outputs):
        with Path(outputs[0]).open() as stream:
            rows = list(csv.DictReader(stream))
        return rows, json.loads(Path(outputs[2]).read_text())

    def test_zero_bias_and_empty_bins(self):
        path = self.colvar(["step", "value", "rbias"], [[0, .1, 0], [1, .2, 0], [2, .6, 0]])
        rows, metadata = self.read_output(self.outputs(path, thermal_energy=2, bins=(4,), bounds=((1, 0, 1),)))
        self.assertEqual([int(row["count"]) for row in rows], [2, 0, 1, 0])
        self.assertEqual(rows[1]["fes"], "")
        self.assertEqual(float(rows[0]["fes"]), 0.)
        self.assertAlmostEqual(float(rows[2]["fes"]), 2 * math.log(2))
        self.assertAlmostEqual(sum(float(row["probability_density"]) * .25 for row in rows), 1.)
        self.assertAlmostEqual(metadata["weight_ess"], 3.)
        self.assertEqual(metadata["empty_bins"], 2)

    def test_known_free_energy_and_gauge_offset_invariance(self):
        # One observation at each uniform bin center represents uniform biased
        # sampling under V=-F. Bias correction must recover the known F.
        centers = -2 + (np.arange(40) + .5) * .1
        energy = .7 * centers**2 + .2 * centers
        path = self.colvar(["step", "value", "bias", "rct"], [
            [index, coordinate, -potential + (10000 if index > 19 else -10000),
             10000 if index > 19 else -10000]
            for index, (coordinate, potential) in enumerate(zip(centers, energy))])
        rows, metadata = self.read_output(self.outputs(path, thermal_energy=.8, bins=(40,), bounds=((1, -2, 2),)))
        actual = np.array([float(row["fes"]) for row in rows])
        np.testing.assert_allclose(actual, energy - energy.min(), atol=4e-12)
        self.assertEqual(metadata["weight_source"], "bias-offset")

    def test_extreme_logweights_and_known_ess(self):
        path = self.colvar(["step", "value", "log_weight"], [[0, .2, 10000], [1, .8, 10000 - math.log(3)]])
        rows, metadata = self.read_output(self.outputs(path, thermal_energy=1, bins=(2,), bounds=((1, 0, 1),)))
        self.assertAlmostEqual(float(rows[1]["fes"]), math.log(3), places=11)
        self.assertAlmostEqual(metadata["weight_ess"], 1.6, places=11)
        self.assertTrue(all(math.isfinite(float(row["probability_density"])) for row in rows))

    def test_two_dimensional_projection_axis_order(self):
        rows = []
        for first in range(4):
            for second in range(3):
                for hidden in (-.5, .5):
                    x, y = first + .5, second + .5
                    rows.append([len(rows), x, y, hidden, -(x*x + 2*y)])
        path = self.colvar(["step", "s1", "s2", "s3", "log_weight"], rows)
        output, metadata = self.read_output(self.outputs(path, thermal_energy=1, axes=(2, 1),
                                                         bins=(3, 4), bounds=((1, 0, 4), (2, 0, 3))))
        self.assertEqual(list(output[0])[:2], ["s2", "s1"])
        expected = np.array([float(row["s1"])**2 + 2 * float(row["s2"]) for row in output])
        expected -= expected.min()
        np.testing.assert_allclose([float(row["fes"]) for row in output], expected, atol=2e-14)
        self.assertTrue(all(int(row["count"]) == 2 for row in output))
        self.assertIn("marginal", metadata["interpretation"])

    def test_wt_offset_formula_and_constant_shift(self):
        values = np.array([0., .1, .5, 1.])
        kbt, gamma = .6, 7.
        reference = kbt * math.log(np.exp(gamma * values / ((gamma - 1) * kbt)).sum() /
                                    np.exp(values / ((gamma - 1) * kbt)).sum())
        self.assertAlmostEqual(wt_offset(values, kbt, gamma), reference, places=14)
        self.assertAlmostEqual(wt_offset(values + 10000, kbt, gamma) - 10000, reference, places=11)
        self.assertEqual(wt_offset(np.zeros(10), kbt, gamma), 0.)
        for factor in (1, .5, float("inf")):
            with self.assertRaises(ValueError):
                wt_offset(values, kbt, factor)

    def test_offline_ct_matches_independent_integral_and_same_step(self):
        history = native(self.root / "HILLS", sigma=(.2,), rows=[[10, -.3, .1], [20, .4, .2], [20, -.1, .1]])
        grid = -1 + (np.arange(1024) + .5) * (2 / 1024)
        correction = WTOffset(history, ((1, -1, 1),), bins=(1024,))
        self.assertEqual(correction.at(0, [0.]), 0.)
        for step in (10, 20, 30):
            values = np.zeros(len(grid))
            for record in history.records(step):
                values += record.heights[0] * np.exp(-.5 * ((grid - record.centers[0]) / .2)**2)
            kbt, gamma = history.metadata.thermal_energy, history.metadata.bias_factor
            independent = kbt * math.log(np.exp(gamma * values / ((gamma-1)*kbt)).sum() /
                                        np.exp(values / ((gamma-1)*kbt)).sum())
            self.assertAlmostEqual(correction.at(step, [.1]), independent, places=13)
        self.assertEqual(correction.hills, 3)
        self.assertEqual(correction.updates, 2)
        correction.finish()

    def test_offline_ct_full_pipeline_and_discard(self):
        history = native(self.root / "HILLS", sigma=(.3,), rows=[[1, .2, .1], [3, .7, .15]])
        records = []
        for step in range(6):
            value = .2 if step % 2 == 0 else .7
            bias = float(evaluate_bias(history, [[value]], through_step=step)[0])
            records.append([step, value, bias])
        path = self.colvar(["step", "value", "bias_eV"], records)
        outputs = self.outputs(path, history=history.path, bins=(2,), bounds=((1, 0, 1),),
                               ct_bounds=((1, -2, 3),), ct_bins=(1024,), from_step=2)
        _, metadata = self.read_output(outputs)
        self.assertEqual(metadata["weight_source"], "wt-hills")
        self.assertEqual(metadata["first_step"], 2)
        self.assertEqual(metadata["offset"]["hills_applied"], 2)
        with Path(outputs[3]).open() as stream:
            weights = list(csv.DictReader(stream))
        first_offset = WTOffset(history, ((1, -2, 3),), bins=(1024,)).at(2, [.2])
        expected = (records[2][2] - first_offset) / history.metadata.thermal_energy
        self.assertAlmostEqual(float(weights[0]["log_weight"]), expected, places=13)

    def test_missing_normalization_fails_and_static_is_explicit(self):
        path = self.colvar(["step", "value", "bias"], [[0, .2, 0], [1, .8, 1]])
        with self.assertRaisesRegex(ValueError, "no normalized"):
            self.outputs(path, thermal_energy=1, bins=(2,), bounds=((1, 0, 1),))
        rows, metadata = self.read_output(self.outputs(path, thermal_energy=1, bins=(2,),
                                                       bounds=((1, 0, 1),), weight_source="static-bias"))
        self.assertEqual(metadata["weight_source"], "static-bias")
        self.assertAlmostEqual(float(rows[0]["fes"]), 1.)

    def test_pb_supplied_weights_and_ct_rejection(self):
        history = native(self.root / "HILLS", mode="PBMETAD", sigma=(.2, .3))
        path = self.colvar(["step", "s1", "s2", "rbias"], [[0, .2, .8, 0], [1, .8, .2, 0]])
        _, metadata = self.read_output(self.outputs(path, history=history.path, bins=(2,), bounds=((1, 0, 1), (2, 0, 1))))
        self.assertEqual(metadata["weight_source"], "rbias")
        with self.assertRaisesRegex(ValueError, "full-HILLS"):
            WTOffset(history, ((1, 0, 1), (2, 0, 1)))

    def test_periodic_wrap_and_range_retention(self):
        path = self.colvar(["step", "value", "rbias"], [[0, -.1, 0], [1, .1, 0], [2, 1.1, 0]])
        rows, _ = self.read_output(self.outputs(path, thermal_energy=1, bins=(2,), periods=((1, 1),)))
        self.assertEqual([int(row["count"]) for row in rows], [2, 1])
        outputs = create_reweighted_fes(path, self.root / "cropped", thermal_energy=1,
                                        bins=(2,), bounds=((1, 0, 1),))
        _, metadata = self.read_output(outputs)
        self.assertEqual(metadata["observations_outside_histogram"], 2)
        self.assertAlmostEqual(metadata["retained_weight_fraction"], 1 / 3)

    def test_ct_domain_and_grid_validation(self):
        history = native(self.root / "HILLS")
        with self.assertRaisesRegex(ValueError, "every biased"):
            WTOffset(history, ())
        with self.assertRaisesRegex(ValueError, "one million"):
            WTOffset(history, ((1, -1, 1),), bins=(1000001,))
        correction = WTOffset(history, ((1, -1, 1),))
        with self.assertRaisesRegex(ValueError, "outside"):
            correction.at(20, [2.])
        non_wt = native(self.root / "ordinary", wt=0)
        with self.assertRaisesRegex(ValueError, "WTMetaD"):
            WTOffset(non_wt, ((1, -1, 1),))

    def test_colvar_int64_headerless_csv_and_errors(self):
        first_step = 2**53 + 100
        path = self.colvar(["step", "value", "rbias"], [[first_step, .2, 0], [first_step + 1, .8, 0]])
        self.assertEqual(next(ColvarReader(path).records())[1]["step"], first_step)
        plain = self.write("plain", "0 .2 0\n1 .8 0\n")
        with self.assertRaisesRegex(ValueError, "header"):
            ColvarReader(plain)
        self.assertEqual(len(list(ColvarReader(plain, ["step", "value", "rbias"]).records())), 2)
        numeric_csv = self.write("samples.csv", "step,value,rbias\n0,.2,0\n1,.8,0\n")
        self.assertEqual(len(list(ColvarReader(numeric_csv).records())), 2)
        bad_cases = ["#! FIELDS step value rbias\n0 .2 0\n0 .8 0\n",
                     "#! FIELDS step value rbias\n0.5 .2 0\n",
                     "#! FIELDS step value rbias\n0 nan 0\n",
                     "#! FIELDS step value rbias\n0 .2",
                     "#! FIELDS step value rbias\n0 .2 0\n#! FIELDS step changed rbias\n"]
        for index, text in enumerate(bad_cases):
            with self.subTest(index=index), self.assertRaises(ValueError):
                list(ColvarReader(self.write("bad%d" % index, text)).records())

    def test_irregular_time_and_input_output_protection(self):
        path = self.colvar(["step", "value", "rbias"], [[0, .2, 0], [1, .8, 0], [3, .4, 0]])
        with self.assertRaisesRegex(ValueError, "equally spaced"):
            self.outputs(path, thermal_energy=1)
        sample = self.write("collision.csv", "step,value,rbias\n0,.2,0\n1,.8,0\n")
        with self.assertRaisesRegex(ValueError, "aliases"):
            create_reweighted_fes(sample, self.root / "collision", thermal_energy=1, force=True)

    def test_default_reweight_old_hills_explicit_and_cli(self):
        history = native(self.root / "HILLS", sigma=(.2,))
        path = self.colvar(["step", "value", "rbias"], [[0, -.2, 0], [1, .2, 0]])
        output = create_preview(history, self.root / "default", quantity="fes", bins=(2,), bounds=((1, -1, 1),))
        self.assertEqual(json.loads(Path(output[2]).read_text())["method"], "reweight")
        old = create_preview(history, self.root / "old", quantity="fes", method="hills", bins=(5,))
        self.assertEqual(json.loads(Path(old[2]).read_text())["method"], "hills")
        with self.assertRaisesRegex(ValueError, "marginal"):
            create_preview(history, self.root / "slice", quantity="fes", fixed=((2, 0),))
        command = [sys.executable, str(ROOT / "tools" / "metadtoy.py"), "fes", "reconstruct", str(path),
                   "--kbt", ".025", "--out", str(self.root / "cli"), "--bins", "2", "--range", "1", "-1", "1"]
        result = subprocess.run(command, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        path.unlink()
        with self.assertRaisesRegex(ValueError, "default FES"):
            create_preview(history, self.root / "missing", quantity="fes")



    def test_offset_multidimensional_quadrature(self):
        for dimension in (2, 3):
            with self.subTest(dimension=dimension):
                history = native(self.root / ("HILLS%d" % dimension), sigma=(.2,) * dimension,
                                 rows=[[10] + [.3] * dimension + [.1]])
                grid = (np.arange(12) + .5) / 12
                points = np.stack(np.meshgrid(*([grid] * dimension), indexing="ij"), axis=-1).reshape(-1, dimension)
                bias = .1 * np.exp(-.5 * (((points - .3) / .2)**2).sum(axis=1))
                kbt, gamma = history.metadata.thermal_energy, history.metadata.bias_factor
                reference = kbt * math.log(np.exp(gamma*bias/((gamma-1)*kbt)).sum() /
                                          np.exp(bias/((gamma-1)*kbt)).sum())
                correction = WTOffset(history, tuple((axis+1, 0, 1) for axis in range(dimension)), bins=(12,))
                self.assertAlmostEqual(correction.at(10, [.3]*dimension), reference, places=14)
                with self.assertRaisesRegex(ValueError, "backwards"):
                    correction.at(9, [.3]*dimension)

    def test_corrected_weight_priority_with_multiple_unused_bias_columns(self):
        path = self.colvar(["step", "value", "log_weight", "first.bias", "second.bias"],
                           [[0, .2, 0, 1, 2], [1, .8, 0, 3, 4]])
        _, metadata = self.read_output(self.outputs(path, thermal_energy=1, bins=(2,), bounds=((1, 0, 1),)))
        self.assertEqual(metadata["weight_columns"], {"log-weight": "log_weight"})

    def test_zero_hills_ct_and_changed_source_preserve_existing_output(self):
        history = native(self.root / "HILLS", rows=[])
        path = self.colvar(["step", "value", "bias"], [[0, .2, 0], [1, .8, 0]])
        outputs = self.outputs(path, history=history.path, bins=(2,), ct_bounds=((1, 0, 1),),
                               bounds=((1, 0, 1),))
        rows, metadata = self.read_output(outputs)
        self.assertEqual([float(row["fes"]) for row in rows], [0., 0.])
        self.assertEqual(metadata["offset"]["hills_applied"], 0)
        original = Path(outputs[0]).read_bytes()
        from metadtoy_cli import reweight
        original_fingerprint = reweight.fingerprint
        counter = [0]
        def changed_fingerprint(source):
            counter[0] += 1
            if counter[0] == 3:
                with path.open("a") as stream:
                    stream.write("2 .3 0\n")
            return original_fingerprint(source)
        with mock.patch.object(reweight, "fingerprint", changed_fingerprint):
            with self.assertRaisesRegex(ValueError, "source changed"):
                self.outputs(path, history=history.path, bins=(2,), ct_bounds=((1, 0, 1),),
                             bounds=((1, 0, 1),), force=True)
        self.assertEqual(Path(outputs[0]).read_bytes(), original)


if __name__ == "__main__":
    unittest.main(verbosity=2)
