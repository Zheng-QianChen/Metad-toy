#!/usr/bin/env python3
"""Trajectory geometry, identity, provenance and CLI regression checks."""
import csv
import gzip
import itertools
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
from metadtoy_cli.trajectory import TrajectoryReader
from metadtoy_cli.descriptors import DistanceFeatures, minimum_image
from metadtoy_cli.dataset import build_dataset


def xyz_frame(distance=1.2, step=None, cell="10 0 0 0 10 0 0 0 10", ids=False,
              pbc="T T T", time=None):
    metadata = ['Properties=species:S:1:pos:R:3' + (':id:I:1' if ids else '')]
    if cell is not None:
        metadata.append('Lattice="%s"' % cell)
    if pbc is not None:
        metadata.append('pbc="%s"' % pbc)
    if step is not None:
        metadata.append("Step=%d" % step)
    if time is not None:
        metadata.append("Time=%.17g" % time)
    return "2\n%s\nO 0 0 0%s\nO %.17g 0 0%s\n" % (
        " ".join(metadata), " 10" if ids else "", distance, " 20" if ids else "")


def dump_frame(step=0, rows=None, coordinates="x y z", box_header="pp pp pp",
               bounds="0 10\n0 10\n0 10", prefix=""):
    rows = rows or ["10 1 0 0 0", "20 1 9 0 0"]
    return (prefix + "ITEM: TIMESTEP\n%d\nITEM: NUMBER OF ATOMS\n%d\n"
            "ITEM: BOX BOUNDS %s\n%s\nITEM: ATOMS id type %s\n%s\n") % (
                step, len(rows), box_header, bounds, coordinates, "\n".join(rows))


class TrajectoryTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)

    def tearDown(self):
        self.directory.cleanup()

    def write(self, name, content):
        path = self.root / name
        path.write_text(content, encoding="utf-8")
        return path

    def config(self, identity="id", image="minimum"):
        value = {"schema_version": 1, "identity": identity, "length_unit": "angstrom",
                 "features": [{"name": "r_OO", "kind": "distance",
                               "atoms": [10, 20] if identity == "id" else [0, 1], "image": image}]}
        return self.write("features.json", json.dumps(value)), value

    def test_id_reordering_int64_and_gzip(self):
        first_id, second_id, first_step = 2**53 + 7, 2**53 + 9, 2**53 + 101
        text = dump_frame(first_step, ["%d 1 0 0 0" % first_id, "%d 1 9 0 0" % second_id])
        text += dump_frame(first_step + 1, ["%d 1 8 0 0" % second_id, "%d 1 0 0 0" % first_id])
        path = self.root / "trajectory.dump.gz"
        with gzip.open(path, "wt") as stream:
            stream.write(text)
        frames = list(TrajectoryReader(path).frames())
        self.assertEqual(frames[0].step, first_step)
        self.assertEqual(frames[1].atom_ids, (second_id, first_id))
        _, config = self.config()
        config["features"][0]["atoms"] = [first_id, second_id]
        evaluator = DistanceFeatures(config)
        self.assertEqual([evaluator.evaluate(frame)[0] for frame in frames], [[1.0], [2.0]])

    def test_restricted_triclinic_scaled_extxyz_equivalence(self):
        # True origin=(1,2,3), row vectors=(4,0,0),(-1,3,0),(.5,.75,2).
        cell = np.array([[4., 0, 0], [-1., 3, 0], [.5, .75, 2.]])
        fractions = np.array([[.1, .2, .3], [.9, .8, .7]])
        positions = fractions @ cell + [1, 2, 3]
        path = self.write("triclinic.dump", dump_frame(
            coordinates="xs ys zs", rows=["10 1 .1 .2 .3", "20 1 .9 .8 .7"],
            box_header="xy xz yz pp ff pp", bounds="0 5.5 -1\n2 5.75 .5\n3 5 .75"))
        frame = next(TrajectoryReader(path).frames())
        np.testing.assert_allclose(frame.cell, cell)
        np.testing.assert_allclose(frame.origin, [1, 2, 3])
        np.testing.assert_allclose(frame.positions, positions)
        self.assertEqual(frame.pbc, (True, False, True))
        text = '2\nProperties=id:I:1:species:S:1:pos:R:3 Lattice="%s" pbc="T F T"\n' % " ".join(map(str, cell.ravel()))
        text += "\n".join("%d O %s" % (atom, " ".join(map(str, pos)))
                          for atom, pos in zip((10, 20), positions)) + "\n"
        other = next(TrajectoryReader(self.write("triclinic.xyz", text)).frames())
        _, config = self.config()
        np.testing.assert_allclose(DistanceFeatures(config).evaluate(frame)[0],
                                   DistanceFeatures(config).evaluate(other)[0])

    def test_general_triclinic_and_explicit_pbc(self):
        path = self.write("general.dump", dump_frame(
            box_header="abc origin", bounds="0 4 0 1\n-3 0 0 2\n0 0 2 3",
            coordinates="xsu ysu zsu", rows=["10 1 0 0 0", "20 1 1.25 0 0"]))
        frame = next(TrajectoryReader(path, pbc=["T", "T", "T"]).frames())
        np.testing.assert_allclose(frame.positions, [[1, 2, 3], [1, 7, 3]])
        _, config = self.config()
        self.assertAlmostEqual(DistanceFeatures(config).evaluate(frame)[0][0], 1.)
        config["features"][0]["image"] = "direct"
        self.assertAlmostEqual(DistanceFeatures(config).evaluate(frame)[0][0], 5.)

    def test_triclinic_minimum_against_independent_enumeration(self):
        random = np.random.default_rng(71)
        cells = [np.array([[2., 0, 0], [1.8, .9, 0], [.5, .8, 1.3]]),
                 np.array([[2., .4, -.2], [.3, 2.1, .8], [-.6, .2, 1.7]])]
        rounding_failure = False
        for cell in cells:
            for pbc in ((True, True, True), (True, False, True), (False, True, False)):
                shifts = np.array(list(itertools.product(*(range(-6, 7) if flag else (0,) for flag in pbc))))
                for iteration in range(30):
                    fractional = random.uniform(-1.3, 1.3, size=3)
                    delta = fractional @ cell
                    candidates = delta - shifts @ cell
                    expected = np.min(np.linalg.norm(candidates, axis=1))
                    actual = np.linalg.norm(minimum_image(delta, cell, pbc))
                    self.assertAlmostEqual(actual, expected, places=12)
                    rounded = delta - np.where(pbc, np.rint(fractional), 0) @ cell
                    rounding_failure |= np.linalg.norm(rounded) > actual + 1e-6
        self.assertTrue(rounding_failure, "fixture must exercise a case where rounding is wrong")

    def test_mixed_boundary_and_changing_cell(self):
        path = self.write("vary.xyz", xyz_frame(9, step=0) + xyz_frame(9, step=1, cell="20 0 0 0 10 0 0 0 10"))
        frames = list(TrajectoryReader(path).frames())
        _, config = self.config("row")
        evaluator = DistanceFeatures(config, True)
        self.assertEqual([evaluator.evaluate(frame)[0] for frame in frames], [[1.], [9.]])
        self.assertTrue(TrajectoryReader(path).summary()["cell_changes"])
        np.testing.assert_allclose(minimum_image(np.array([9., 9., 9.]), np.diag([10., 10., 10.]),
                                                 (True, False, True)), [-1, 9, -1])

    def test_xyz_properties_order_time_no_invented_step(self):
        text = ('2\nTime = 1.25 Lattice = "10 0 0 0 10 0 0 0 10" pbc="T T T" '
                'Properties=force:R:3:id:I:1:pos:R:3:species:S:1:active:L:1\n'
                '0 0 0 20 1.5 0 0 Fe3+ T\n0 0 0 10 0 0 0 Fe3+ F\n')
        frame = next(TrajectoryReader(self.write("columns.xyz", text)).frames())
        self.assertIsNone(frame.step)
        self.assertEqual(frame.time, 1.25)
        self.assertEqual(frame.labels, ("Fe3+", "Fe3+"))
        _, config = self.config()
        self.assertEqual(DistanceFeatures(config).evaluate(frame)[0], [1.5])

    def test_fixed_rows_require_acknowledgement_and_detect_changes(self):
        _, config = self.config("row")
        with self.assertRaisesRegex(ValueError, "assume-fixed-order"):
            DistanceFeatures(config)
        frames = list(TrajectoryReader(self.write("rows.xyz", xyz_frame(ids=True) + xyz_frame(ids=True))).frames())
        evaluator = DistanceFeatures(config, True)
        evaluator.evaluate(frames[0])
        frames[1].atom_ids = tuple(reversed(frames[1].atom_ids))
        with self.assertRaisesRegex(ValueError, "identity/type"):
            evaluator.evaluate(frames[1])
        frames[1].atom_ids = frames[0].atom_ids
        frames[1].labels = ("O", "H")
        with self.assertRaisesRegex(ValueError, "identity/type"):
            evaluator.evaluate(frames[1])

    def test_missing_pbc_requires_argument_for_minimum(self):
        path = self.write("plain.xyz", "2\nplain molecule\nO 0 0 0\nO 1.2 0 0\n")
        _, config = self.config("row")
        frame = next(TrajectoryReader(path).frames())
        self.assertIsNone(frame.cell)
        with self.assertRaisesRegex(ValueError, "explicit PBC"):
            DistanceFeatures(config, True).evaluate(frame)
        frame = next(TrajectoryReader(path, pbc=["F"] * 3).frames())
        self.assertEqual(DistanceFeatures(config, True).evaluate(frame)[0], [1.2])
        path = self.write("periodic.xyz", xyz_frame())
        with self.assertRaisesRegex(ValueError, "conflicts"):
            list(TrajectoryReader(path, pbc=["F"] * 3).frames())

    def test_lammps_units_time_and_coordinates(self):
        path = self.write("time.dump", dump_frame(
            prefix="ITEM: UNITS\nmetal\nITEM: TIME\n.5\n",
            coordinates="xu yu zu", rows=["10 1 0 0 0", "20 1 11 0 0"]))
        frame = next(TrajectoryReader(path).frames())
        self.assertEqual(frame.time, .5)
        self.assertEqual(frame.metadata["unit_style"], "metal")
        _, config = self.config()
        self.assertEqual(DistanceFeatures(config).evaluate(frame)[0], [1.])
        config["features"][0]["image"] = "direct"
        self.assertEqual(DistanceFeatures(config).evaluate(frame)[0], [11.])

    def test_invalid_frames(self):
        fixtures = {
            "truncated": xyz_frame()[:-9],
            "duplicate_id": xyz_frame(ids=True).replace(" 20\n", " 10\n"),
            "nan": xyz_frame().replace("O 0 0 0", "O nan 0 0"),
            "singular": xyz_frame(cell="10 0 0 0 0 0 0 0 10"),
            "no_cell": xyz_frame(cell=None),
            "unknown_dtype": xyz_frame().replace("pos:R:3", "pos:Q:3"),
            "duplicate_property": xyz_frame().replace("pos:R:3", "pos:R:3:pos:R:3"),
            "wrong_width": xyz_frame().replace("O 0 0 0", "O 0 0 0 9"),
            "duplicate_metadata": xyz_frame().replace('pbc="T T T"', 'pbc="T T T" pbc="F F F"'),
            "backwards_step": xyz_frame(step=10) + xyz_frame(step=5),
            "backwards_time": xyz_frame(time=10) + xyz_frame(time=5),
            "empty": "",
            "bad_dump_flag": dump_frame(box_header="pf pp pp"),
        }
        for name, text in fixtures.items():
            with self.subTest(name=name), self.assertRaises(ValueError):
                list(TrajectoryReader(self.write(name, text)).frames())

    def test_dataset_csv_frame_manifest_and_split(self):
        path = self.write("sample.xyz", "".join(xyz_frame(1 + index * .01, step=2**53 + index,
                                                         ids=True, time=index * .1) for index in range(10)))
        config_path, _ = self.config()
        outputs = build_dataset([path], config_path, self.root / "data", validation_fraction=.2, gap_frames=2)
        with Path(outputs[0]).open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(rows[0]["step"], str(2**53))
        self.assertEqual([row["split"] for row in rows], ["train"] * 6 + ["gap"] * 2 + ["validation"] * 2)
        np.testing.assert_allclose([float(row["r_OO"]) for row in rows], 1 + np.arange(10) * .01)
        manifest = json.loads(Path(outputs[2]).read_text())
        self.assertEqual(manifest["sources"][0]["selected_frames"], 10)
        self.assertEqual(len(manifest["sources"][0]["sha256"]), 64)
        frame_rows = [json.loads(line) for line in Path(outputs[1]).read_text().splitlines()]
        self.assertEqual(frame_rows[0]["selection"][0]["id"], 10)
        self.assertEqual(frame_rows[-1]["frame"], 9)
        self.assertEqual(frame_rows[0]["cell_rows"], np.diag([10., 10., 10.]).tolist())

    def test_multiple_sources_stride_and_no_step(self):
        first = self.write("a.xyz", "".join(xyz_frame(ids=True) for index in range(7)))
        second = self.write("b.xyz", "".join(xyz_frame(ids=True) for index in range(5)))
        config_path, _ = self.config()
        outputs = build_dataset([first, second], config_path, self.root / "strided", stride=2)
        with Path(outputs[0]).open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual([(row["source"], row["frame"]) for row in rows],
                         [("0", "0"), ("0", "2"), ("0", "4"), ("0", "6"), ("1", "0"), ("1", "2"), ("1", "4")])
        self.assertTrue(all(row["step"] == row["time"] == "" for row in rows))
        self.assertTrue(all(row["split"] == "unassigned" for row in rows))

    def test_failed_build_preserves_outputs_and_inputs(self):
        path = self.write("bad.xyz", xyz_frame(ids=True) + "2\ntruncated\n")
        config_path, _ = self.config()
        output = self.write("safe.csv", "original")
        with self.assertRaisesRegex(ValueError, "truncated"):
            build_dataset([path], config_path, self.root / "safe", force=True)
        self.assertEqual(output.read_text(), "original")
        self.assertFalse((self.root / "safe.json").exists())
        with self.assertRaisesRegex(ValueError, "every input"):
            build_dataset([path], config_path, self.root / "features", force=True)
        with self.assertRaisesRegex(ValueError, "distinct"):
            build_dataset([path, path], config_path, self.root / "unique")

    def test_source_change_detected(self):
        path = self.write("changing.xyz", xyz_frame(ids=True))
        config_path, _ = self.config()
        original = DistanceFeatures.evaluate
        def modify_config(evaluator, frame):
            result = original(evaluator, frame)
            config_path.write_text(config_path.read_text() + " ")
            return result
        with mock.patch.object(DistanceFeatures, "evaluate", modify_config):
            with self.assertRaisesRegex(ValueError, "input changed"):
                build_dataset([path], config_path, self.root / "changed")
        self.assertFalse((self.root / "changed.csv").exists())

    def test_invalid_config_selection_and_split(self):
        path = self.write("one.xyz", xyz_frame(ids=True))
        config_path, config = self.config()
        with self.assertRaisesRegex(ValueError, "too few"):
            build_dataset([path], config_path, self.root / "small", validation_fraction=.2)
        with self.assertRaisesRegex(ValueError, "gap"):
            build_dataset([path], config_path, self.root / "nogap", validation_fraction=.2, gap_frames=0)
        frame = next(TrajectoryReader(path).frames())
        config["features"][0]["atoms"] = [10, 30]
        with self.assertRaisesRegex(ValueError, "missing"):
            DistanceFeatures(config).evaluate(frame)
        config["features"][0]["atoms"] = [0, 1]
        config["identity"] = "row"
        frame.positions[:] = 0
        with self.assertRaisesRegex(ValueError, "nonzero"):
            DistanceFeatures(config, True).evaluate(frame)

    def test_units_mismatch_and_duplicate_config_keys(self):
        path = self.write("units.dump", dump_frame(prefix="ITEM: UNITS\nsi\n"))
        config_path, _ = self.config()
        with self.assertRaisesRegex(ValueError, "length_unit"):
            build_dataset([path], config_path, self.root / "units")
        config_path.write_text('{"schema_version":1,"schema_version":1}')
        with self.assertRaisesRegex(ValueError, "duplicate"):
            build_dataset([path], config_path, self.root / "duplicate")

    def test_cli_commands_and_controlled_error(self):
        path = self.write("cli.xyz", xyz_frame(ids=True))
        config_path, _ = self.config()
        command = [sys.executable, str(ROOT / "tools" / "metadtoy.py")]
        result = subprocess.run(command + ["trajectory", "inspect", str(path)], text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["frames"], 1)
        result = subprocess.run(command + ["dataset", "build", str(path), "--features", str(config_path),
                                           "--out", str(self.root / "cli")], text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        result = subprocess.run(command + ["dataset", "build", str(path), "--features", str(config_path),
                                           "--out", str(self.root / "cli")], text=True, capture_output=True)
        self.assertEqual(result.returncode, 2)
        self.assertNotIn("Traceback", result.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
