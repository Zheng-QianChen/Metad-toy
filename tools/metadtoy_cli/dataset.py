"""Build reproducible descriptor datasets without retaining a whole trajectory."""
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
from .analysis import output_transaction
from .descriptors import DistanceFeatures
from .trajectory import TrajectoryReader


def fingerprint(path):
    """Hash input bytes in bounded blocks; reject a writer changing the file."""
    path = Path(path).resolve()
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("input changed while hashing: %s" % path)
    return {"path": str(path), "bytes": after.st_size, "mtime_ns": after.st_mtime_ns,
            "sha256": digest.hexdigest()}


def _strict_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate configuration key: %s" % key)
        result[key] = value
    return result


def build_dataset(inputs, config_path, prefix, format="auto", pbc=None,
                  assume_fixed_order=False, stride=1, validation_fraction=0.,
                  gap_frames=1, force=False):
    """Write feature CSV, per-frame JSONL and a source/config manifest.

    Source/row pairs are stable join keys. No MD step or physical time is
    invented when the trajectory omits it. Only selected atom identities are
    repeated in JSONL; the complete input is identified by SHA256.

    Splits use each source's chronological selected-frame order. A gap is kept
    as split='gap' in the export, so training consumers can exclude it explicitly.
    """
    if stride < 1 or gap_frames < 0:
        raise ValueError("stride must be positive and gap_frames nonnegative")
    if not math.isfinite(validation_fraction) or not 0 <= validation_fraction < 1:
        raise ValueError("validation_fraction must be in [0,1)")
    if validation_fraction > 0 and gap_frames < 1:
        raise ValueError("chronological validation requires at least one gap frame")
    config_path = Path(config_path).resolve()
    sources = [Path(path).resolve() for path in inputs]
    if not sources or len(set(sources)) != len(sources) or config_path in sources:
        raise ValueError("provide distinct trajectory inputs and descriptor configuration")
    protected = [config_path] + sources
    snapshots = [fingerprint(path) for path in protected]
    config = json.loads(config_path.read_text(encoding="utf-8"), object_pairs_hook=_strict_object)
    descriptors = DistanceFeatures(config, assume_fixed_order)
    prefix = Path(prefix)
    paths = [Path(str(prefix) + suffix).resolve() for suffix in (".csv", ".frames.jsonl", ".json")]
    if any(path in protected for path in paths):
        raise ValueError("output paths must differ from every input")
    for output in paths:
        if output.exists() and any(output.samefile(source) for source in protected):
            raise ValueError("output aliases an input file")
    columns = ["source", "frame", "step", "time", "split"] + descriptors.names
    counts, total_frames, ranges, time_presence = [], [], [], []
    unit_map = {"metal": "angstrom", "real": "angstrom", "lj": "reduced",
                "si": "m", "cgs": "cm", "electron": "bohr",
                "micro": "micrometer", "nano": "nanometer"}
    with output_transaction(paths, config_path, force) as staged:
        # A spool allows chronological partitioning once each source's count
        # is known, without reloading or reparsing its coordinate trajectory.
        spool = staged[0].with_suffix(".spool")
        with spool.open("w", encoding="utf-8", newline="") as csv_stream, \
                staged[1].open("w", encoding="utf-8") as frame_stream:
            writer = csv.writer(csv_stream)
            writer.writerow(columns)
            for source_index, source in enumerate(sources):
                descriptors = DistanceFeatures(config, assume_fixed_order)
                selected_count = frame_count = 0
                first_frame = last_frame = None
                observed_time = False
                for frame in TrajectoryReader(source, format, pbc).frames():
                    frame_count += 1
                    style = frame.metadata.get("unit_style")
                    if style in unit_map and config["length_unit"] != unit_map[style]:
                        raise ValueError("length_unit conflicts with LAMMPS units %s (expected %s)" %
                                         (style, unit_map[style]))
                    # Validate identity on every frame, including ones omitted
                    # by stride; reordering between selected samples is invalid.
                    values, selection = descriptors.evaluate(frame)
                    if frame.index % stride:
                        continue
                    writer.writerow([source_index, frame.index,
                                     "" if frame.step is None else frame.step,
                                     "" if frame.time is None else format_float(frame.time), "unassigned"] +
                                    [format_float(value) for value in values])
                    detail = {"source": source_index, "frame": frame.index, "step": frame.step,
                              "time": frame.time, "atoms": len(frame.positions),
                              "cell_rows": None if frame.cell is None else frame.cell.tolist(),
                              "origin": frame.origin.tolist(), "pbc": frame.pbc,
                              "selection": selection, "reader": frame.metadata}
                    frame_stream.write(json.dumps(detail, allow_nan=False) + "\n")
                    if first_frame is None:
                        first_frame = frame.index
                    last_frame = frame.index
                    observed_time = observed_time or frame.time is not None
                    selected_count += 1
                counts.append(selected_count)
                total_frames.append(frame_count)
                ranges.append([first_frame, last_frame])
                time_presence.append(observed_time)
        partitions = []
        for count in counts:
            if validation_fraction:
                validation_count = max(1, math.ceil(count * validation_fraction))
                train_count = count - validation_count - gap_frames
                if train_count < 1:
                    raise ValueError("too few selected frames for train/validation plus the requested gap")
                partitions.append((train_count, train_count + gap_frames))
            else:
                partitions.append((count, count))
        source_ordinals = Counter()
        split_counts = Counter()
        with spool.open(encoding="utf-8", newline="") as source_stream, \
                staged[0].open("w", encoding="utf-8", newline="") as destination:
            reader, writer = csv.reader(source_stream), csv.writer(destination)
            writer.writerow(next(reader))
            for row in reader:
                source_index = int(row[0])
                ordinal = source_ordinals[source_index]
                train_end, validation_start = partitions[source_index]
                split = ("unassigned" if not validation_fraction else
                         "train" if ordinal < train_end else
                         "gap" if ordinal < validation_start else "validation")
                row[4] = split
                writer.writerow(row)
                source_ordinals[source_index] += 1
                split_counts[split] += 1
        # A second byte hash makes the manifest describe the same snapshots
        # used for parsing, including gzip inputs and the descriptor JSON.
        for path, expected in zip(protected, snapshots):
            if fingerprint(path) != expected:
                raise ValueError("input changed during dataset build: %s" % path)
        manifest = {
            "schema_version": 1, "product": "distance_dataset", "config": config,
            "config_source": snapshots[0], "columns": columns,
            "sources": [dict(snapshot, frames=total_frames[index], selected_frames=counts[index],
                             selected_frame_range=ranges[index], time_present=time_presence[index])
                        for index, snapshot in enumerate(snapshots[1:])],
            "rows": sum(counts), "stride": stride, "format_request": format,
            "pbc_argument": pbc, "assume_fixed_order": assume_fixed_order,
            "split": {"method": "chronological_per_source" if validation_fraction else "none",
                      "validation_fraction": validation_fraction, "gap_selected_frames": gap_frames if validation_fraction else 0,
                      "counts": dict(split_counts)},
            "length_unit": config["length_unit"], "time_unit": "as stored; no conversion or inferred timestep",
            "cell_convention": "row lattice vectors: cartesian = fractional @ cell + origin",
            "identity_note": "row selection assumes fixed ordering; identical-species swaps without IDs are undetectable",
            "periodic_distance": "Euclidean closest image; skewed cells use bounded lattice enumeration",
            "products": {"features": paths[0].name, "frames": paths[1].name},
        }
        staged[2].write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return [str(path) for path in paths]


def format_float(value):
    return "%.17g" % value
