"""Streaming trajectory adapters for offline descriptor generation.

Frame positions are Cartesian [atom, xyz]; cell rows are the three lattice
vectors, so Cartesian = fractional @ cell + origin. IDs stay Python integers
to preserve large LAMMPS tags. Readers retain one frame at a time.
"""
from dataclasses import dataclass
from pathlib import Path
import gzip
import math
import re
import shlex


def finite_number(token):
    value = float(token)
    if not math.isfinite(value):
        raise ValueError("non-finite numeric value: %s" % token)
    return value


def integer(token):
    if not re.fullmatch(r"[+-]?\d+", str(token)):
        raise ValueError("expected an integer, got %s" % token)
    return int(token)


def parse_pbc(tokens):
    if len(tokens) != 3:
        raise ValueError("PBC requires three flags")
    mapping = {"t": True, "true": True, "1": True,
               "f": False, "false": False, "0": False}
    try:
        return tuple(mapping[token.lower()] for token in tokens)
    except KeyError:
        raise ValueError("PBC flags must be T/F, true/false or 1/0") from None


@dataclass
class Frame:
    index: int
    step: object
    time: object
    positions: object
    cell: object
    origin: object
    pbc: object
    atom_ids: object
    labels: tuple
    metadata: dict

    def validate(self):
        import numpy as np
        if not np.isfinite(self.positions).all():
            raise ValueError("coordinates must be finite")
        if self.atom_ids is not None:
            if min(self.atom_ids) < 0 or len(set(self.atom_ids)) != len(self.atom_ids):
                raise ValueError("atom IDs must be unique nonnegative integers")
        if self.cell is not None:
            if not np.isfinite(self.cell).all():
                raise ValueError("cell must be finite")
            # A finite nonsingular 3D cell gives a well-defined image lattice.
            # Singular 2D cells should be supplied with a finite third vector.
            if np.linalg.matrix_rank(self.cell) != 3:
                raise ValueError("cell must contain three independent lattice vectors")
        if self.pbc is not None and any(self.pbc) and self.cell is None:
            raise ValueError("periodic frame requires a cell")
        return self


class _Lines:
    """Line-numbered errors keep malformed large trajectories diagnosable."""
    def __init__(self, stream, path):
        self.stream, self.path, self.number = stream, path, 0

    def read(self, optional=False):
        text = self.stream.readline()
        if text == "":
            if optional:
                return None
            raise ValueError("truncated frame")
        self.number += 1
        return text.strip()

    def expect(self, text):
        actual = self.read()
        if actual != text:
            raise ValueError("expected '%s', got '%s'" % (text, actual))


class TrajectoryReader:
    def __init__(self, path, format="auto", pbc=None):
        self.path = Path(path).resolve()
        if format not in ("auto", "lammps-dump", "xyz"):
            raise ValueError("unsupported trajectory format: %s" % format)
        self.format = format
        self.pbc = None if pbc is None else parse_pbc(pbc)

    def frames(self):
        opener = gzip.open if self.path.suffix.lower() == ".gz" else open
        with opener(self.path, "rt", encoding="utf-8") as stream:
            lines = _Lines(stream, self.path)
            first = lines.read(optional=True)
            if first is None:
                raise ValueError("%s: empty trajectory" % self.path)
            format = self.format
            if format == "auto":
                format = "lammps-dump" if first.startswith("ITEM:") else "xyz"
            reader = self._lammps if format == "lammps-dump" else self._xyz
            previous_step = previous_time = None
            try:
                for frame in reader(lines, first):
                    if frame.pbc is None:
                        frame.pbc = self.pbc
                        frame.metadata["pbc_source"] = "argument" if self.pbc is not None else "missing"
                    elif self.pbc is not None and self.pbc != frame.pbc:
                        raise ValueError("--pbc conflicts with file PBC")
                    frame.validate()
                    if frame.step is not None:
                        if previous_step is not None and frame.step <= previous_step:
                            raise ValueError("steps must increase strictly; put restart segments in separate inputs")
                        previous_step = frame.step
                    if frame.time is not None:
                        if previous_time is not None and frame.time <= previous_time:
                            raise ValueError("times must increase strictly; put restart segments in separate inputs")
                        previous_time = frame.time
                    yield frame
            except ValueError as error:
                raise ValueError("%s:%d: %s" % (self.path, lines.number, error)) from error

    def _lammps(self, lines, first):
        import numpy as np
        index, units = 0, None
        header = first
        while header is not None:
            time = None
            # dump_modify units/time can insert these records before TIMESTEP.
            seen = set()
            while header in ("ITEM: UNITS", "ITEM: TIME"):
                if header in seen:
                    raise ValueError("duplicate %s record" % header)
                seen.add(header)
                if header == "ITEM: UNITS":
                    new_units = lines.read()
                    if units is not None and units != new_units:
                        raise ValueError("LAMMPS unit style changed within trajectory")
                    units = new_units
                else:
                    time = finite_number(lines.read())
                header = lines.read()
            if header != "ITEM: TIMESTEP":
                raise ValueError("expected ITEM: TIMESTEP")
            step = integer(lines.read())
            lines.expect("ITEM: NUMBER OF ATOMS")
            count = integer(lines.read())
            if count <= 0:
                raise ValueError("atom count must be positive")
            box_header = lines.read().split()
            if box_header[:3] != ["ITEM:", "BOX", "BOUNDS"]:
                raise ValueError("expected ITEM: BOX BOUNDS")
            options = box_header[3:]
            if options[:3] == ["xy", "xz", "yz"]:
                box_kind, flags, width = "restricted_triclinic", options[3:], 3
            elif options[:2] == ["abc", "origin"]:
                box_kind, flags, width = "general_triclinic", options[2:], 4
            else:
                box_kind, flags, width = "orthogonal", options, 2
            if flags:
                if len(flags) != 3 or any(not re.fullmatch(r"[pfsm]{2}", flag) for flag in flags):
                    raise ValueError("invalid BOX BOUNDS boundary flags")
                if any("p" in flag and flag != "pp" for flag in flags):
                    raise ValueError("periodic boundary flags must be pp")
                pbc = tuple(flag == "pp" for flag in flags)
            else:
                pbc = None
            rows = []
            for axis in range(3):
                row = [finite_number(value) for value in lines.read().split()]
                if len(row) != width:
                    raise ValueError("wrong number of BOX BOUNDS values")
                rows.append(row)
            bounds = np.array(rows)
            if box_kind == "general_triclinic":
                cell, origin = bounds[:, :3], bounds[:, 3]
            else:
                lower, upper = bounds[:, 0].copy(), bounds[:, 1].copy()
                xy = xz = yz = 0.0
                if box_kind == "restricted_triclinic":
                    xy, xz, yz = bounds[:, 2]
                    lower[0] -= min(0., xy, xz, xy + xz)
                    upper[0] -= max(0., xy, xz, xy + xz)
                    lower[1] -= min(0., yz)
                    upper[1] -= max(0., yz)
                lengths = upper - lower
                if np.any(lengths <= 0):
                    raise ValueError("box lengths must be positive")
                cell = np.array([[lengths[0], 0., 0.], [xy, lengths[1], 0.], [xz, yz, lengths[2]]])
                origin = lower
            atom_header = lines.read().split()
            if atom_header[:2] != ["ITEM:", "ATOMS"]:
                raise ValueError("expected ITEM: ATOMS")
            columns = atom_header[2:]
            if len(set(columns)) != len(columns):
                raise ValueError("duplicate ATOMS columns")
            # Prefer Cartesian wrapped coordinates if multiple representations
            # are present. The chosen representation is saved in frame metadata.
            coordinate_sets = (("x", "y", "z"), ("xu", "yu", "zu"),
                               ("xs", "ys", "zs"), ("xsu", "ysu", "zsu"))
            coordinates = next((names for names in coordinate_sets if all(name in columns for name in names)), None)
            if coordinates is None:
                raise ValueError("ATOMS requires x/y/z, xu/yu/zu, xs/ys/zs or xsu/ysu/zsu")
            coordinate_indices = [columns.index(name) for name in coordinates]
            id_index = columns.index("id") if "id" in columns else None
            label_name = next((name for name in ("element", "typelabel", "type") if name in columns), None)
            label_index = columns.index(label_name) if label_name is not None else None
            positions = np.empty((count, 3))
            atom_ids = [] if id_index is not None else None
            labels = []
            for atom in range(count):
                fields = lines.read().split()
                if len(fields) != len(columns):
                    raise ValueError("ATOMS row width does not match its header")
                positions[atom] = [finite_number(fields[column]) for column in coordinate_indices]
                if atom_ids is not None:
                    atom_id = integer(fields[id_index])
                    if atom_id <= 0:
                        raise ValueError("LAMMPS atom IDs must be positive")
                    atom_ids.append(atom_id)
                labels.append(fields[label_index] if label_index is not None else None)
            if coordinates[0] in ("xs", "xsu"):
                positions = positions @ cell + origin
            yield Frame(index, step, time, positions, cell, origin, pbc,
                        None if atom_ids is None else tuple(atom_ids), tuple(labels),
                        {"format": "lammps-dump", "box_kind": box_kind,
                         "coordinates": list(coordinates), "unit_style": units,
                         "label_kind": label_name, "pbc_source": "file" if pbc is not None else "missing"})
            index += 1
            header = lines.read(optional=True)

    def _xyz(self, lines, first):
        import numpy as np
        index, count_line = 0, first
        while count_line is not None:
            count = integer(count_line)
            if count <= 0:
                raise ValueError("atom count must be positive")
            comment = lines.read()
            # GPUMD writes the conventional quoted, flat key=value dialect.
            # Parse quotes before splitting values; reject duplicate keys.
            metadata = {}
            if re.search(r"\bProperties\s*=", comment):
                lexer = shlex.shlex(comment, posix=True, punctuation_chars="=")
                lexer.whitespace_split, lexer.commenters = True, ""
                tokens = list(lexer)
                if len(tokens) % 3:
                    raise ValueError("unsupported extended XYZ comment syntax (expected key=value)")
                for offset in range(0, len(tokens), 3):
                    key, equals, value = tokens[offset:offset + 3]
                    if equals != "=" or key in metadata:
                        raise ValueError("invalid or duplicate extended XYZ metadata")
                    metadata[key] = value
            elif re.search(r"\b(Lattice|pbc)\s*=", comment):
                raise ValueError("XYZ with Lattice/pbc must also declare Properties")
            extended = "Properties" in metadata
            properties = {}
            width = 0
            definition = metadata.get("Properties", "species:S:1:pos:R:3").split(":")
            if len(definition) % 3:
                raise ValueError("Properties must contain name:type:count triples")
            for offset in range(0, len(definition), 3):
                name, dtype, size = definition[offset:offset + 3]
                size = integer(size)
                if not name or name in properties or dtype not in ("S", "R", "I", "L") or size <= 0:
                    raise ValueError("invalid or duplicate Properties entry")
                properties[name] = (dtype, size, width)
                width += size
            if properties.get("pos", ())[:2] != ("R", 3):
                raise ValueError("XYZ requires pos:R:3")
            label_name = next((name for name in ("species", "Z", "type") if name in properties), None)
            if label_name is not None and properties[label_name][1] != 1:
                raise ValueError("species/Z/type must have one column")
            if "id" in properties and properties["id"][:2] != ("I", 1):
                raise ValueError("explicit IDs require id:I:1")
            lattice = metadata.get("Lattice")
            cell = None
            if lattice is not None:
                values = [finite_number(value) for value in lattice.split()]
                if len(values) != 9:
                    raise ValueError("Lattice requires nine values: ax ay az bx by bz cx cy cz")
                cell = np.array(values).reshape(3, 3)
            # Missing PBC stays unknown; the user can supply --pbc explicitly.
            # This avoids silently treating an undocumented XYZ as nonperiodic.
            pbc = parse_pbc(metadata["pbc"].split()) if "pbc" in metadata else None
            def optional_scalar(names, convert):
                present = [key for key in names if key in metadata]
                if len(present) > 1:
                    raise ValueError("ambiguous metadata keys: %s" % present)
                return convert(metadata[present[0]]) if present else None
            step = optional_scalar(("step", "Step", "timestep", "Timestep"), integer)
            time = optional_scalar(("time", "Time"), finite_number)
            positions = np.empty((count, 3))
            atom_ids = [] if "id" in properties else None
            labels = []
            for atom in range(count):
                fields = lines.read().split()
                if len(fields) != width:
                    raise ValueError("XYZ row width does not match Properties (plain XYZ requires four columns)")
                # Validate all declared typed columns even when only positions
                # are selected, so truncated/malformed rows cannot pass silently.
                for dtype, size, offset in properties.values():
                    for token in fields[offset:offset + size]:
                        if dtype == "R":
                            finite_number(token)
                        elif dtype == "I":
                            integer(token)
                        elif dtype == "L":
                            parse_pbc([token] * 3)
                coordinate_offset = properties["pos"][2]
                positions[atom] = [finite_number(value) for value in fields[coordinate_offset:coordinate_offset + 3]]
                if atom_ids is not None:
                    atom_ids.append(integer(fields[properties["id"][2]]))
                labels.append(fields[properties[label_name][2]] if label_name is not None else None)
            yield Frame(index, step, time, positions, cell, np.zeros(3), pbc,
                        None if atom_ids is None else tuple(atom_ids), tuple(labels),
                        {"format": "extxyz" if extended else "xyz", "coordinates": ["pos"],
                         "label_kind": label_name, "pbc_source": "file" if pbc is not None else "missing",
                         "comment": comment})
            index += 1
            count_line = lines.read(optional=True)

    def summary(self):
        count = 0
        minimum_atoms = maximum_atoms = None
        first_step = last_step = first_time = last_time = None
        all_ids, all_pbc, cells_change = True, True, False
        first_cell = first_pbc = None
        formats = set()
        import numpy as np
        for frame in self.frames():
            atom_count = len(frame.positions)
            minimum_atoms = atom_count if minimum_atoms is None else min(minimum_atoms, atom_count)
            maximum_atoms = atom_count if maximum_atoms is None else max(maximum_atoms, atom_count)
            if count == 0:
                first_step, first_time = frame.step, frame.time
                first_cell, first_pbc = frame.cell, frame.pbc
            elif not np.array_equal(first_cell, frame.cell):
                cells_change = True
            last_step, last_time = frame.step, frame.time
            all_ids = all_ids and frame.atom_ids is not None
            all_pbc = all_pbc and frame.pbc is not None
            formats.add(frame.metadata["format"])
            count += 1
        return {"path": str(self.path), "frames": count, "formats": sorted(formats),
                "atoms_min": minimum_atoms, "atoms_max": maximum_atoms,
                "first_step": first_step, "last_step": last_step,
                "first_time": first_time, "last_time": last_time,
                "all_frames_have_ids": all_ids, "all_frames_have_pbc": all_pbc,
                "first_cell_rows": None if first_cell is None else first_cell.tolist(),
                "first_pbc": first_pbc, "cell_changes": cells_change}
