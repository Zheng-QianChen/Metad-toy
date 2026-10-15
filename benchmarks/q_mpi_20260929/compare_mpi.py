#!/usr/bin/env python3
import json
import math
import re
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
NPS = (1, 2, 4)


def read_dump(path: Path):
    lines = path.read_text().splitlines()
    frames = []
    i = 0
    while i < len(lines):
        if lines[i] != "ITEM: TIMESTEP":
            i += 1
            continue
        step = int(lines[i + 1])
        n = int(lines[i + 3])
        header = lines[i + 8].split()[2:]
        data = np.loadtxt(lines[i + 9:i + 9 + n], ndmin=2)
        order = np.argsort(data[:, 0].astype(np.int64))
        frames.append((step, header, data[order]))
        i += 9 + n
    if not frames:
        raise RuntimeError(f"no frames in {path}")
    return frames[-1]


def read_thermo(path: Path):
    lines = path.read_text(errors="replace").splitlines()
    rows = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped.startswith("Step "):
            continue
        names = stripped.split()
        j = i + 1
        while j < len(lines):
            fields = lines[j].split()
            if len(fields) != len(names):
                break
            try:
                values = [float(x) for x in fields]
            except ValueError:
                break
            rows.append(dict(zip(names, values)))
            j += 1
    if not rows:
        raise RuntimeError(f"no thermo rows in {path}")
    return rows[-1]


reference_step, header, reference = read_dump(ROOT / "np1" / "atom_state.dump")
reference_thermo = read_thermo(ROOT / "np1" / "log.lammps")
summary = {
    "reference_step": reference_step,
    "columns": header,
    "ranks": {},
}

for np_rank in NPS:
    step, names, data = read_dump(ROOT / f"np{np_rank}" / "atom_state.dump")
    thermo = read_thermo(ROOT / f"np{np_rank}" / "log.lammps")
    if names != header or data.shape != reference.shape:
        raise RuntimeError(f"shape/header mismatch for np={np_rank}")
    if not np.array_equal(data[:, 0].astype(np.int64), reference[:, 0].astype(np.int64)):
        raise RuntimeError(f"atom IDs mismatch for np={np_rank}")

    per_column = {}
    for c, name in enumerate(header[2:], start=2):
        delta = np.abs(data[:, c] - reference[:, c])
        scale = max(float(np.max(np.abs(reference[:, c]))), 1e-300)
        per_column[name] = {
            "max_abs": float(np.max(delta)),
            "rms": float(np.sqrt(np.mean(delta * delta))),
            "max_rel_to_ref_scale": float(np.max(delta) / scale),
        }

    thermo_delta = {}
    for name, ref_value in reference_thermo.items():
        if name not in thermo:
            continue
        delta = abs(thermo[name] - ref_value)
        thermo_delta[name] = {
            "value": thermo[name],
            "reference": ref_value,
            "abs_delta": delta,
            "rel_delta": delta / max(abs(ref_value), 1e-300),
        }
    summary["ranks"][str(np_rank)] = {
        "step": step,
        "per_atom": per_column,
        "thermo": thermo_delta,
        "force_norms": {
            "l1": [float(np.sum(np.abs(data[:, c]))) for c in (5, 6, 7)],
            "linf": [float(np.max(np.abs(data[:, c]))) for c in (5, 6, 7)],
            "net": [float(np.sum(data[:, c])) for c in (5, 6, 7)],
        },
    }

(ROOT / "comparison.json").write_text(json.dumps(summary, indent=2) + "\n")

important = ["fx", "fy", "fz", "c_pa[1]", "c_pa[2]", "c_pa[3]", "c_pa[4]",
             "c_pa[5]", "c_pa[6]", "c_pa[7]", "c_pa[8]"]
print("np,column,max_abs,max_rel_to_np1_scale")
for np_rank in NPS[1:]:
    cols = summary["ranks"][str(np_rank)]["per_atom"]
    for name in important:
        v = cols[name]
        print(f"{np_rank},{name},{v['max_abs']:.17g},{v['max_rel_to_ref_scale']:.17g}")

print("\nnp,force_l1_x,force_l1_y,force_l1_z,net_fx,net_fy,net_fz")
for np_rank in NPS:
    norms = summary["ranks"][str(np_rank)]["force_norms"]
    values = norms["l1"] + norms["net"]
    print(str(np_rank) + "," + ",".join(f"{v:.17g}" for v in values))
