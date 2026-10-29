#!/usr/bin/env python3
"""Generate held-out fixed-duration Langevin trajectories for FES recovery.

The CV is the physical coordinate. A fresh shared table is deposited by MD,
frozen, and then sampled with explicit burn-in from four starting quadrants.
No reference marginal, CDF, or FES enters this program.
"""
import argparse
import concurrent.futures
import ctypes
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from kinetics_matched import MatchedBridge
from probability import pointer

class FESBridge(MatchedBridge):
    def __init__(self, library, placeholder):
        super().__init__(library, placeholder)
        pointer_type = ctypes.POINTER(ctypes.c_double)
        self.library.kinetics_fes_sample.argtypes = [
            ctypes.c_char_p, ctypes.c_int, ctypes.c_double, ctypes.c_double,
            ctypes.c_double, ctypes.c_ulonglong, ctypes.c_int, ctypes.c_int,
            pointer_type, pointer_type]

    def sample(self, table, biased, step, burn, duration, seed, quadrant):
        rows = round(duration / .1)
        samples = np.empty((rows, 6), dtype=np.float64)
        diagnostics = np.empty(8)
        self.call("kinetics_fes_sample", str(table).encode(), biased, step, burn,
                  duration, seed, quadrant, rows, pointer(samples), pointer(diagnostics))
        assert int(diagnostics[7]) == rows
        assert np.isfinite(samples).all()
        return samples, diagnostics

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replicas", type=int, default=16)
    parser.add_argument("--duration", type=float, default=10000.)
    parser.add_argument("--burn", type=float, default=500.)
    parser.add_argument("--step", type=float, default=.0005)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    assert args.replicas >= 8 and args.replicas % 4 == 0
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    # The legacy deposition ABI accepts a mapping file. Physical-coordinate
    # mode ignores it; a zero table supplies the format without reference data.
    empty = output / "zero.table"
    grid = np.linspace(-3., 3., 4097)
    np.savetxt(empty, np.column_stack([grid, 0*grid, 0*grid]),
               header="METAD_TOY_SHARED_FROZEN 1", fmt="%.17g")
    bridge = FESBridge(args.library, empty)
    prefix = output / "deposited"
    start = time.perf_counter()
    deposition = bridge.train(empty, 4, args.step, 200., 187210, 0, prefix)
    table = prefix.with_suffix(".table")
    checksum = hashlib.sha256(table.read_bytes()).hexdigest()
    force_error = bridge.force_check(table, 3)
    assert deposition[0] == 800 and deposition[5] < 1e-6 and deposition[6] < .01
    # Burn-in is an exact continuation of the same random path.
    short, _ = bridge.sample(table, 1, args.step, .5, 1., 12451, 2)
    complete, _ = bridge.sample(table, 1, args.step, 0., 1.5, 12451, 2)
    assert np.array_equal(short, complete[5:])
    protocol = dict(model="coupled quartic double well; kBT=mass=friction=1",
        physical_cv=["x", "y"], total_bias="(u(x)+u(y))/2",
        deposition_seed=187210, deposition_duration=200., height=.02,
        sigma=.12, deposition_pace=.5, deposition_hills=int(deposition[0]),
        deposition_transitions=int(deposition[1]),
        table_energy_error=float(deposition[5]), table_gradient_error=float(deposition[6]),
        force_fd_max=force_error, burn_continuation_bitwise=True,
        table_sha256=checksum, step=args.step, burn=args.burn,
        duration=args.duration, replicas=args.replicas, record_interval=.1,
        production_seeds=[931840+i for i in range(args.replicas)],
        quadrants=[i % 4 for i in range(args.replicas)],
        reference_used_by_cv_or_deposition=False,
        criteria=dict(fes_rmse_kbt=.12, covered_reference_mass=.98,
            bin_weight_ess=60, total_variation=.035, moment_max_abs=.035,
            moment_max_sem=.010, confidence_multiplier=4., confidence_abs_floor=.003))
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2)+"\n")
    print("DEPOSITED 800 hills; force and burn-continuation checks passed", flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        for name, biased in [("frozen", 1), ("unbiased", 0)]:
            directory = output / name
            directory.mkdir()
            group_start = time.perf_counter()
            futures = {pool.submit(bridge.sample, table, biased, args.step, args.burn,
                        args.duration, 931840+replica, replica % 4): replica
                       for replica in range(args.replicas)}
            for number, future in enumerate(concurrent.futures.as_completed(futures), 1):
                replica = futures[future]
                samples, diagnostics = future.result()
                np.savez(directory / f"replica_{replica:02d}.npz",
                         samples=samples, diagnostics=diagnostics)
                print(f"{name}: {number}/{args.replicas}; replica {replica}; "
                      f"transitions={int(diagnostics[0])}; T={diagnostics[6]:.5f}", flush=True)
            protocol[name+"_wall_seconds"] = time.perf_counter()-group_start
    assert hashlib.sha256(table.read_bytes()).hexdigest() == checksum
    protocol["frozen_table_unchanged"] = True
    protocol["wall_seconds"] = time.perf_counter()-start
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2)+"\n")
    print("SAMPLING COMPLETE", protocol["wall_seconds"], flush=True)

if __name__ == "__main__":
    main()
