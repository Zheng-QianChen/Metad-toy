#!/usr/bin/env python3
"""Shared frozen bias: coupled double-well equilibrium probability validation.

The sampler uses Metropolis sweeps, not MD time. Independent quadrature supplies
the reference joint distribution. Eight independent production chains start in
all four sign basins, after a separate training stage and burn-in.
"""
import argparse
import ctypes
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
from metadtoy_cli.reweight import create_reweighted_fes

BINS = 32
LIMIT = 3.0
# Criteria are fixed before running. Bin FES uses cell-integrated reference
# probabilities, and only bins with enough weight ESS; coverage is gated too.
CRITERIA = dict(moment_max_abs=0.035, moment_max_sem=0.010,
                joint_total_variation=0.035, covered_reference_mass=0.98,
                fes_rmse_kbt=0.12, bin_weight_ess=60,
                confidence_multiplier=4.0, confidence_abs_floor=0.003)

def pointer(values):
    return values.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

class Probe:
    def __init__(self, path, cuda_path=None):
        self.library = ctypes.CDLL(str(path.resolve()))
        double_pointer = ctypes.POINTER(ctypes.c_double)
        self.library.shared_probability_error.restype = ctypes.c_char_p
        self.library.shared_probability_prepare.argtypes = [
            ctypes.c_char_p, ctypes.c_int, ctypes.c_ulonglong, ctypes.c_double, double_pointer]
        self.library.shared_probability_evaluate.argtypes = [
            ctypes.c_char_p, ctypes.c_int, double_pointer, double_pointer]
        self.library.shared_probability_sample.argtypes = [
            ctypes.c_char_p, ctypes.c_double, ctypes.c_double, ctypes.c_ulonglong,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, double_pointer, double_pointer]
        self.cuda = ctypes.CDLL(str(cuda_path.resolve())) if cuda_path else None
        if self.cuda:
            self.cuda.shared_probability_cuda_error.restype = ctypes.c_char_p
            self.cuda.shared_probability_cuda_evaluate.argtypes = [
                ctypes.c_char_p, ctypes.c_int, double_pointer, double_pointer, double_pointer]

    def call(self, name, *args):
        if getattr(self.library, name)(*args):
            raise RuntimeError(self.library.shared_probability_error().decode())

    def prepare(self, prefix, kind, seed, strength):
        diagnostics = np.zeros(5)
        self.call("shared_probability_prepare", str(prefix).encode(), kind, seed, strength, pointer(diagnostics))
        original = prefix.with_suffix(".table").read_bytes()
        restored = prefix.with_suffix(".reloaded.table").read_bytes()
        assert original == restored, "table restart is not bitwise identical"
        assert diagnostics[2] < 2e-7 and diagnostics[3] < 5e-5 and diagnostics[4] < 1e-6, diagnostics
        return dict(depositions=int(diagnostics[0]), history_bitwise=True, table_bitwise=True,
                    interpolation_energy_max=float(diagnostics[2]),
                    interpolation_gradient_max=float(diagnostics[3]),
                    finite_difference_max=float(diagnostics[4]),
                    sha256=hashlib.sha256(original).hexdigest())

    def evaluate(self, table, coordinates):
        coordinates = np.ascontiguousarray(coordinates, dtype=np.float64).reshape(-1)
        result = np.empty((len(coordinates), 2))
        self.call("shared_probability_evaluate", str(table).encode(), len(coordinates),
                  pointer(coordinates), pointer(result))
        return result

    def sample(self, table, coupling, scale, seed, basin, count):
        result = np.empty((count, 5))
        acceptance = np.zeros(1)
        self.call("shared_probability_sample", str(table).encode(), coupling, scale,
                  seed, basin, 20000, count, 8, pointer(result), pointer(acceptance))
        return result, float(acceptance[0])

    def check_cuda(self, table):
        if not self.cuda:
            return {"status": "not_run"}
        rng = np.random.default_rng(817)
        pairs = rng.uniform(-3.4, 3.4, (256, 2))
        # Endpoints, adjacent cells, identical centers and opposite wells.
        pairs[:6] = [[-3, 3], [-3.01, 3.01], [0, 0], [-1, 1],
                     [-3 + 1e-8, 3 - 1e-8], [0.0078125, -0.0078125]]
        energies = np.empty(len(pairs))
        gradients = np.empty_like(pairs)
        def evaluate(points):
            points = np.ascontiguousarray(points)
            if self.cuda.shared_probability_cuda_evaluate(
                    str(table).encode(), len(points), pointer(points), pointer(energies), pointer(gradients)):
                raise RuntimeError(self.cuda.shared_probability_cuda_error().decode())
            return energies.copy(), gradients.copy()
        measured_energy, measured_gradient = evaluate(pairs)
        host = self.evaluate(table, pairs).reshape(-1, 2, 2)
        energy_error = float(np.max(abs(measured_energy - host[:, :, 0].mean(axis=1))))
        gradient_error = float(np.max(abs(measured_gradient - host[:, :, 1] / 2)))
        swapped_energy, swapped_gradient = evaluate(pairs[:, ::-1])
        permutation_error = max(float(np.max(abs(swapped_energy - measured_energy))),
                                float(np.max(abs(swapped_gradient[:, ::-1] - measured_gradient))))
        delta = 1e-5
        plus, minus = pairs.copy(), pairs.copy()
        plus[:, 0] += delta
        minus[:, 0] -= delta
        finite_difference = (evaluate(plus)[0] - evaluate(minus)[0]) / (2 * delta)
        fd_error = float(np.max(abs(finite_difference - measured_gradient[:, 0])))
        assert energy_error < 2e-12 and gradient_error < 2e-11
        assert permutation_error < 2e-12 and fd_error < 1e-6
        return dict(status="passed", pairs=len(pairs), energy_max=energy_error,
                    gradient_max=gradient_error, permutation_max=permutation_error,
                    finite_difference_max=fd_error)

def observables(x, y):
    """Include pair structure: single-site marginals alone cannot test coupling."""
    return np.stack(np.broadcast_arrays(x > 0, y > 0, x*y > 0, x*y, x*x, y*y), axis=-1).astype(float)

def quadrature(coupling, subdivisions, limit=LIMIT):
    # Gauss-Legendre integration within each histogram cell avoids comparing
    # cell probabilities to pointwise Boltzmann densities.
    nodes, weights = np.polynomial.legendre.leggauss(subdivisions)
    width = 2 * limit / BINS
    centers = (np.arange(BINS) + 0.5) * width - limit
    axis = (centers[:, None] + width * nodes[None, :] / 2).reshape(-1)
    measure = np.tile(weights * width / 2, BINS)
    x, y = axis[:, None], axis[None, :]
    potential = 4*((x*x-1)**2 + (y*y-1)**2) + 0.5*coupling*(x-y)**2
    probability = np.exp(-potential) * measure[:, None] * measure[None, :]
    partition = probability.sum()
    probability /= probability.sum()
    moments = np.einsum("ij,ijk->k", probability, observables(x, y))
    histogram = probability.reshape(BINS, subdivisions, BINS, subdivisions).sum(axis=(1, 3))
    return dict(probability=histogram, moments=moments, partition=float(partition))

def histogram(samples, log_weights):
    weights = np.exp(log_weights - np.max(log_weights))
    kwargs = dict(bins=BINS, range=[[-LIMIT, LIMIT], [-LIMIT, LIMIT]])
    mass = np.histogram2d(samples[:, 0], samples[:, 1], weights=weights, **kwargs)[0]
    square = np.histogram2d(samples[:, 0], samples[:, 1], weights=weights*weights, **kwargs)[0]
    counts = np.histogram2d(samples[:, 0], samples[:, 1], **kwargs)[0]
    if counts.sum() != len(samples):
        raise RuntimeError("sample outside reference histogram; expand reference bounds")
    ess = np.divide(mass*mass, square, out=np.zeros_like(mass), where=square > 0)
    return mass/mass.sum(), ess, weights

def estimates(samples, log_weights, reference):
    probabilities, bin_ess, weights = histogram(samples, log_weights)
    moments = np.average(observables(samples[:, 0], samples[:, 1]), axis=0, weights=weights)
    covered = bin_ess >= CRITERIA["bin_weight_ess"]
    reference_mass = float(reference["probability"][covered].sum())
    residual = -np.log(probabilities[covered]) + np.log(reference["probability"][covered])
    reference_weights = reference["probability"][covered] / reference_mass
    residual -= np.sum(reference_weights * residual)
    rmse = float(np.sqrt(np.sum(reference_weights * residual**2)))
    return dict(moments=moments, probability=probabilities,
                total_variation=float(np.abs(probabilities-reference["probability"]).sum()/2),
                fes_rmse_kbt=rmse, covered_reference_mass=reference_mass,
                weight_ess=float(weights.sum()**2 / (weights @ weights)))

def compact(estimate):
    return {key: value.tolist() if isinstance(value, np.ndarray) else value
            for key, value in estimate.items() if key != "probability"}

def cli_check(samples, directory, reference, force=False):
    source = directory / "frames.csv"
    # Keep a complete independent production chain. Integer sweep index is
    # deliberately named sample: these rows do not represent physical MD time.
    np.savetxt(source, samples[:, :3], delimiter=",", header="x,y,bias", comments="", fmt="%.17g")
    prefix = directory / "fes"
    create_reweighted_fes(source, prefix, cv_columns=["x", "y"], axes=[1, 2],
                          bins=(BINS, BINS), bounds=[(1, -LIMIT, LIMIT), (2, -LIMIT, LIMIT)],
                          thermal_energy=1.0, weight_source="static-bias", bias_column="bias",
                          energy_unit="kBT", force=force)
    exported = np.genfromtxt(prefix.with_suffix(".csv"), delimiter=",", names=True)
    cell_area = (2*LIMIT/BINS)**2
    expected = estimates(samples, samples[:, 2], reference)["probability"]
    error = float(np.max(abs(exported["probability_density"].reshape(BINS, BINS)*cell_area-expected)))
    assert error < 2e-13, error
    return {"density_mass_max_error": error, "frames": len(samples)}

def make_plot(records, references, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    chosen = [r for r in records if r["name"] in ("fixed_strong_k2_sum", "learned_17_k2_mean")]
    figure, axes = plt.subplots(len(chosen), 4, figsize=(13, 3.4*len(chosen)), constrained_layout=True, squeeze=False)
    reference = references[2.0]["probability"]
    for row, record in enumerate(chosen):
        for column, (label, probability) in enumerate([
                ("Reference", reference), ("Raw biased samples", record["arrays"]["raw"]),
                ("Full-bias reweighted", record["arrays"]["correct"]),
                ("Only own bias (wrong)", record["arrays"]["partial"])]):
            fes = np.full_like(probability, np.nan)
            populated = probability > 0
            fes[populated] = -np.log(probability[populated])
            fes -= np.nanmin(fes)
            colors = plt.get_cmap("viridis").copy()
            colors.set_bad("#dedede")  # Unobserved bins have no estimated FES.
            artist = axes[row, column].imshow(fes.T, origin="lower", extent=(-LIMIT,LIMIT,-LIMIT,LIMIT),
                                             vmin=0, vmax=8, cmap=colors, aspect="equal")
            axes[row, column].set_xlim(-1.8, 1.8)
            axes[row, column].set_ylim(-1.8, 1.8)
            axes[row, column].set_title(label)
            axes[row, column].set_xlabel("x")
            axes[row, column].set_ylabel(("Strong SUM" if record["name"].startswith("fixed") else "Learned MEAN")+"\ny" if column == 0 else "y")
    figure.colorbar(artist, ax=axes, label="Bin FES / kBT")
    figure.savefig(output / "probability_comparison.png", dpi=160)
    plt.close(figure)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--cuda-library", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=60000, help="retained samples per independent chain")
    parser.add_argument("--replicates", type=int, default=8)
    parser.add_argument("--reuse-samples", action="store_true",
                        help="reanalyze existing samples; verify their bias energies against the restored tables")
    parser.add_argument("--strong-only", action="store_true",
                        help="extend the two strong SUM cases with unchanged statistical criteria")
    args = parser.parse_args()
    if args.samples < 10000 or args.replicates < 4:
        parser.error("use >=10000 samples and >=4 independent chains")
    args.output.mkdir(parents=True, exist_ok=args.reuse_samples)
    started = time.time()
    probe = Probe(args.library, args.cuda_library)
    tables, table_records = {}, {}
    for name, kind, seed, strength in [
            ("zero",0,0,0), ("fixed_weak",1,0,0.5), ("fixed_strong",1,0,1.5),
            ("learned_17",2,17,1), ("learned_28",2,28,1)]:
        prefix = args.output / name
        old_digest = (hashlib.sha256(prefix.with_suffix(".table").read_bytes()).hexdigest()
                      if args.reuse_samples else None)
        diagnostics = probe.prepare(prefix, kind, seed, strength)
        if old_digest is not None and old_digest != diagnostics["sha256"]:
            raise RuntimeError("restored table differs from table used for existing samples")
        table = prefix.with_suffix(".table")
        diagnostics["cuda"] = probe.check_cuda(table)
        tables[name] = table
        table_records[name] = diagnostics
        print(json.dumps({"table":name, **diagnostics}), flush=True)
    # Same serialized table must produce the same complete chain for the same
    # seed and initial state, including rejection decisions.
    before, _ = probe.sample(tables["learned_17"], 2, 0.5, 481, 2, 1000)
    after, _ = probe.sample(args.output/"learned_17.reloaded.table", 2, 0.5, 481, 2, 1000)
    assert np.array_equal(before, after), "restarted trajectory differs"

    references, quadrature_records = {}, {}
    for coupling in (0.0, 2.0):
        coarse, fine = quadrature(coupling, 8), quadrature(coupling, 16)
        wider = quadrature(coupling, 16, 4.0)
        error = max(float(np.max(abs(coarse["probability"]-fine["probability"]))),
                    float(np.max(abs(coarse["moments"]-fine["moments"]))))
        tail_error = max(abs(wider["partition"]/fine["partition"]-1),
                         float(np.max(abs(wider["moments"]-fine["moments"]))))
        assert error < 1e-8 and tail_error < 1e-8, (error, tail_error)
        references[coupling] = fine
        quadrature_records[str(coupling)] = dict(grid_error=error, domain_error=tail_error,
                                                moments=fine["moments"].tolist())
    cases = []
    for coupling in (0.0, 2.0):
        cases.append(("zero", coupling, "sum", 1.0))
        for name in ("fixed_weak", "fixed_strong"):
            for mode, scale in (("sum",1.0), ("mean",0.5)):
                cases.append((name, coupling, mode, scale))
    cases.extend((name,2.0,"mean",0.5) for name in ("learned_17", "learned_28"))

    records, failures = [], []
    for case_index, (table_name,coupling,mode,scale) in enumerate(cases):
        name = "%s_k%g_%s" % (table_name,coupling,mode)
        if args.strong_only and not (table_name == "fixed_strong" and mode == "sum"):
            continue
        directory = args.output/name
        directory.mkdir(exist_ok=args.reuse_samples)
        saved_chains = None
        if args.reuse_samples:
            with np.load(directory/"samples.npz") as stored:
                saved_chains = stored["chains"]
            if saved_chains.shape != (args.replicates, args.samples, 5):
                raise ValueError("saved sample dimensions disagree with CLI settings")
        reference = references[coupling]
        chains, acceptances, chain_moments, block_moments = [], [], [], []
        for replicate in range(args.replicates):
            if saved_chains is None:
                samples, acceptance = probe.sample(
                    tables[table_name], coupling, scale, 910001+1000*case_index+replicate,
                    replicate%4, args.samples)
            else:
                samples, acceptance = saved_chains[replicate], None
                expected = probe.evaluate(tables[table_name], samples[:, :2]).reshape(-1, 2, 2)
                if not np.array_equal(samples[:, 3:5], expected[:, :, 0]) or not np.array_equal(
                        samples[:, 2], scale*expected[:, :, 0].sum(axis=1)):
                    raise RuntimeError("saved energies do not match the frozen sampling potential")
            chains.append(samples); acceptances.append(acceptance)
            chain_moments.append(estimates(samples,samples[:,2],reference)["moments"])
            for block in np.array_split(samples, 8):
                weights=np.exp(block[:,2]-block[:,2].max())
                block_moments.append(np.average(observables(block[:,0],block[:,1]),axis=0,weights=weights))
        combined = np.concatenate(chains)
        correct = estimates(combined,combined[:,2],reference)
        raw = estimates(combined,np.zeros(len(combined)),reference)
        partial = estimates(combined,scale*combined[:,3],reference)
        # SE from independent chains is the primary uncertainty. Block SE is a
        # diagnostic; neither assumes that distinct atoms are independent.
        independent_sem = np.std(chain_moments,axis=0,ddof=1)/np.sqrt(args.replicates)
        block_sem = np.std(block_moments,axis=0,ddof=1)/np.sqrt(len(block_moments))
        error = abs(correct["moments"]-reference["moments"])
        half = np.concatenate([chain[:len(chain)//2] for chain in chains])
        half_estimate = estimates(half,half[:,2],reference)
        checks = dict(
            moment_accuracy=bool(np.max(error)<CRITERIA["moment_max_abs"]),
            precision=bool(np.max(independent_sem)<CRITERIA["moment_max_sem"]),
            confidence=bool(np.all(error < CRITERIA["confidence_multiplier"]*independent_sem+CRITERIA["confidence_abs_floor"])),
            joint_distribution=correct["total_variation"]<CRITERIA["joint_total_variation"],
            fes=correct["fes_rmse_kbt"]<CRITERIA["fes_rmse_kbt"],
            coverage=correct["covered_reference_mass"]>CRITERIA["covered_reference_mass"])
        if not all(checks.values()):
            failures.append(name)
        if not args.reuse_samples:
            np.savez_compressed(directory/"samples.npz", chains=np.stack(chains))
        np.savez_compressed(directory/"histograms.npz", reference=reference["probability"],
                            correct=correct["probability"], raw=raw["probability"], partial=partial["probability"])
        record=dict(name=name, coupling=coupling, aggregation=mode, scale=scale,
                    correct=compact(correct), raw=compact(raw), partial=compact(partial),
                    independent_sem=independent_sem.tolist(), block_sem=block_sem.tolist(),
                    half_length=compact(half_estimate), acceptance=acceptances,
                    checks=checks, passed=all(checks.values()),
                    arrays={key:value["probability"] for key,value in
                            (("correct",correct),("raw",raw),("partial",partial))})
        if name in ("fixed_strong_k2_sum","learned_17_k2_mean"):
            record["cli"] = cli_check(chains[0],directory,reference,force=args.reuse_samples)
        records.append(record)
        print(json.dumps({k:v for k,v in record.items() if k not in ("arrays","raw","partial","acceptance","half_length")}),flush=True)

    selected = next(r for r in records if r["name"]=="fixed_strong_k2_sum")
    # Negative controls target the marginal of x: own-bias weighting can be
    # valid when x,y are independent, but is wrong once y's bias couples to x.
    reference_x = float(references[2.0]["moments"][0])
    negative_checks = dict(raw_detected=abs(selected["raw"]["moments"][0]-reference_x)>0.05,
                           own_only_detected=abs(selected["partial"]["moments"][0]-reference_x)>0.05)
    if not all(negative_checks.values()):
        failures.append("negative_controls")
    make_plot(records,references,args.output)
    for record in records:
        record.pop("arrays")
    manifest=dict(schema_version=1, passed=not failures, failures=failures, criteria=CRITERIA,
        scope="Steps 1-2 only: equilibrium Metropolis sampling with fixed shared bias; no kinetic or adaptive reweighting claim.",
        model="beta U=4*((x*x-1)^2+(y*y-1)^2)+coupling/2*(x-y)^2; kBT=1",
        sampling=dict(replicates=args.replicates,samples_per_chain=args.samples,burn_sweeps=20000,
                      stride=8, initial_basins="all four sign combinations",
                      seed_rule="910001+1000*case_index+replicate"),
        training=dict(coupling=2.0,aggregation="mean",sweeps=24000,pace=200,
                      height_per_center=0.025,sigma=0.45,seeds=[17,28],
                      production_uses_training_samples=False),
        history_and_tables=table_records,quadrature=quadrature_records,
        restarted_chain_bitwise=True,negative_controls=negative_checks,cases=records,
        runtime=dict(seconds=time.time()-started,reused_samples=args.reuse_samples,
                     python=sys.version,numpy=np.__version__,host=platform.node()),
        sources={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest()
                 for path in (Path(__file__),Path(__file__).with_name("probability_probe.cpp"),
                              ROOT/"src/sampling/zqc_hill_bias.cpp",ROOT/"src/sampling/zqc_shared_bias.cu",
                              ROOT/"tools/metadtoy_cli/reweight.py")},
        binaries={str(path):hashlib.sha256(path.read_bytes()).hexdigest()
                  for path in (args.library,args.cuda_library) if path})
    (args.output/"summary.json").write_text(json.dumps(manifest,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"passed":not failures,"failures":failures,"seconds":manifest["runtime"]["seconds"]}),flush=True)
    return int(bool(failures))

if __name__=="__main__":
    raise SystemExit(main())
