#!/usr/bin/env python3
"""Reconstruct FES from recorded frames, then independently evaluate accuracy.

Run 'reconstruct' first. That stage reads only recorded CVs, total bias and
sampling metadata. The model potential occurs only in reference_distribution(),
called by the subsequent 'evaluate' stage.
"""
import argparse
import json
import sys
from pathlib import Path
import numpy as np

BINS = 32
LIMIT = 3.
MOMENT_NAMES = ["P(x>0)", "P(y>0)", "P(x*y>0)", "<xy>", "<x^2>", "<y^2>"]

def observable_values(samples):
    x, y = samples[:, 0], samples[:, 1]
    return np.column_stack([x>0, y>0, x*y>0, x*y, x*x, y*y])

def collect(chains, use_bias):
    """Store unnormalized per-chain sums using one common log-weight offset."""
    shift = max(float(row[:, 2].max()) for row in chains) if use_bias else 0.
    masses, squares, moments, halves = [], [], [], []
    for samples in chains:
        if np.any(abs(samples[:, :2]) > LIMIT):
            raise RuntimeError("histogram bounds must cover every recorded frame")
        weights = np.exp(samples[:, 2]-shift) if use_bias else np.ones(len(samples))
        kwargs = dict(bins=BINS, range=[[-LIMIT, LIMIT], [-LIMIT, LIMIT]])
        masses.append(np.histogram2d(*samples[:, :2].T, weights=weights, **kwargs)[0])
        squares.append(np.histogram2d(*samples[:, :2].T, weights=weights**2, **kwargs)[0])
        moments.append(weights @ observable_values(samples))
        midpoint = len(samples)//2
        halves.append(np.stack([np.histogram2d(*part[:, :2].T, weights=part_weights, **kwargs)[0]
            for part, part_weights in [(samples[:midpoint], weights[:midpoint]),
                                       (samples[midpoint:], weights[midpoint:])]]))
    masses, squares = np.asarray(masses), np.asarray(squares)
    total = masses.sum(axis=0)
    probability = total/total.sum()
    bin_ess = np.divide(total**2, squares.sum(axis=0),
                        out=np.zeros_like(total), where=squares.sum(axis=0)>0)
    return dict(probability=probability, bin_ess=bin_ess, chain_mass=masses,
                chain_square=squares, chain_moments=np.asarray(moments),
                half_mass=np.asarray(halves), log_weight_offset=shift)

def reconstruct(directory):
    protocol = json.loads((directory/"protocol.json").read_text())
    for group in ["frozen", "unbiased"]:
        paths = sorted((directory/group).glob("replica_*.npz"))
        assert len(paths) == protocol["replicas"]
        loaded = [np.load(path) for path in paths]
        chains = [entry["samples"] for entry in loaded]
        diagnostics = np.stack([entry["diagnostics"] for entry in loaded])
        arrays = {"diagnostics": diagnostics}
        for estimator, use_bias in [("raw", False), ("weighted", True)]:
            for key, value in collect(chains, use_bias).items():
                arrays[estimator+"_"+key] = value
        np.savez(directory/group/"reconstruction.npz", **arrays)
        centers = (np.arange(BINS)+.5)*2*LIMIT/BINS-LIMIT
        x, y = np.meshgrid(centers, centers, indexing="ij")
        for estimator in ["raw", "weighted"]:
            probability = arrays[estimator+"_probability"]
            fes = np.full_like(probability, np.nan)
            mask = probability>0
            fes[mask] = -np.log(probability[mask])
            fes[mask] -= fes[mask].min()
            np.savetxt(directory/group/f"{estimator}_fes.csv",
                np.column_stack([x.ravel(), y.ravel(), probability.ravel(), fes.ravel(),
                                 arrays[estimator+"_bin_ess"].ravel()]),
                delimiter=",", header="x,y,cell_probability,F_over_kBT,weight_ESS", comments="")
        if group == "frozen":
            # Verify the user-facing CLI against the independent reconstruction
            # on the complete production dataset, before opening any reference.
            sys.path.insert(0, str(Path(__file__).resolve().parents[2]/"tools"))
            from metadtoy_cli.reweight import create_reweighted_fes
            source = directory/"production.csv"
            with source.open("w") as file:
                file.write("x,y,bias\n")
                for samples in chains:
                    np.savetxt(file, samples[:, :3], delimiter=",", fmt="%.17g")
            create_reweighted_fes(source, directory/"cli_fes", cv_columns=["x", "y"],
                axes=[1, 2], bins=(BINS, BINS),
                bounds=[(1, -LIMIT, LIMIT), (2, -LIMIT, LIMIT)],
                thermal_energy=1., weight_source="static-bias", bias_column="bias",
                energy_unit="kBT", force=True)
            exported = np.genfromtxt(directory/"cli_fes.csv", delimiter=",", names=True)
            difference = float(np.max(abs(exported["probability_density"].reshape(BINS, BINS)*
                               (2*LIMIT/BINS)**2-arrays["weighted_probability"])))
            assert difference < 2e-12, difference
            (directory/"reconstruction_checks.json").write_text(json.dumps(dict(
                cli_probability_max_error=difference, frames=sum(len(row) for row in chains),
                reference_used=False, weight_formula="exp(B/kBT), full frame bias B"), indent=2)+"\n")
        print("RECONSTRUCTED", group, flush=True)

def reference_distribution(order):
    """Evaluation only: integrate Boltzmann mass inside the same cells."""
    nodes, weights = np.polynomial.legendre.leggauss(order)
    width = 2*LIMIT/BINS
    centers = (np.arange(BINS)+.5)*width-LIMIT
    axis = (centers[:, None]+width*nodes/2).ravel()
    measure = np.tile(weights*width/2, BINS)
    x, y = axis[:, None], axis[None, :]
    potential = 4*((x*x-1)**2+(y*y-1)**2)+(x-y)**2
    weighted_density = np.exp(-potential)*measure[:, None]*measure[None, :]
    partition = weighted_density.sum()
    weighted_density /= partition
    moments = np.array([(weighted_density*value).sum() for value in
        [x>0, y>0, x*y>0, x*y, x*x, y*y]])
    mass = weighted_density.reshape(BINS, order, BINS, order).sum(axis=(1, 3))
    return mass, moments, partition

def fes_error(probability, reference, support):
    mass = reference[support]
    residual = -np.log(probability[support])+np.log(reference[support])
    residual -= np.average(residual, weights=mass)
    return float(np.sqrt(np.average(residual**2, weights=mass)))

def normalized_fes(probability):
    result = np.full_like(probability, np.nan)
    mask = probability>0
    result[mask] = -np.log(probability[mask])
    result[mask] -= np.min(result[mask])
    return result

def evaluate(directory):
    protocol = json.loads((directory/"protocol.json").read_text())
    criteria = protocol["criteria"]
    reference, reference_moments, partition = reference_distribution(16)
    coarse, coarse_moments, coarse_partition = reference_distribution(8)
    quadrature_error = float(np.max(abs(reference-coarse)))
    assert quadrature_error < 1e-10
    assert np.max(abs(reference_moments-coarse_moments)) < 1e-10
    records, plot_data = {}, {}
    random = np.random.default_rng(411902)
    replica_count = protocol["replicas"]
    frequencies = random.multinomial(replica_count, np.ones(replica_count)/replica_count, size=2000)
    for group, estimator in [("frozen", "raw"), ("frozen", "weighted"), ("unbiased", "raw")]:
        name = group+"_"+estimator
        with np.load(directory/group/"reconstruction.npz") as arrays:
            probability = arrays[estimator+"_probability"]
            ess = arrays[estimator+"_bin_ess"]
            chain_mass = arrays[estimator+"_chain_mass"]
            chain_square = arrays[estimator+"_chain_square"]
            chain_moments = arrays[estimator+"_chain_moments"]
            half_mass = arrays[estimator+"_half_mass"]
            diagnostics = arrays["diagnostics"]
        chain_weight = chain_mass.sum(axis=(1, 2))
        pooled_moments = chain_moments.sum(axis=0)/chain_weight.sum()
        # Cluster delta-method SEM accounts for correlation within each path.
        influence = (chain_moments-chain_weight[:, None]*pooled_moments)/chain_weight.mean()
        moment_sem = influence.std(axis=0, ddof=1)/np.sqrt(replica_count)
        bootstrap_mass = (frequencies @ chain_mass.reshape(replica_count, -1)).reshape(-1, BINS, BINS)
        bootstrap_mass /= bootstrap_mass.sum(axis=(1, 2), keepdims=True)
        bootstrap_moments = (frequencies @ chain_moments)/(frequencies @ chain_weight)[:, None]
        moment_ci = np.quantile(bootstrap_moments, [.025, .975], axis=0).T
        support = ess >= criteria["bin_weight_ess"]
        coverage = float(reference[support].sum())
        rmse = fes_error(probability, reference, support)
        temperature = diagnostics[:, 6]
        temperature_sem = float(temperature.std(ddof=1)/np.sqrt(replica_count))
        moments_error = abs(pooled_moments-reference_moments)
        checks = dict(fes_rmse=rmse<criteria["fes_rmse_kbt"],
            coverage=coverage>=criteria["covered_reference_mass"],
            total_variation=float(abs(probability-reference).sum()/2)<criteria["total_variation"],
            moment_accuracy=float(moments_error.max())<criteria["moment_max_abs"],
            moment_precision=float(moment_sem.max())<criteria["moment_max_sem"],
            moment_consistency=bool(np.all(moments_error < criteria["confidence_multiplier"]*
                                        moment_sem+criteria["confidence_abs_floor"])),
            temperature=abs(float(temperature.mean())-1)<4*temperature_sem+.003)
        halves = half_mass.sum(axis=0)
        halves /= halves.sum(axis=(1, 2), keepdims=True)
        marginal_rmse = []
        for axis in (0, 1):
            measured_marginal = probability.sum(axis=1-axis)
            reference_marginal = reference.sum(axis=1-axis)
            marginal_squares = chain_square.sum(axis=(0, 2 if axis==0 else 1))
            marginal_sum = chain_mass.sum(axis=(0, 2 if axis==0 else 1))
            marginal_ess = np.divide(marginal_sum**2, marginal_squares,
                out=np.zeros_like(marginal_sum), where=marginal_squares>0)
            marginal_rmse.append(fes_error(measured_marginal, reference_marginal,
                                           marginal_ess>=criteria["bin_weight_ess"]))
        records[name] = dict(fes_rmse_kbt=rmse, marginal_fes_rmse_kbt=marginal_rmse,
            covered_reference_mass=coverage,
            total_variation=float(abs(probability-reference).sum()/2),
            moments=pooled_moments.tolist(), moment_sem=moment_sem.tolist(),
            moment_95ci=moment_ci.tolist(), moment_max_abs=float(moments_error.max()),
            weight_ess=float(chain_weight.sum()**2/chain_square.sum()),
            transitions=int(diagnostics[:, 0].sum()),
            kinetic_temperature=float(temperature.mean()), temperature_sem=temperature_sem,
            first_half_P_x_positive=float(halves[0, BINS//2:, :].sum()),
            second_half_P_x_positive=float(halves[1, BINS//2:, :].sum()),
            half_distributions_TV=float(abs(halves[0]-halves[1]).sum()/2),
            criteria_pass=checks, passed=all(checks.values()))
        plot_data[name] = dict(probability=probability, bootstrap=bootstrap_mass, support=support)
        print(name, json.dumps(records[name]), flush=True)
    summary = dict(protocol=protocol, criteria=criteria,
        reference=dict(quadrature_mass_max_error=quadrature_error,
            partition=partition, moments=dict(zip(MOMENT_NAMES, reference_moments.tolist()))),
        reconstruction=json.loads((directory/"reconstruction_checks.json").read_text()),
        records=records, uncertainty="independent-trajectory cluster bootstrap; weight ESS excludes time autocorrelation")
    (directory/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    make_plots(directory, reference, records, plot_data)
    # Persist the independent answer only after reconstruction has completed.
    np.savez(directory/"reference_evaluation_only.npz", probability=reference, moments=reference_moments)
    if not records["frozen_weighted"]["passed"]:
        raise SystemExit("Frozen-bias FES recovery failed its predeclared statistical criteria")

def make_plots(directory, reference, records, results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    centers = (np.arange(BINS)+.5)*2*LIMIT/BINS-LIMIT
    extent = [-LIMIT, LIMIT, -LIMIT, LIMIT]
    figure, axes = plt.subplots(1, 4, figsize=(15, 3.9), constrained_layout=True)
    items = [("Reference", reference, None),
             ("Raw biased frames", results["frozen_raw"]["probability"], results["frozen_raw"]["support"]),
             ("Full-bias reweighted", results["frozen_weighted"]["probability"], results["frozen_weighted"]["support"])]
    for axis, (title, probability, support) in zip(axes, items):
        fes = normalized_fes(probability)
        fes[fes>10] = np.nan
        if support is not None:
            fes[~support] = np.nan
        color = axis.imshow(fes.T, origin="lower", extent=extent, cmap="viridis", vmin=0, vmax=10)
        axis.set(title=title, xlabel="x", ylabel="y", xlim=(-1.8, 1.8), ylim=(-1.8, 1.8))
    figure.colorbar(color, ax=list(axes[:3]), label="F / kBT", shrink=.75)
    weighted = results["frozen_weighted"]
    mask = weighted["support"]
    error = np.full_like(reference, np.nan)
    error[mask] = -np.log(weighted["probability"][mask])+np.log(reference[mask])
    error[mask] -= np.average(error[mask], weights=reference[mask])
    residual_image = axes[3].imshow(error.T, origin="lower", extent=extent, cmap="coolwarm",
                                   vmin=-.25, vmax=.25)
    axes[3].set(title="Recovery error", xlabel="x", ylabel="y", xlim=(-1.8, 1.8), ylim=(-1.8, 1.8))
    figure.colorbar(residual_image, ax=axes[3], label="Delta F / kBT", shrink=.75)
    figure.suptitle("Held-out Langevin MD: shared bias deposited, frozen, equilibrated, then reweighted\n"
        f"2D weighted RMSE = {records['frozen_weighted']['fes_rmse_kbt']:.4f} kBT; "
        f"reference mass covered = {100*records['frozen_weighted']['covered_reference_mass']:.2f}%")
    figure.savefig(directory/"fes_recovery_2d.png", dpi=180)
    plt.close(figure)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)
    for component, axis in enumerate(axes):
        axis.plot(centers, normalized_fes(reference.sum(axis=1-component)),
                  color="black", label="Reference", linewidth=2.2)
        for name, label, color in [("frozen_raw", "Raw biased", "#c96738"),
                ("frozen_weighted", "Full-bias reweighted", "#167c99"),
                ("unbiased_raw", "Unbiased MD control", "#71945a")]:
            probability = results[name]["probability"].sum(axis=1-component)
            axis.plot(centers, normalized_fes(probability), marker=".", markersize=3,
                      color=color, label=label, alpha=.9)
            if name=="frozen_weighted":
                bootstrap = results[name]["bootstrap"].sum(axis=2-component)
                curves = np.stack([normalized_fes(row) for row in bootstrap])
                # Every bootstrap must populate a displayed bin to form its CI.
                valid = np.isfinite(curves).all(axis=0)
                lower, upper = np.full(BINS, np.nan), np.full(BINS, np.nan)
                lower[valid], upper[valid] = np.quantile(curves[:, valid], [.025, .975], axis=0)
                axis.fill_between(centers, lower, upper, color=color, alpha=.22,
                                  label="95% chain-bootstrap interval")
        axis.set(xlabel=["x", "y"][component], ylabel="F / kBT",
                 title=["Marginal F(x)", "Marginal F(y)"][component],
                 xlim=(-1.6, 1.6), ylim=(-.2, 7))
        axis.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    figure.suptitle("Reconstruction uses CVs + full frame bias; reference is evaluation only")
    figure.savefig(directory/"fes_recovery_1d.png", dpi=180)
    plt.close(figure)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["reconstruct", "evaluate"])
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    {"reconstruct": reconstruct, "evaluate": evaluate}[args.stage](args.directory.resolve())

if __name__ == "__main__":
    main()
