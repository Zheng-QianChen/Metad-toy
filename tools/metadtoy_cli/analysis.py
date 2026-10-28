"""Offline analysis operations independent of argparse or terminal widgets."""
from contextlib import contextmanager
from pathlib import Path
import csv
import json
import math
import os
import tempfile
from .hills import evaluate_bias


@contextmanager
def output_transaction(paths, source, force=False):
    """Stage products beside their destinations, then publish completed files.

    Each replacement is atomic. A source size/mtime check detects a live HILLS
    writer between parsing and publication. A stopped run or copied snapshot is
    the reproducible input for offline analysis.
    """
    paths = [Path(path).resolve() for path in paths]
    source = Path(source).resolve()
    if len(set(paths)) != len(paths) or source in paths:
        raise ValueError("Output paths must be distinct from each other and the input")
    if len({path.parent for path in paths}) != 1:
        raise ValueError("Output products must share a directory")
    for path in paths:
        if path.exists() and (not force or not path.is_file()):
            raise ValueError("Output exists: %s (use --force to replace a file)" % path)
    source_stat = source.stat()
    paths[0].parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".metadtoy-", dir=str(paths[0].parent)) as directory:
        staged = [Path(directory) / path.name for path in paths]
        yield staged
        current_stat = source.stat()
        if (current_stat.st_size, current_stat.st_mtime_ns) != (source_stat.st_size, source_stat.st_mtime_ns):
            raise ValueError("Input changed during analysis; use a completed HILLS snapshot")
        for path in paths:
            if path.exists() and not force:
                raise ValueError("Output appeared during analysis: %s" % path)
        for temporary, destination in zip(staged, paths):
            os.replace(str(temporary), str(destination))


def export_csv(reader, destination, through_step=None, force=False):
    """Stream raw depositions to CSV and preserve settings in a JSON sidecar."""
    destination = Path(destination)
    if destination.suffix.lower() != ".csv":
        raise ValueError("--csv must end with .csv")
    paths = [destination, destination.with_suffix(".json")]
    dimension = reader.metadata.dimension
    height_count = dimension if reader.metadata.mode == "PBMETAD" else 1
    fields = (["step"] + ["s%d" % (axis + 1) for axis in range(dimension)]
              + ["height%d" % (axis + 1) for axis in range(height_count)]
              + ["sigma%d" % (axis + 1) for axis in range(dimension)])
    with output_transaction(paths, reader.path, force) as staged:
        with staged[0].open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(fields)
            for record in reader.records(through_step):
                writer.writerow([record.step] + [format(value, ".17g")
                                for value in record.centers + record.heights + record.sigma])
        metadata = reader.summary(through_step)
        metadata.update(product="depositions", through_step=through_step,
                        columns=fields, units="unchanged simulation units")
        staged[1].write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return [str(path.resolve()) for path in paths]


def _axis_values(entries, dimension, label, count):
    values = {}
    for entry in entries:
        if len(entry) != count:
            raise ValueError("Invalid %s specification" % label)
        axis = int(entry[0]) - 1
        if axis < 0 or axis >= dimension or axis in values:
            raise ValueError("%s axes must be unique and between 1 and %d" % (label, dimension))
        numeric = tuple(float(value) for value in entry[1:])
        if not all(math.isfinite(value) for value in numeric):
            raise ValueError("%s values must be finite" % label)
        values[axis] = numeric
    return values


def create_preview(reader, prefix, axes=None, bins=(200,), bounds=(), fixed=(),
                   quantity="bias", bias_factor=None, energy_unit="native units",
                   through_step=None, hill_batch=256, point_batch=512,
                   pb_normalization="mean", force=False, method="reweight", colvar=None, reweight_options=None):
    """Export bias surfaces or FES; FES defaults to trajectory reweighting.

    Explicit method=hills estimates WT FES for full-HILLS histories. PB hills yield
    the combined bias, since marginal PB hills alone do not determine a joint FES.
    """
    if method not in ("reweight", "hills"):
        raise ValueError("FES method must be reweight or hills")
    if quantity == "fes" and method == "reweight":
        if fixed:
            raise ValueError("reweighted FES is a marginal histogram; --at slices belong to --method hills")
        if bias_factor is not None:
            raise ValueError("--bias-factor applies to --method hills; offline c(t) reads native HILLS metadata")
        from .reweight import create_reweighted_fes
        samples = Path(colvar) if colvar is not None else reader.path.parent / "COLVAR"
        if not samples.is_file():
            raise ValueError("default FES method reweight needs COLVAR samples (--colvar); use --method hills for the HILLS-only estimate")
        return create_reweighted_fes(samples, prefix, history=reader.path, axes=axes, bins=bins,
                                     bounds=bounds, through_step=through_step, energy_unit=energy_unit,
                                     force=force, **(reweight_options or {}))

    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    if quantity not in ("bias", "fes"):
        raise ValueError("quantity must be bias or fes")
    if hill_batch <= 0 or point_batch <= 0 or hill_batch * point_batch > 4000000:
        raise ValueError("Batch sizes must be positive and their product at most 4000000")
    prefix = Path(prefix)
    paths = [Path(str(prefix) + suffix) for suffix in (".csv", ".png", ".json")]
    dimension = reader.metadata.dimension
    axes = tuple(int(axis) - 1 for axis in (axes or tuple(range(1, min(dimension, 2) + 1))))
    if len(axes) not in (1, 2) or len(set(axes)) != len(axes) or any(axis < 0 or axis >= dimension for axis in axes):
        raise ValueError("--axes selects one or two distinct 1-based axes")
    bins = tuple(int(value) for value in bins)
    if len(bins) == 1:
        bins *= len(axes)
    if len(bins) != len(axes) or any(value < 2 for value in bins):
        raise ValueError("--bins needs one value or one per plotted axis; each must be at least 2")
    point_count = math.prod(bins)
    if point_count > 1000000 or point_count * dimension > 8000000:
        raise ValueError("Preview is too large; reduce --bins (limit: 1M points and 8M coordinates)")
    selected_bounds = _axis_values(bounds, dimension, "range", 3)
    selected_fixed = _axis_values(fixed, dimension, "at", 2)
    if set(selected_bounds) - set(axes):
        raise ValueError("--range applies only to plotted axes")
    if set(selected_fixed) != set(range(dimension)) - set(axes):
        raise ValueError("Specify --at AXIS VALUE for every unplotted axis, and only those axes")
    factor = None
    if quantity == "fes":
        if reader.metadata.mode == "PBMETAD":
            raise ValueError("PBMetaD hills do not determine a joint FES; use --quantity bias for slices, or trajectory reweighting")
        if reader.metadata.version and not reader.metadata.well_tempered:
            raise ValueError("FES estimation currently requires a well-tempered history")
        factor = reader.metadata.bias_factor if reader.metadata.version else bias_factor
        if factor is None or not math.isfinite(factor) or factor <= 1:
            raise ValueError("Legacy WT histories require --bias-factor greater than 1")
        if bias_factor is not None and (not math.isfinite(bias_factor) or not math.isclose(factor, bias_factor)):
            raise ValueError("--bias-factor disagrees with the native history")
    elif bias_factor is not None:
        raise ValueError("--bias-factor is used with --quantity fes")
    with output_transaction(paths, reader.path, force) as staged:
        summary = reader.summary(through_step)
        if not summary["depositions"]:
            raise ValueError("No depositions in the selected step range")
        coordinates, ranges = [], []
        for axis, count in zip(axes, bins):
            period = reader.metadata.period[axis]
            default = ((0.0, period) if period else
                       (summary["center_min"][axis] - 3 * summary["max_sigma"][axis],
                        summary["center_max"][axis] + 3 * summary["max_sigma"][axis]))
            lower, upper = selected_bounds.get(axis, default)
            if not math.isfinite(lower) or not math.isfinite(upper) or lower >= upper:
                raise ValueError("Every preview range must have finite lower < upper")
            endpoint = not (period and math.isclose(upper - lower, period))
            coordinates.append(np.linspace(lower, upper, count, endpoint=endpoint))
            ranges.append([lower, upper])
        meshes = np.meshgrid(*coordinates, indexing="ij")
        points = np.zeros((point_count, dimension))
        for axis, value in selected_fixed.items():
            points[:, axis] = value[0]
        for axis, mesh in zip(axes, meshes):
            points[:, axis] = mesh.ravel()
        values = evaluate_bias(reader, points, through_step, hill_batch, point_batch, pb_normalization)
        if quantity == "fes":
            values = -factor / (factor - 1) * values
            values -= values.min()
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite preview values")
        columns = ["s%d" % (axis + 1) for axis in axes] + [quantity]
        with staged[0].open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(columns)
            for index, value in enumerate(values):
                writer.writerow([format(points[index, axis], ".17g") for axis in axes] + [format(value, ".17g")])
        label = ("Bias" if quantity == "bias" else "WTMetaD FES estimate") + " [%s]" % energy_unit
        figure, plot = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
        try:
            if len(axes) == 1:
                plot.plot(coordinates[0], values)
                plot.set_ylabel(label)
                plot.grid(alpha=0.25)
            else:
                artist = plot.pcolormesh(coordinates[0], coordinates[1], values.reshape(bins).T, shading="auto")
                figure.colorbar(artist, ax=plot, label=label)
                plot.set_ylabel(columns[1])
            plot.set_xlabel(columns[0])
            title = "Analytic hill sum" if reader.metadata.mode == "HILLS" else "PBMetaD combined bias"
            if quantity == "fes":
                title = "WTMetaD asymptotic FES estimate"
            if selected_fixed:
                title += "\nSlice: " + ", ".join("s%d=%g" % (axis + 1, value[0]) for axis, value in sorted(selected_fixed.items()))
            plot.set_title(title)
            figure.savefig(str(staged[1]), dpi=150)
        finally:
            plt.close(figure)
        metadata = dict(summary, product="surface", quantity=quantity, method="hills", axes=[axis + 1 for axis in axes],
                        bins=bins, bounds=ranges,
                        fixed={str(axis + 1): value[0] for axis, value in selected_fixed.items()},
                        energy_unit=energy_unit, through_step=through_step, columns=columns,
                        pb_normalization=pb_normalization, bias_factor_used=factor,
                        evaluator="analytic hill sum; runtime grid interpolation is not replayed",
                        interpretation="slice" if selected_fixed else "full selected surface")
        if quantity == "fes":
            metadata["fes_assumption"] = "F=-gamma/(gamma-1)*V+C at WTMetaD convergence; minimum on the selected grid is zero"
            metadata["coordinate_measure"] = "Free energy in the recorded CV coordinate; no Jacobian/entropy correction applied"
        staged[2].write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return [str(path.resolve()) for path in paths]
