"""Bias-corrected trajectory histograms and free-energy reconstruction.

For time-dependent bias the input log weight is (V(x_t,t)-c(t))/kBT.
The WT-HILLS fallback estimates c(t) by quadrature over every biased dimension;
its finite domain and quasi-stationary WT assumption are saved in the output.
"""
import csv
import json
import math
from pathlib import Path
import re
from .analysis import output_transaction, _axis_values
from .colvar import ColvarReader
from .hills import HillsReader
from .dataset import fingerprint
from .progress import ProgressReporter


def logsumexp(values):
    import numpy as np
    maximum = float(np.max(values))
    if maximum == -math.inf:
        return -math.inf
    return maximum + math.log(float(np.exp(values - maximum).sum()))


def wt_offset(values, thermal_energy, bias_factor):
    """Uniform-volume quadrature for c(t), shifted to avoid exponential overflow.

    c = kBT log[ integral exp(gamma*V/((gamma-1)*kBT)) /
                 integral exp(V/((gamma-1)*kBT)) ].
    Equal cell volumes cancel. Subtracting max(V) before exponentiation also
    preserves invariance under adding a constant to the entire bias surface.
    """
    import numpy as np
    values = np.asarray(values, dtype=float)
    if (values.size == 0 or not np.isfinite(values).all() or
            not math.isfinite(thermal_energy) or thermal_energy <= 0 or
            not math.isfinite(bias_factor) or bias_factor <= 1):
        raise ValueError("WT offset requires finite grid values, kBT>0 and biasfactor>1")
    maximum = float(values.max())
    scaled = (values - maximum) / ((bias_factor - 1.) * thermal_energy)
    result = maximum + thermal_energy * (logsumexp(bias_factor * scaled) - logsumexp(scaled))
    if not math.isfinite(result):
        raise ValueError("nonfinite c(t)")
    return result


class WTOffset:
    """Incrementally replay native full-HILLS bias on a midpoint integration grid."""
    def __init__(self, history, bounds, bins=None, through_step=None, on_hill=None):
        import numpy as np
        metadata = history.metadata
        if not metadata.version or metadata.mode != "HILLS" or not metadata.well_tempered:
            raise ValueError("offline c(t) requires native full-HILLS WTMetaD; PB/legacy need supplied log weights or rbias")
        dimension = metadata.dimension
        if dimension > 3:
            raise ValueError("offline c(t) quadrature supports at most 3 biased dimensions; supply log weights or rbias")
        ranges = _axis_values(bounds, dimension, "ct-range", 3)
        if set(ranges) != set(range(dimension)):
            raise ValueError("provide --ct-range AXIS LOWER UPPER for every biased dimension")
        if bins is None:
            bins = ({1: 1024, 2: 128, 3: 48}[dimension],) * dimension
        elif len(bins) == 1:
            bins = tuple(bins) * dimension
        if len(bins) != dimension or any(type(count) is not int or count < 2 for count in bins):
            raise ValueError("--ct-bins needs >=2 bins for every biased dimension")
        if math.prod(bins) > 1000000:
            raise ValueError("c(t) grid exceeds one million cells; reduce --ct-bins")
        coordinates = []
        for axis, count in enumerate(bins):
            lower, upper = ranges[axis]
            if lower >= upper:
                raise ValueError("ct-range needs lower < upper")
            period = metadata.period[axis]
            if period and not math.isclose(upper - lower, period, rel_tol=1e-10):
                raise ValueError("periodic ct-range must span exactly one period")
            coordinates.append(lower + (np.arange(count) + .5) * ((upper - lower) / count))
        self.coordinates = coordinates
        self.values = np.zeros(tuple(bins))
        self.contribution = np.empty(tuple(bins))
        self.factors = [np.empty_like(axis) for axis in coordinates]
        self.on_hill = on_hill
        self.iterator = iter(history.records(through_step))
        self.pending = next(self.iterator, None)
        self.metadata = metadata
        self.ranges, self.bins = ranges, bins
        self.offset, self.updates, self.hills = 0., 0, 0
        self.last_step = -1

    def _add_hill(self, record):
        """Factor the diagonal Gaussian into 1D factors on the regular grid.

        This evaluates every grid cell; no tail cutoff or coarser quadrature
        is introduced. Reusable arrays bound temporary storage to the grid.
        """
        import numpy as np
        self.contribution.fill(record.heights[0])
        dimension = self.metadata.dimension
        for axis, period in enumerate(self.metadata.period):
            factor = self.factors[axis]
            np.subtract(self.coordinates[axis], record.centers[axis], out=factor)
            if not np.isfinite(factor).all():
                raise ValueError("Coordinate difference overflow")
            if period:
                np.remainder(factor, period, out=factor)
                factor[factor > period/2] -= period
                factor *= math.pi / period
                np.sin(factor, out=factor)
                factor /= math.pi * record.sigma[axis] / period
            else:
                factor /= record.sigma[axis]
            with np.errstate(over="ignore", under="ignore"):
                np.square(factor, out=factor)
                factor *= -.5
                np.exp(factor, out=factor)
            shape = [1] * dimension
            shape[axis] = len(factor)
            np.multiply(self.contribution, factor.reshape(shape), out=self.contribution)
        self.values += self.contribution

    def batch_at(self, steps, coordinates):
        """Reuse one offset for all samples between two deposition steps."""
        import numpy as np
        if len(steps) == 0:
            return np.empty(0)
        if steps[0] < self.last_step or np.any(steps[1:] < steps[:-1]):
            raise ValueError("c(t) replay cannot move backwards")
        for axis, period in enumerate(self.metadata.period):
            if not period:
                lower, upper = self.ranges[axis]
                if np.any(coordinates[:, axis] < lower) or np.any(coordinates[:, axis] > upper):
                    raise ValueError("sample lies outside c(t) integration domain on biased axis %d" % (axis+1))
        offsets = np.empty(len(steps))
        start = 0
        while start < len(steps):
            value = self.at(int(steps[start]), coordinates[start])
            stop = (len(steps) if self.pending is None else
                    int(np.searchsorted(steps, self.pending.step, side="left")))
            offsets[start:stop] = value
            start = stop
        self.last_step = int(steps[-1])
        return offsets

    def at(self, step, biased_coordinates):
        if step < self.last_step:
            raise ValueError("c(t) replay cannot move backwards")
        self.last_step = step
        # Native LAMMPS/GPUMD adapters deposit before evaluating/logging bias.
        # Therefore all hills with deposition.step <= sample.step are included.
        for axis, coordinate in enumerate(biased_coordinates):
            if not self.metadata.period[axis]:
                lower, upper = self.ranges[axis]
                if not lower <= coordinate <= upper:
                    raise ValueError("sample lies outside c(t) integration domain on biased axis %d" % (axis + 1))
        changed = False
        while self.pending is not None and self.pending.step <= step:
            for axis, center in enumerate(self.pending.centers):
                if not self.metadata.period[axis]:
                    lower, upper = self.ranges[axis]
                    if not lower <= center <= upper:
                        raise ValueError("hill center lies outside c(t) integration domain")
            self._add_hill(self.pending)
            self.hills += 1
            if self.on_hill is not None and self.hills % 128 == 0:
                self.on_hill(self.hills)
            changed = True
            self.pending = next(self.iterator, None)
        if changed:
            self.offset = wt_offset(self.values, self.metadata.thermal_energy, self.metadata.bias_factor)
            self.updates += 1
        return self.offset

    def finish(self):
        # Consume the remaining stream so malformed future history rows also
        # fail validation, even when a selected trajectory ends before them.
        for record in self.iterator:
            pass

    def description(self):
        return {"source": "native_wt_hills_quadrature",
                "bounds": [self.ranges[axis] for axis in range(self.metadata.dimension)],
                "bins": self.bins, "quadrature": "uniform midpoint", "hills_applied": self.hills,
                "offset_updates": self.updates, "deposition_order": "hill.step <= sample.step",
                "bias_evaluator": "analytic hills; host grid interpolation is not replayed",
                "assumption": "quasi-stationary WT approximation F=-gamma/(gamma-1)*V in the specified integration domain"}


def _column(fields, explicit, suffixes):
    if explicit is not None:
        if explicit not in fields:
            raise ValueError("missing COLVAR column: %s" % explicit)
        return explicit
    matches = [name for name in fields if name in suffixes or name.rsplit(".", 1)[-1] in suffixes]
    if len(matches) > 1:
        raise ValueError("ambiguous weight columns %s; select a column explicitly" % matches)
    return matches[0] if matches else None


def _cv_names(fields, requested):
    if requested is not None:
        result = list(requested)
    elif "value" in fields:
        result = ["value"]
    else:
        result = sorted((name for name in fields if re.fullmatch(r"s[1-9]\d*", name)),
                        key=lambda name: int(name[1:]))
    if not result or len(set(result)) != len(result) or any(name not in fields for name in result):
        raise ValueError("select existing unique CV columns with --cv NAME [NAME ...]")
    return result



class _LogHistogram:
    """Batch accumulation with stable per-bin log sums and exact sample counts."""
    def __init__(self, bins, axes, ranges, periodic):
        import numpy as np
        self.bins, self.axes, self.ranges, self.periodic = bins, axes, ranges, periodic
        self.counts = np.zeros(math.prod(bins), dtype=np.int64)
        self.log_mass = np.full(len(self.counts), -np.inf)
        self.log_square_mass = np.full(len(self.counts), -np.inf)
        self.all_mass = self.all_square_mass = -math.inf
        self.log_origin = None
        self.outside = 0

    def add(self, coordinates, log_weights):
        import numpy as np
        if not np.isfinite(log_weights).all():
            raise ValueError("nonfinite log weight")
        if self.log_origin is None:
            self.log_origin = float(log_weights[0])
        shifted = log_weights - self.log_origin
        if not np.isfinite(2*shifted).all():
            raise ValueError("log-weight dynamic range exceeds floating-point limits")
        self.all_mass = float(np.logaddexp(self.all_mass, np.logaddexp.reduce(shifted)))
        self.all_square_mass = float(np.logaddexp(self.all_square_mass, np.logaddexp.reduce(2*shifted)))
        inside = np.ones(len(coordinates), dtype=bool)
        plotted = []
        for axis, (lower, upper) in zip(self.axes, self.ranges):
            values = coordinates[:, axis]
            if axis in self.periodic:
                values = lower + np.remainder(values-lower, self.periodic[axis])
            if not np.isfinite(values).all():
                raise ValueError("nonfinite periodic coordinates")
            inside &= (values >= lower) & (values <= upper)
            plotted.append(values)
        if inside.any():
            indices = [np.minimum(count-1, ((values[inside]-lower)/(upper-lower)*count).astype(np.int64))
                       for values, (lower, upper), count in zip(plotted, self.ranges, self.bins)]
            flat = np.ravel_multi_index(tuple(indices), self.bins)
            np.add.at(self.counts, flat, 1)
            # at() applies duplicate bin indices in sample order; unlike a
            # plain indexed assignment it keeps every observation's weight.
            np.logaddexp.at(self.log_mass, flat, shifted[inside])
            np.logaddexp.at(self.log_square_mass, flat, 2*shifted[inside])
        self.outside += int((~inside).sum())
        return inside


def create_reweighted_fes(colvar, prefix, history=None, cv_columns=None, fields=None,
                          axes=None, bins=(100,), bounds=(), periods=(), thermal_energy=None,
                          weight_source="auto", log_weight_column=None, rbias_column=None,
                          bias_column=None, offset_column=None, bias_cv_columns=None,
                          ct_bounds=(), ct_bins=None, from_step=None, through_step=None,
                          discard_rows=0, energy_unit="native units", force=False,
                          sample_batch=8192, progress=None):
    """Export a 1D/2D weighted marginal FES plus per-frame audit weights.

    All observations must be equally spaced in step/time when those fields
    exist. Unequal spacing requires explicit residence-time weights upstream.
    ESS measures weight concentration only, not time correlation.
    """
    reporter = ProgressReporter(progress)
    reporter.begin("initialize")
    if type(sample_batch) is not int or sample_batch < 1:
        raise ValueError("sample batch size must be positive")
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    if discard_rows < 0 or (from_step is not None and from_step < 0):
        raise ValueError("discard rows and from-step must be nonnegative")
    if through_step is not None and (through_step < 0 or (from_step is not None and through_step < from_step)):
        raise ValueError("invalid step window")
    paths = [Path(str(prefix) + suffix).resolve() for suffix in (".csv", ".png", ".json", ".weights.csv")]
    sources = [Path(colvar).resolve()] + ([Path(history).resolve()] if history is not None else [])
    if len(set(sources)) != len(sources):
        raise ValueError("COLVAR and HILLS must be distinct inputs")
    for path in paths:
        if path in sources or (path.exists() and any(path.samefile(source) for source in sources)):
            raise ValueError("output aliases a source file")
    reporter.begin("hash_inputs", len(sources), "files")
    snapshots = []
    for source in sources:
        snapshots.append(fingerprint(source))
        reporter.update(len(snapshots), detail=source.name, force=True)
    reader = ColvarReader(colvar, fields)
    hills = HillsReader(history) if history is not None else None
    if thermal_energy is None and hills is not None:
        thermal_energy = hills.metadata.thermal_energy
    if thermal_energy is None or not math.isfinite(thermal_energy) or thermal_energy <= 0:
        raise ValueError("FES requires --kbt in the bias energy unit, or native HILLS with positive kBT")
    if hills is not None and hills.metadata.thermal_energy is not None and not math.isclose(
            thermal_energy, hills.metadata.thermal_energy, rel_tol=1e-10):
        raise ValueError("--kbt disagrees with native HILLS")
    cvs = _cv_names(reader.fields, cv_columns)
    dimension = len(cvs)
    selected_axes = tuple(axis - 1 for axis in (axes or range(1, min(2, dimension) + 1)))
    if len(selected_axes) not in (1, 2) or len(set(selected_axes)) != len(selected_axes) or any(
            axis < 0 or axis >= dimension for axis in selected_axes):
        raise ValueError("--axes selects one or two distinct 1-based CV axes")
    bins = tuple(bins) * len(selected_axes) if len(bins) == 1 else tuple(bins)
    if len(bins) != len(selected_axes) or any(type(count) is not int or count < 2 for count in bins) or math.prod(bins) > 1000000:
        raise ValueError("histogram needs >=2 bins per selected axis and at most one million bins")
    chosen_bounds = _axis_values(bounds, dimension, "range", 3)
    if set(chosen_bounds) - set(selected_axes):
        raise ValueError("--range applies only to histogram axes")
    if (from_step is not None or through_step is not None) and "step" not in reader.fields:
        raise ValueError("step filtering requires an integer step column")
    def included(indices, columns):
        mask = indices >= discard_rows
        if from_step is not None:
            mask &= columns["step"] >= from_step
        if through_step is not None:
            mask &= columns["step"] <= through_step
        return mask
    # Select exactly one weighting contract. Dynamic raw bias is never an
    # implicit substitute for normalized rbias or for a supplied log weight.
    choices = {"auto", "log-weight", "rbias", "bias-offset", "wt-hills", "static-bias"}
    if weight_source not in choices:
        raise ValueError("unknown weight source")
    specifications = {
        "log-weight": (log_weight_column, ("log_weight", "logweight")),
        "rbias": (rbias_column, ("rbias",)),
        "bias": (bias_column, ("bias", "bias_eV")),
        "offset": (offset_column, ("rct", "ct", "offset")),
    }
    selected_weight_columns = {}
    def discover(role):
        if role not in selected_weight_columns:
            explicit, suffixes = specifications[role]
            selected_weight_columns[role] = _column(reader.fields, explicit, suffixes)
        return selected_weight_columns[role]
    mode = weight_source
    if mode == "auto":
        if discover("log-weight"):
            mode = "log-weight"
        elif discover("rbias"):
            mode = "rbias"
        elif discover("bias") and discover("offset"):
            mode = "bias-offset"
        elif hills is not None and discover("bias"):
            mode = "wt-hills"
        else:
            raise ValueError("no normalized weights: supply log_weight, rbias, bias+rct, or native WT HILLS with --ct-range; raw bias requires explicit --weight-source static-bias")
    required = {"log-weight": ("log-weight",), "rbias": ("rbias",), "bias-offset": ("bias", "offset"),
                "wt-hills": ("bias",), "static-bias": ("bias",)}[mode]
    if any(not discover(name) for name in required):
        raise ValueError("missing columns for weight source %s" % mode)
    if any(explicit is not None and role not in required for role, (explicit, suffixes) in specifications.items()):
        raise ValueError("a selected weight column is unused; set --weight-source to the intended contract")
    selected_weight_columns = {name: selected_weight_columns[name] for name in required}
    correction = None
    bias_cvs = None
    if mode == "wt-hills":
        if hills is None or "step" not in reader.fields:
            raise ValueError("offline c(t) requires native HILLS and an integer COLVAR step column")
        bias_cvs = _cv_names(reader.fields, bias_cv_columns or cvs)
        if len(bias_cvs) != hills.metadata.dimension:
            raise ValueError("--bias-cv must list all biased CVs in HILLS dimension order")
        correction = WTOffset(hills, ct_bounds, ct_bins, through_step)
    elif ct_bounds or ct_bins is not None or bias_cv_columns is not None:
        raise ValueError("--ct-range/--ct-bins/--bias-cv are used only with wt-hills weights")
    periodic = {axis: value[0] for axis, value in _axis_values(periods, dimension, "period", 2).items()}
    if any(value <= 0 for value in periodic.values()):
        raise ValueError("periods must be positive")
    if hills is not None:
        mapping = bias_cvs or []
        for column, period in zip(mapping, hills.metadata.period):
            if column in cvs and period:
                axis = cvs.index(column)
                if axis in periodic and not math.isclose(periodic[axis], period):
                    raise ValueError("--period disagrees with HILLS")
                periodic[axis] = period
    # The scan validates cadence across batch boundaries and finds CV ranges.
    reporter.begin("scan", unit="frames")
    minima, maxima = np.full(dimension, np.inf), np.full(dimension, -np.inf)
    observed = selected = 0
    first_step = last_step = None
    previous_clock = interval = None
    clock_name = "time" if "time" in reader.fields else "step" if "step" in reader.fields else None
    for indices, batch in reader.batches(sample_batch):
        observed += len(indices)
        mask = included(indices, batch)
        reporter.update(observed)
        if not mask.any():
            continue
        values = np.column_stack([batch[column][mask] for column in cvs])
        minima = np.minimum(minima, values.min(axis=0))
        maxima = np.maximum(maxima, values.max(axis=0))
        if clock_name is not None:
            clocks = batch[clock_name][mask]
            chain = clocks if previous_clock is None else np.concatenate(([previous_clock], clocks))
            differences = np.diff(chain)
            if len(differences):
                if interval is None:
                    interval = differences[0].item()
                tolerance = np.maximum(1e-12, 1e-6 * np.maximum(np.abs(differences), abs(interval)))
                if np.any(np.abs(differences - interval) > tolerance):
                    raise ValueError("selected COLVAR observations are not equally spaced; resample or supply a consistent time measure")
            previous_clock = clocks[-1].item()
        if "step" in batch:
            steps = batch["step"][mask]
            if selected == 0:
                first_step = int(steps[0])
            last_step = int(steps[-1])
        selected += int(mask.sum())
    reporter.update(observed, force=True)
    if not selected:
        raise ValueError("no COLVAR observations in the selected window")
    ranges, edges = [], []
    for axis, count in zip(selected_axes, bins):
        if axis in chosen_bounds:
            lower, upper = chosen_bounds[axis]
        elif axis in periodic:
            lower, upper = 0., periodic[axis]
        else:
            lower, upper = float(minima[axis]), float(maxima[axis])
        if not math.isfinite(lower) or not math.isfinite(upper) or lower >= upper:
            raise ValueError("constant/invalid CV range; provide an explicit --range")
        if axis in periodic and not math.isclose(upper - lower, periodic[axis], rel_tol=1e-10):
            raise ValueError("periodic histogram range must span one full period")
        ranges.append([lower, upper])
        edges.append(np.linspace(lower, upper, count + 1))
    size = math.prod(bins)
    histogram = _LogHistogram(bins, selected_axes, ranges, periodic)
    reporter.begin("weights", selected, "frames")
    processed = 0
    if correction is not None:
        correction.on_hill = lambda count: reporter.update(processed, detail="HILLS %d" % count)
    with output_transaction(paths, sources[0], force) as staged:
        with staged[3].open("w", encoding="utf-8", newline="") as stream:
            weight_writer = csv.writer(stream)
            weight_writer.writerow(["row", "step", "log_weight", "offset", "in_histogram"] + cvs)
            for indices, batch in reader.batches(sample_batch):
                mask = included(indices, batch)
                if not mask.any():
                    reporter.update(processed)
                    continue
                indices = indices[mask]
                batch = {name: values[mask] for name, values in batch.items()}
                coordinates = np.column_stack([batch[column] for column in cvs])
                offsets = None
                if mode == "log-weight":
                    log_weights = batch[selected_weight_columns["log-weight"]]
                elif mode == "rbias":
                    log_weights = batch[selected_weight_columns["rbias"]] / thermal_energy
                else:
                    bias = batch[selected_weight_columns["bias"]]
                    if mode == "bias-offset":
                        offsets = batch[selected_weight_columns["offset"]]
                    elif mode == "wt-hills":
                        offsets = correction.batch_at(batch["step"], np.column_stack([batch[column] for column in bias_cvs]))
                    else:
                        offsets = np.zeros(len(indices))
                    log_weights = (bias - offsets) / thermal_energy
                inside = histogram.add(coordinates, log_weights)
                steps = batch.get("step")
                weight_writer.writerows(
                    [int(index), "" if steps is None else int(steps[position]), "%.17g" % log_weights[position],
                     "" if offsets is None else "%.17g" % offsets[position], int(inside[position])] +
                    ["%.17g" % value for value in coordinates[position]]
                    for position, index in enumerate(indices))
                processed += len(indices)
                reporter.update(processed)
        reporter.update(processed, force=True)
        reporter.begin("export")
        counts, log_mass, log_square_mass = histogram.counts, histogram.log_mass, histogram.log_square_mass
        all_mass, all_square_mass = histogram.all_mass, histogram.all_square_mass
        out_of_range, log_origin = histogram.outside, histogram.log_origin
        if correction is not None:
            correction.finish()
        elif hills is not None:
            for record in hills.records():
                pass
        populated = counts > 0
        if not populated.any():
            raise ValueError("no selected observations fall inside histogram bounds")
        mass_total = logsumexp(log_mass)
        volume = math.prod((upper - lower) / count for (lower, upper), count in zip(ranges, bins))
        log_density = log_mass - mass_total - math.log(volume)
        density = np.exp(log_density)
        fes = np.full(size, np.nan)
        fes[populated] = -thermal_energy * (log_density[populated] - np.max(log_density[populated]))
        if not np.isfinite(fes[populated]).all():
            raise ValueError("FES exceeds floating-point range")
        ess = np.zeros(size)
        ess[populated] = np.minimum(counts[populated], np.exp(np.minimum(
            2 * log_mass[populated] - log_square_mass[populated], np.log(counts[populated]))))
        centers = [(edge[:-1] + edge[1:]) * .5 for edge in edges]
        grid = np.stack(np.meshgrid(*centers, indexing="ij"), axis=-1).reshape(size, len(selected_axes))
        columns = [cvs[axis] for axis in selected_axes] + ["fes", "probability_density", "log_probability_density", "count", "weight_ess"]
        with staged[0].open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(columns)
            for index in range(size):
                writer.writerow(["%.17g" % value for value in grid[index]] +
                                ["" if not populated[index] else "%.17g" % fes[index],
                                 "%.17g" % density[index], "" if not populated[index] else "%.17g" % log_density[index],
                                 int(counts[index]), "%.17g" % ess[index]])
        reporter.begin("plot")
        figure, plot = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
        try:
            label = "Reweighted FES [%s]" % energy_unit
            if len(selected_axes) == 1:
                plot.plot(centers[0], fes)
                plot.set_ylabel(label)
                plot.grid(alpha=.25)
            else:
                artist = plot.pcolormesh(edges[0], edges[1], np.ma.masked_invalid(fes.reshape(bins).T), shading="flat")
                figure.colorbar(artist, ax=plot, label=label)
                plot.set_ylabel(columns[1])
            plot.set_xlabel(columns[0])
            plot.set_title("Trajectory-reweighted FES")
            figure.savefig(str(staged[1]), dpi=150)
        finally:
            plt.close(figure)
        reporter.begin("validate", len(sources), "files")
        for source_index, (source, snapshot) in enumerate(zip(sources, snapshots), 1):
            if fingerprint(source) != snapshot:
                raise ValueError("source changed during FES reconstruction")
            reporter.update(source_index, detail=source.name, force=True)
        total_ess = math.exp(min(math.log(selected), 2 * all_mass - all_square_mass))
        retained_fraction = min(1., math.exp(mass_total - all_mass))
        reporter.begin("publish")
        manifest = {
            "schema_version": 1, "product": "fes", "method": "reweight", "weight_source": mode,
            "sources": snapshots, "thermal_energy": thermal_energy, "energy_unit": energy_unit,
            "cv_columns": cvs, "axes": [axis + 1 for axis in selected_axes], "bins": bins, "bounds": ranges,
            "periods": {str(axis + 1): period for axis, period in periodic.items()}, "columns": columns,
            "observations_total": observed, "observations_selected": selected,
            "execution": {"sample_batch": sample_batch, "phase_seconds_before_publication": dict(reporter.timings)},
            "observations_in_histogram": int(counts.sum()), "observations_outside_histogram": out_of_range,
            "first_step": first_step, "last_step": last_step, "discard_rows": discard_rows,
            "from_step": from_step, "through_step": through_step, "weight_columns": selected_weight_columns,
            "weight_ess": total_ess, "retained_weight_fraction": retained_fraction,
            "empty_bins": int((~populated).sum()), "log_weight_origin": log_origin,
            "offset": correction.description() if correction is not None else {"source": mode},
            "sampling": {"clock": clock_name, "interval": interval,
                         "assumption": "equal-time observations; equilibrated/quasi-stationary sampling at constant temperature"},
            "interpretation": "marginal histogram over all unplotted variables",
            "normalization": "probability density normalized within histogram bounds; minimum populated-bin FES is zero",
            "coordinate_measure": "CV-coordinate free energy; no radial Jacobian correction",
            "ess_note": "weight concentration only; does not correct time correlation",
            "missing_bins": "zero probability, undefined/infinite FES; CSV blank and plot masked",
        }
        staged[2].write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    reporter.done()
    return [str(path) for path in paths]
