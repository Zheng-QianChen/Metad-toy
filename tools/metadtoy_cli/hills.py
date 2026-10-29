"""Read deposition histories and evaluate their analytic bias kernels.

Parsing and CSV export use the standard library. NumPy is loaded only for
surface evaluation. Coordinates and energies retain the simulation's units.
"""
from dataclasses import asdict, dataclass
from itertools import islice
from pathlib import Path
import math


@dataclass(frozen=True)
class HistoryMetadata:
    # Per-axis sigma/period follow s1, s2, ...; legacy sigma is stored per row.
    version: int
    mode: str
    dimension: int
    sigma: tuple
    period: tuple
    thermal_energy: object = None
    bias_factor: object = None
    well_tempered: object = None
    kernel: str = "SMOOTH_GAUSSIAN"

    def as_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class Deposition:
    step: int
    centers: tuple
    heights: tuple
    sigma: tuple


def finite_number(text, label, positive=False, nonnegative=False):
    try:
        value = float(text)
    except ValueError:
        raise ValueError("%s must be numeric" % label) from None
    if not math.isfinite(value):
        raise ValueError("%s must be finite" % label)
    if positive and value <= 0:
        raise ValueError("%s must be positive" % label)
    if nonnegative and value < 0:
        raise ValueError("%s must be nonnegative" % label)
    return value


class HillsReader:
    """Reopenable streaming reader; offsets refer to a UTF-8 text stream."""

    def __init__(self, path, dimension=None):
        self.path = Path(path)
        self.requested_dimension = dimension
        if dimension is not None and dimension < 1:
            raise ValueError("dimension must be positive")
        with self.path.open(encoding="utf-8-sig") as stream:
            self.metadata, self.data_offset, self.header_lines = self._header(stream)

    def _header(self, stream):
        line_number = 0
        while True:
            offset = stream.tell()
            line = stream.readline()
            if not line:
                raise ValueError("Empty history without a native header")
            line_number += 1
            text = line.strip()
            if not text:
                continue
            if text.startswith("# METAD_TOY_BIAS"):
                if not line.endswith("\n"):
                    raise ValueError("Incomplete native HILLS header")
                fields = text.split()
                if len(fields) != 5:
                    raise ValueError("Invalid METAD_TOY_BIAS header")
                version, mode, dimension = int(fields[2]), fields[3], int(fields[4])
                if version not in (1, 2, 3) or mode not in ("HILLS", "PBMETAD") or dimension < 1:
                    raise ValueError("Unsupported native HILLS version, mode or dimension")
                expected = ("PARAMETERS", "SIGMA", "PERIOD", "FIELDS")
                header = {}
                for name in expected:
                    line = stream.readline()
                    line_number += 1
                    tokens = line.split()
                    if not line.endswith("\n") or tokens[:2] != ["#", name]:
                        raise ValueError("Missing or incomplete %s header" % name)
                    header[name] = tokens[2:]
                parameters = header["PARAMETERS"]
                if len(parameters) != (5 if version == 3 else 4):
                    raise ValueError("Invalid PARAMETERS header")
                finite_number(parameters[0], "initial height", positive=True)
                thermal = finite_number(parameters[1], "kBT", positive=True)
                factor = finite_number(parameters[2], "bias factor")
                if parameters[3] not in ("0", "1"):
                    raise ValueError("WT flag must be 0 or 1")
                tempered = parameters[3] == "1"
                if tempered and factor <= 1:
                    raise ValueError("WT bias factor must exceed one")
                kernel = parameters[4] if version == 3 else "SMOOTH_GAUSSIAN"
                if kernel not in ("SMOOTH_GAUSSIAN", "PLUMED_2_10"):
                    raise ValueError("Unsupported PB kernel: %s" % kernel)
                sigma = tuple(finite_number(x, "sigma", positive=True) for x in header["SIGMA"])
                period = tuple(finite_number(x, "period", nonnegative=True) for x in header["PERIOD"])
                height_count = dimension if mode == "PBMETAD" else 1
                names = (["step"] + ["s%d" % (i + 1) for i in range(dimension)]
                         + ["height%d" % (i + 1) for i in range(height_count)])
                if len(sigma) != dimension or len(period) != dimension or header["FIELDS"] != names:
                    raise ValueError("HILLS axis counts or FIELDS do not match the dimension")
                metadata = HistoryMetadata(version, mode, dimension, sigma, period,
                                           thermal, factor, tempered, kernel)
                break
            if text.startswith("#!"):
                raise ValueError("PLUMED HILLS files need a separate importer; this reader accepts MetaD-toy histories")
            if text.startswith("#"):
                continue
            # Legacy rows: step, D centers, actual height, shared sigma.
            dimension = len(text.split()) - 3
            if dimension < 1:
                raise ValueError("Legacy rows require step, centers, height and sigma")
            metadata = HistoryMetadata(0, "HILLS", dimension, (), (0.0,) * dimension)
            stream.seek(offset)
            line_number -= 1
            break
        if self.requested_dimension is not None and metadata.dimension != self.requested_dimension:
            raise ValueError("Requested dimension disagrees with the HILLS file")
        return metadata, stream.tell(), line_number

    def records(self, through_step=None):
        """Validate all rows, yielding depositions up to the optional step.

        Unterminated rows are rejected so a live writer cannot supply half a hill.
        Use a completed snapshot when analysing a simulation that is still running.
        """
        metadata = self.metadata
        previous_step = -1
        height_count = metadata.dimension if metadata.mode == "PBMETAD" else 1
        expected_count = 1 + metadata.dimension + height_count + (metadata.version == 0)
        with self.path.open(encoding="utf-8-sig") as stream:
            current_metadata, offset, header_lines = self._header(stream)
            if current_metadata != metadata:
                raise ValueError("History header changed while reading")
            stream.seek(offset)
            for line_number, line in enumerate(stream, header_lines + 1):
                if not line.endswith("\n"):
                    raise ValueError("Line %d is incomplete; use a completed history snapshot" % line_number)
                text = line.strip()
                if not text or (metadata.version == 0 and text.startswith("#")):
                    continue
                try:
                    fields = text.split()
                    if len(fields) != expected_count:
                        raise ValueError("expected %d fields" % expected_count)
                    step = int(fields[0])
                    if step < previous_step or step < 0 or step > 9223372036854775807:
                        raise ValueError("steps must be nonnegative, ordered int64 values")
                    centers = tuple(finite_number(x, "center") for x in fields[1:1 + metadata.dimension])
                    heights = tuple(finite_number(x, "height", nonnegative=True)
                                    for x in fields[1 + metadata.dimension:1 + metadata.dimension + height_count])
                    sigma = metadata.sigma
                    if metadata.version == 0:
                        sigma = (finite_number(fields[-1], "sigma", positive=True),) * metadata.dimension
                    previous_step = step
                except ValueError as error:
                    raise ValueError("%s:%d: %s" % (self.path, line_number, error)) from None
                if through_step is None or step <= through_step:
                    yield Deposition(step, centers, heights, sigma)

    def summary(self, through_step=None):
        dimension = self.metadata.dimension
        minimum = [math.inf] * dimension
        maximum = [-math.inf] * dimension
        largest_sigma = [0.0] * dimension
        count, first, last = 0, None, None
        for record in self.records(through_step):
            first = record.step if first is None else first
            last = record.step
            count += 1
            for axis in range(dimension):
                minimum[axis] = min(minimum[axis], record.centers[axis])
                maximum[axis] = max(maximum[axis], record.centers[axis])
                largest_sigma[axis] = max(largest_sigma[axis], record.sigma[axis])
        return dict(self.metadata.as_dict(), source=str(self.path.resolve()),
                    depositions=count, first_step=first, last_step=last,
                    center_min=minimum if count else None, center_max=maximum if count else None,
                    max_sigma=largest_sigma if count else None)


def evaluate_bias(reader, points, through_step=None, hill_batch=256, point_batch=512,
                  pb_normalization="mean"):
    """Evaluate analytic hill sums with bounded point-by-hill temporary arrays.

    points has shape [sample, dimension]. PB keeps one accumulated component per
    sample and CV; combination is performed after all deposition chunks.
    """
    import numpy as np
    metadata = reader.metadata
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != metadata.dimension or not np.isfinite(points).all():
        raise ValueError("Points must be a finite [sample, dimension] array")
    if hill_batch < 1 or point_batch < 1:
        raise ValueError("Batch sizes must be positive")
    if pb_normalization not in ("mean", "sum"):
        raise ValueError("PB normalization must be mean or sum")
    parallel = metadata.mode == "PBMETAD"
    result = np.zeros((len(points), metadata.dimension) if parallel else len(points))
    iterator = iter(reader.records(through_step))
    while True:
        records = list(islice(iterator, hill_batch))
        if not records:
            break
        centers = np.asarray([record.centers for record in records])
        sigma = np.asarray([record.sigma for record in records])
        heights = np.asarray([record.heights for record in records])
        for start in range(0, len(points), point_batch):
            stop = min(start + point_batch, len(points))
            quadratic_sum = np.zeros((stop - start, len(records)))
            for axis, period in enumerate(metadata.period):
                delta = points[start:stop, axis, None] - centers[None, :, axis]
                if not np.isfinite(delta).all():
                    raise ValueError("Coordinate difference overflow")
                if period:
                    delta = np.remainder(delta, period)
                    delta = np.where(delta > period / 2, delta - period, delta)
                with np.errstate(over="ignore", under="ignore"):
                    if period and not (parallel and metadata.kernel == "PLUMED_2_10"):
                        scaled = np.sin(math.pi * (delta / period)) / (math.pi * sigma[None, :, axis] / period)
                    else:
                        scaled = delta / sigma[None, :, axis]
                    quadratic = scaled * scaled
                    if parallel:
                        weights = np.exp(-0.5 * quadratic)
                        if metadata.kernel == "PLUMED_2_10":
                            weights = np.where(quadratic < 12.5, weights, 0.0)
                        result[start:stop, axis] += (weights * heights[None, :, axis]).sum(axis=1)
                    else:
                        quadratic_sum += quadratic
            if not parallel:
                result[start:stop] += (np.exp(-0.5 * quadratic_sum) * heights[None, :, 0]).sum(axis=1)
    if parallel:
        minimum = result.min(axis=1)
        partition = np.exp(-(result - minimum[:, None]) / metadata.thermal_energy).sum(axis=1)
        if pb_normalization == "mean":
            partition /= metadata.dimension
        result = minimum - metadata.thermal_energy * np.log(partition)
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite reconstructed bias")
    return result
