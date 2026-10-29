"""Thin argument parser; analysis operations remain reusable by a future TUI."""
import argparse
import json
import sys
from . import __version__
from .analysis import create_preview, export_csv
from .hills import HillsReader


def _nonnegative(value):
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return parsed



def _reweight_arguments(action, standalone=False):
    """Shared FES controls used by both the dedicated command and HILLS preview."""
    if standalone:
        action.add_argument("colvar", help="Completed named-column COLVAR or numeric CSV")
        action.add_argument("--hills", help="Native HILLS for kBT and optional offline c(t)")
        action.add_argument("--out", required=True)
        action.add_argument("--axes", nargs="+", type=int)
        action.add_argument("--bins", nargs="+", type=int, default=[100])
        action.add_argument("--range", dest="bounds", action="append", nargs=3, default=[],
                            metavar=("AXIS", "LOWER", "UPPER"))
        action.add_argument("--through-step", type=_nonnegative)
        action.add_argument("--energy-unit", default="native units", help="Label only; no unit conversion")
        action.add_argument("--force", action="store_true")
    else:
        action.add_argument("--colvar", help="FES samples; defaults to COLVAR beside HILLS")
    action.add_argument("--progress", choices=("auto", "always", "never"), default="auto",
                        help="FES progress on stderr; auto enables it in interactive terminals")
    action.add_argument("--sample-batch", type=int, default=8192, help="FES observations per bounded processing batch")
    action.add_argument("--cv", nargs="+", help="CV column names in axis order; default value or s1,s2,...")
    action.add_argument("--fields", nargs="+", help="Explicit column names for headerless input")
    action.add_argument("--kbt", type=float, help="kBT in the same energy unit as bias; defaults to native HILLS")
    action.add_argument("--weight-source", choices=("auto", "log-weight", "rbias", "bias-offset", "wt-hills", "static-bias"),
                        default="auto", help="auto requires normalized weights or native WT c(t); static-bias is explicit")
    action.add_argument("--log-weight-column")
    action.add_argument("--rbias-column")
    action.add_argument("--bias-column")
    action.add_argument("--offset-column", help="Column storing c(t), in the bias energy unit")
    action.add_argument("--bias-cv", nargs="+", help="All biased CV columns in HILLS order for offline c(t)")
    action.add_argument("--ct-range", action="append", nargs=3, default=[], metavar=("AXIS", "LOWER", "UPPER"),
                        help="Integration domain for c(t); repeat for every biased dimension")
    action.add_argument("--ct-bins", nargs="+", type=int, help="c(t) midpoint quadrature resolution")
    action.add_argument("--period", action="append", nargs=2, default=[], metavar=("AXIS", "PERIOD"))
    action.add_argument("--from-step", type=_nonnegative)
    action.add_argument("--discard-rows", type=_nonnegative, default=0, help="Discard this many initial observations")


def _reweight_options(arguments, progress=None):
    return dict(cv_columns=arguments.cv, fields=arguments.fields, thermal_energy=arguments.kbt,
                weight_source=arguments.weight_source, log_weight_column=arguments.log_weight_column,
                rbias_column=arguments.rbias_column, bias_column=arguments.bias_column,
                offset_column=arguments.offset_column, bias_cv_columns=arguments.bias_cv,
                ct_bounds=arguments.ct_range, ct_bins=arguments.ct_bins, periods=arguments.period,
                from_step=arguments.from_step, discard_rows=arguments.discard_rows,
                sample_batch=arguments.sample_batch, progress=progress)


def parser():
    root = argparse.ArgumentParser(prog="metadtoy", description="Offline MetaD-toy analysis (LAMMPS and GPUMD histories)")
    root.add_argument("--version", action="version", version=__version__)
    commands = root.add_subparsers(dest="command", required=True)
    hills = commands.add_parser("hills", help="Inspect, export and preview deposited bias histories")
    actions = hills.add_subparsers(dest="action", required=True)
    inspect = actions.add_parser("inspect", help="Validate a history and print its settings/statistics as JSON")
    export = actions.add_parser("export", help="Export raw depositions to CSV plus a JSON settings sidecar")
    preview = actions.add_parser("preview", help="Write a 1D/2D surface CSV, PNG and JSON; higher-D inputs need a slice")
    for action in (inspect, export, preview):
        action.add_argument("history", help="MetaD-toy native HILLS v1-v3 or legacy step/centers/height/sigma")
        action.add_argument("--dimensions", type=int, help="Optional expected CV dimension")
        action.add_argument("--through-step", type=_nonnegative, help="Include depositions up to this step")
    export.add_argument("--csv", required=True, help="Destination .csv file")
    export.add_argument("--force", action="store_true", help="Replace existing output files")
    preview.add_argument("--out", required=True, help="Output prefix for .csv, .png and .json")
    preview.add_argument("--axes", nargs="+", type=int, help="One or two 1-based axes, e.g. --axes 1 2")
    preview.add_argument("--bins", nargs="+", type=int, default=[200], help="Grid points: one count or one per selected axis")
    preview.add_argument("--range", dest="bounds", action="append", nargs=3, default=[], metavar=("AXIS", "LOWER", "UPPER"))
    preview.add_argument("--at", dest="fixed", action="append", nargs=2, default=[], metavar=("AXIS", "VALUE"),
                         help="Fixed coordinate for each unplotted axis; repeat as needed")
    preview.add_argument("--quantity", choices=("bias", "fes"), default="bias")
    preview.add_argument("--bias-factor", type=float, help="Required for FES from legacy WT HILLS")
    preview.add_argument("--energy-unit", default="native units", help="Plot label only; no unit conversion")
    preview.add_argument("--pb-normalization", choices=("mean", "sum"), default="mean",
                         help="PB combination convention; the native HILLS header does not store it")
    preview.add_argument("--hill-batch", type=int, default=256)
    preview.add_argument("--point-batch", type=int, default=512)
    preview.add_argument("--force", action="store_true")

    preview.add_argument("--method", choices=("reweight", "hills"), default="reweight",
                         help="FES estimator (default reweight); bias plots are unchanged")
    _reweight_arguments(preview)
    fes = commands.add_parser("fes", help="Reconstruct FES from bias-reweighted CV trajectories")
    fes_actions = fes.add_subparsers(dest="action", required=True)
    fes_reconstruct = fes_actions.add_parser("reconstruct", help="Write reweighted FES, PNG, metadata and frame weights")
    _reweight_arguments(fes_reconstruct, standalone=True)

    trajectory = commands.add_parser("trajectory", help="Inspect LAMMPS dump or XYZ trajectories")
    trajectory_actions = trajectory.add_subparsers(dest="action", required=True)
    trajectory_inspect = trajectory_actions.add_parser("inspect", help="Validate all frames and summarize identity/cell metadata")
    trajectory_inspect.add_argument("input")
    dataset = commands.add_parser("dataset", help="Build offline descriptor datasets")
    dataset_actions = dataset.add_subparsers(dest="action", required=True)
    dataset_build = dataset_actions.add_parser("build", help="Export Distance features with source/frame metadata")
    dataset_build.add_argument("inputs", nargs="+", help="Completed trajectory snapshots; each input is a separate segment")
    dataset_build.add_argument("--features", required=True, help="JSON descriptor definitions including identity and length_unit")
    dataset_build.add_argument("--out", required=True, help="Prefix for .csv, .frames.jsonl and .json")
    dataset_build.add_argument("--assume-fixed-order", action="store_true", help="Acknowledge stable atom rows for identity=row")
    dataset_build.add_argument("--stride", type=int, default=1, help="Select every Nth frame, beginning at frame zero")
    dataset_build.add_argument("--validation-fraction", type=float, default=0., help="Use the final fraction of each source for validation")
    dataset_build.add_argument("--gap-frames", type=int, default=1, help="Selected frames excluded between train/validation blocks")
    dataset_build.add_argument("--force", action="store_true")
    for action in (trajectory_inspect, dataset_build):
        action.add_argument("--format", choices=("auto", "lammps-dump", "xyz"), default="auto")
        action.add_argument("--pbc", nargs=3, metavar=("X", "Y", "Z"),
                            help="T/F flags when absent from input; conflicting file flags are rejected")

    return root


def main(argv=None):
    arguments = parser().parse_args(argv)
    from .progress import TerminalProgress
    show_fes = (arguments.command == "fes" or
                (arguments.command == "hills" and arguments.action == "preview" and
                 arguments.quantity == "fes" and arguments.method == "reweight"))
    display = TerminalProgress(arguments.progress if show_fes else "never")
    callback = display if display.enabled else None
    try:

        if arguments.command == "trajectory":
            from .trajectory import TrajectoryReader
            print(json.dumps(TrajectoryReader(arguments.input, arguments.format, arguments.pbc).summary(),
                             indent=2, allow_nan=False))
            return 0
        if arguments.command == "dataset":
            from .dataset import build_dataset
            outputs = build_dataset(arguments.inputs, arguments.features, arguments.out,
                                    format=arguments.format, pbc=arguments.pbc,
                                    assume_fixed_order=arguments.assume_fixed_order, stride=arguments.stride,
                                    validation_fraction=arguments.validation_fraction,
                                    gap_frames=arguments.gap_frames, force=arguments.force)
            print(json.dumps({"outputs": outputs}, indent=2))
            return 0

        if arguments.command == "fes":
            from .reweight import create_reweighted_fes
            outputs = create_reweighted_fes(arguments.colvar, arguments.out, history=arguments.hills,
                                            axes=arguments.axes, bins=arguments.bins, bounds=arguments.bounds,
                                            through_step=arguments.through_step, energy_unit=arguments.energy_unit,
                                            force=arguments.force, **_reweight_options(arguments, callback))
            print(json.dumps({"outputs": outputs}, indent=2))
            return 0

        reader = HillsReader(arguments.history, arguments.dimensions)
        if arguments.action == "inspect":
            print(json.dumps(reader.summary(arguments.through_step), indent=2, allow_nan=False))
            return 0
        if arguments.action == "export":
            outputs = export_csv(reader, arguments.csv, arguments.through_step, arguments.force)
        else:
            outputs = create_preview(reader, arguments.out, axes=arguments.axes, bins=arguments.bins,
                                     bounds=arguments.bounds, fixed=arguments.fixed, quantity=arguments.quantity,
                                     bias_factor=arguments.bias_factor, energy_unit=arguments.energy_unit,
                                     through_step=arguments.through_step, hill_batch=arguments.hill_batch,
                                     point_batch=arguments.point_batch, pb_normalization=arguments.pb_normalization,
                                     force=arguments.force, method=arguments.method, colvar=arguments.colvar,
                                     reweight_options=_reweight_options(arguments, callback))
        print(json.dumps({"outputs": outputs}, indent=2))
        return 0
    except KeyboardInterrupt:
        display.stop("已取消")
        return 130
    except ImportError as error:
        display.stop()
        print("metadtoy: this command requires optional analysis dependencies; install tools/requirements-analysis.txt (%s)" % error, file=sys.stderr)
        return 2
    except (ValueError, OSError, OverflowError) as error:
        display.stop()
        print("metadtoy: %s" % error, file=sys.stderr)
        return 2
