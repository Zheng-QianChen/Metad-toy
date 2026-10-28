#!/usr/bin/env python3
"""Build a wheel and install a private CLI runtime for the CMake cli component.

The build copies package inputs into a temporary staging directory. Installation
creates a venv under the final prefix and publishes a small launcher after the
installed package passes its dependency/import checks. Pip's normal environment
settings (PIP_INDEX_URL, PIP_NO_INDEX, PIP_FIND_LINKS) also apply to these steps.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import venv

LAUNCHER_MARKER = "# MetaD-toy managed CLI launcher"
OWNER = "metadtoy-cli"


def run(command, **options):
    print("+ " + shlex.join([str(value) for value in command]), flush=True)
    subprocess.run([str(value) for value in command], check=True, **options)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(source, output):
    """Stage sources and publish wheels only after both pip commands succeed."""
    source, output = source.resolve(), output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="cli-build-", dir=output.parent) as temporary:
        staging = Path(temporary)
        package = staging / "package"
        package.mkdir()
        for name in ("pyproject.toml", "requirements-analysis.txt"):
            shutil.copy2(source / name, package / name)
        shutil.copytree(source / "metadtoy_cli", package / "metadtoy_cli",
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        wheels = staging / "wheels"
        wheels.mkdir()
        run([sys.executable, "-m", "pip", "wheel", "--disable-pip-version-check",
             "--no-deps", "--wheel-dir", wheels, package])
        # This wheel bootstraps the target venv even if the OS omits ensurepip.
        run([sys.executable, "-m", "pip", "download", "--disable-pip-version-check",
             "--only-binary=:all:", "--no-deps", "--dest", wheels, "pip>=22.3"])
        package_wheel, = wheels.glob("metadtoy_cli-*.whl")
        pip_wheel, = wheels.glob("pip-*.whl")
        manifest = {}
        for role, wheel in (("package", package_wheel), ("pip", pip_wheel)):
            shutil.copy2(wheel, output / wheel.name)
            manifest[role] = {"file": wheel.name, "sha256": digest(wheel)}
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def child(prefix, relative):
    """Require installation directories to remain beneath the chosen prefix."""
    relative = Path(relative)
    if relative.is_absolute() or ".." in relative.parts or relative == Path("."):
        raise ValueError("installation directory must be a nonempty relative path: %s" % relative)
    path = (prefix / relative).resolve()
    if not path.is_relative_to(prefix) or path == prefix:
        raise ValueError("installation directory escapes the prefix: %s" % path)
    return path


def install(wheels, prefix, bindir, runtime_dir):
    if os.name != "posix":
        raise ValueError("CMake CLI installation requires Linux/WSL or macOS")
    prefix, wheels = prefix.resolve(), wheels.resolve()
    bin_directory = child(prefix, bindir)
    runtime = child(prefix, runtime_dir)
    if runtime == bin_directory or runtime.is_relative_to(bin_directory) or bin_directory.is_relative_to(runtime):
        raise ValueError("launcher and runtime directories must be separate")
    launcher = bin_directory / "metadtoy"
    ownership = runtime / "install-owner.json"
    environment = runtime / "venv"
    python = environment / "bin/python"
    base_python = str(Path(getattr(sys, "_base_executable", sys.executable)).resolve())

    # Check ownership before pip or venv can alter a preexisting installation.
    if launcher.is_symlink() or (launcher.exists() and
            (not launcher.is_file() or LAUNCHER_MARKER not in launcher.read_text())):
        raise ValueError("refusing to replace an unrelated command: %s" % launcher)
    if runtime.exists():
        if not ownership.is_file():
            raise ValueError("runtime directory is not owned by MetaD-toy: %s" % runtime)
        owner = json.loads(ownership.read_text())
        if owner.get("owner") != OWNER or owner.get("base_python") != base_python:
            raise ValueError("existing runtime owner/Python differs; choose a new runtime directory")

    manifest = json.loads((wheels / "manifest.json").read_text())
    wheel_paths = {}
    for role in ("package", "pip"):
        record = manifest[role]
        name = record["file"]
        if Path(name).name != name:
            raise ValueError("invalid wheel filename in manifest")
        path = wheels / name
        if digest(path) != record["sha256"]:
            raise ValueError("wheel changed since packaging: %s" % path)
        wheel_paths[role] = path

    runtime.mkdir(parents=True, exist_ok=True)
    ownership.write_text(json.dumps({"owner": OWNER, "base_python": base_python}, indent=2) + "\n")
    # Reuse an existing managed environment so satisfied dependencies stay put.
    if not python.exists():
        venv.EnvBuilder(with_pip=False, symlinks=True).create(environment)
    environment_vars = os.environ.copy()
    environment_vars.pop("PYTHONHOME", None)
    environment_vars["PYTHONPATH"] = str(wheel_paths["pip"])
    run([python, "-m", "pip", "install", "--disable-pip-version-check", wheel_paths["pip"]],
        env=environment_vars)
    environment_vars.pop("PYTHONPATH", None)
    pip = [python, "-I", "-m", "pip", "--disable-pip-version-check"]
    # Resolve dependencies, then replace the package even for an unchanged
    # development version number. The second operation leaves dependencies alone.
    run(pip + ["install", wheel_paths["package"]], env=environment_vars)
    run(pip + ["install", "--no-deps", "--force-reinstall", wheel_paths["package"]],
        env=environment_vars)
    run(pip + ["check"], env=environment_vars)
    run([python, "-I", "-c",
         "import numpy, matplotlib; from metadtoy_cli.cli import main; import metadtoy_cli; print('metadtoy', metadtoy_cli.__version__)"],
        env=environment_vars)

    # The command uses the private interpreter regardless of the caller's
    # active Conda/venv, current directory or PYTHONPATH.
    text = "#!/bin/sh\n" + LAUNCHER_MARKER + "\nexec " + shlex.quote(str(python)) + ' -I -m metadtoy_cli "$@"\n'
    bin_directory.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".metadtoy-", dir=bin_directory)
    try:
        with os.fdopen(descriptor, "w") as stream:
            stream.write(text)
        os.chmod(temporary, 0o755)
        os.replace(temporary, launcher)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    (runtime / "installed-wheel.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("Installed command: %s" % launcher)
    if str(bin_directory) not in os.environ.get("PATH", "").split(os.pathsep):
        print("Add this directory to PATH: %s" % bin_directory)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    builder = actions.add_parser("build")
    builder.add_argument("--source", type=Path, required=True)
    builder.add_argument("--output", type=Path, required=True)
    installer = actions.add_parser("install")
    installer.add_argument("--wheels", type=Path, required=True)
    installer.add_argument("--prefix", type=Path, required=True)
    installer.add_argument("--bindir", default="bin")
    installer.add_argument("--runtime-dir", default="libexec/metadtoy")
    args = parser.parse_args()
    try:
        if args.action == "build":
            build(args.source, args.output)
        else:
            install(args.wheels, args.prefix, args.bindir, args.runtime_dir)
    except (ValueError, OSError, KeyError, subprocess.CalledProcessError) as error:
        print("metadtoy installer: %s" % error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
