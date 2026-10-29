#!/usr/bin/env python3
"""Exercise CMake CLI installation on the designated validation host.

Provide --source (repository snapshot), --work and --cmake. Pip can run offline
using PIP_NO_INDEX/PIP_FIND_LINKS. All installs stay under --work.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--work", required=True, type=Path)
    parser.add_argument("--cmake", default="cmake")
    args = parser.parse_args()
    work = args.work.resolve()
    work.mkdir(parents=True, exist_ok=True)
    source = work / "source copy"
    shutil.copytree(args.source, source, dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    build = work / "build cli"
    configured_prefix = work / "unused prefix"
    prefix = work / "installed with spaces"
    outside = work / "outside"
    outside.mkdir(exist_ok=True)
    checks = []
    def run(command, success=True, **kwargs):
        print("+", " ".join(map(str, command)), flush=True)
        result = subprocess.run(list(map(str, command)), text=True, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, **kwargs)
        print(result.stdout, flush=True)
        if (result.returncode == 0) != success:
            raise AssertionError("unexpected exit %d" % result.returncode)
        return result.stdout
    def passed(name):
        checks.append(name)
        print("PASS:", name, flush=True)

    run([args.cmake, "-S", source, "-B", build, "-DZQC_CLI_ONLY=ON",
         "-DPython3_EXECUTABLE="+sys.executable, "-DCMAKE_INSTALL_PREFIX="+str(configured_prefix),
         "-DCMAKE_CXX_COMPILER=/deliberately/unavailable/compiler"])
    cache = (build/"CMakeCache.txt").read_text()
    assert "CMAKE_CUDA_COMPILER:" not in cache
    assert "CMAKE_CXX_COMPILER_ABI:" not in cache
    passed("CLI-only configuration requires no C++/CUDA compiler")
    run([args.cmake, "--build", build, "--target", "metadtoy-cli"])
    stamp = build/"tools/cli-wheel/ready.stamp"
    initial_time = stamp.stat().st_mtime_ns
    run([args.cmake, "--build", build, "--target", "metadtoy-cli"])
    assert initial_time == stamp.stat().st_mtime_ns
    passed("unchanged builds reuse the packaged wheel")

    run([args.cmake, "--install", build, "--component", "cli", "--prefix", prefix])
    executable = prefix/"bin/metadtoy"
    assert executable.is_file() and not configured_prefix.exists()
    passed("install-time prefix override and paths containing spaces")
    env = os.environ.copy()
    env["PATH"] = str(executable.parent)+os.pathsep+env["PATH"]
    env["PYTHONPATH"] = str(outside)
    (outside/"metadtoy_cli.py").write_text("raise RuntimeError('untrusted current directory')\n")
    help_text = run(["metadtoy", "--help"], cwd=outside, env=env)
    assert all(word in help_text for word in ("hills", "fes", "trajectory", "dataset"))
    assert "0.1.0" in run(["metadtoy", "--version"], cwd=outside, env=env)
    passed("bare metadtoy works outside the repository without activation")

    colvar = outside/"COLVAR"
    colvar.write_text("# step value rbias\n0 .2 0\n1 .8 0\n2 .2 0\n3 .8 0\n")
    fes = outside/"fes"
    result = run(["metadtoy", "fes", "reconstruct", colvar, "--kbt", "1",
                  "--bins", "2", "--range", "1", "0", "1", "--out", fes, "--progress", "always"],
                 cwd=outside, env=env)
    assert "100.0%" in result
    assert (outside/"fes.png").stat().st_size > 0
    metadata = json.loads((outside/"fes.json").read_text())
    assert metadata["observations_selected"] == 4
    passed("installed dependencies execute FES and progress output")

    # An unrelated executable must remain intact even if the user chooses its prefix.
    conflict = work/"conflicting prefix"
    (conflict/"bin").mkdir(parents=True, exist_ok=True)
    (conflict/"bin/metadtoy").write_text("#!/bin/sh\necho unrelated\n")
    run([args.cmake, "--install", build, "--component", "cli", "--prefix", conflict], success=False)
    assert (conflict/"bin/metadtoy").read_text() == "#!/bin/sh\necho unrelated\n"
    assert not (conflict/"libexec").exists()
    passed("unrelated command is preserved")
    destination_env = dict(os.environ, DESTDIR=str(work/"staging"))
    run([args.cmake, "--install", build, "--component", "cli", "--prefix", work/"staged"],
        success=False, env=destination_env)
    assert not (work/"staged").exists()
    passed("unsupported DESTDIR is rejected before installation")

    # CMake installs the newest source even when the development version stays unchanged.
    cli_source = source/"tools/metadtoy_cli/cli.py"
    cli_source.write_text(cli_source.read_text().replace("Offline MetaD-toy analysis", "INSTALL-UPDATE Offline MetaD-toy analysis"))
    run([args.cmake, "--build", build, "--target", "metadtoy-cli"])
    run([args.cmake, "--install", build, "--component", "cli", "--prefix", prefix])
    assert "INSTALL-UPDATE" in run(["metadtoy", "--help"], cwd=outside, env=env)
    passed("same-version source update is installed")

    hidden_source = work/"hidden source"
    source.rename(hidden_source)
    try:
        assert "0.1.0" in run(["metadtoy", "--version"], cwd=outside, env=env)
    finally:
        hidden_source.rename(source)
    passed("installed runtime does not depend on the source checkout")

    # The optional component also coexists with the existing Host build.
    host_build = work/"host build"
    run([args.cmake, "-S", source, "-B", host_build, "-DZQC_ENABLE_CUDA=OFF",
         "-DZQC_BUILD_TESTS=OFF", "-DZQC_BUILD_CLI=ON", "-DPython3_EXECUTABLE="+sys.executable])
    run([args.cmake, "--build", host_build, "--target", "metadtoy-cli", "zqc_bias", "-j2"])
    passed("CLI target coexists with the Host bias library")
    result = {"passed": len(checks), "checks": checks, "python": sys.version}
    (work/"installation-results.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
