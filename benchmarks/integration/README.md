# Unified-source integration checks

`build.py` and `validate.py` use explicit dependency paths and separate output
directories. They do not replace installed plugins or submit production jobs.
LAMMPS, Kokkos, MPI and Weave must come from a matching build; CUDA architecture
must match the target GPU. The current recorded local check uses LAMMPS 22Jul2025,
CUDA 12.6, OpenMPI 5.0.6 and an RTX 4080 SUPER (SM 89).

Example for this WSL installation, from the repository root:

```bash
python benchmarks/integration/build.py \
  --output "$PWD/build/integration-weave" --profile weave \
  --lammps-source /home/zqc/apps/lammps/lammps-22Jul2025 \
  --host-build /home/zqc/work/Al3Sc-kokkos-joint-20261027/lammps-build \
  --cuda /usr/local/cuda --mpi-prefix /home/zqc/apps/deepmd \
  --weave-source /home/zqc/work/Al3Sc-kokkos-shared-20261028/source/weavemd \
  --weave-library /home/zqc/work/Al3Sc-kokkos-shared-20261028/weave-build/weavebatchplugin.so \
  --cmake /home/zqc/apps/deepmd/bin/cmake \
  --ninja /home/zqc/apps/deepmd/bin/ninja --arch 89 --jobs 4
```

Use the matching Python package, not a different installed LAMMPS wheel:

```bash
/home/zqc/apps/deepmd/bin/python benchmarks/integration/validate.py \
  --phase al3sc --profile weave --np 1 2 \
  --build "$PWD/build/integration-weave" \
  --output "$PWD/build/validation/al3sc" \
  --host-build /home/zqc/work/Al3Sc-kokkos-joint-20261027/lammps-build \
  --mpi-prefix /home/zqc/apps/deepmd \
  --python-package /home/zqc/work/Al3Sc-kokkos-shared-20261028/python \
  --weave-library /home/zqc/work/Al3Sc-kokkos-shared-20261028/weave-build/weavebatchplugin.so \
  --assets /home/zqc/work/Al3Sc-kokkos-shared-20261028/assets
```

Additional phases accept the same runtime paths and a fresh output directory:

- `ctest`: compiled independent tests; pass `--ctest /path/to/ctest`.
- `probe`: real adapter, 12 memory/layout/virial combinations, input reuse,
  cancel, empty batches, one reverse/writeback; pass `--fixture .../al3sc/raw-np1`
  and `--cmake /path/to/cmake` to build the probe with the plugin's ABI flags.
- `global`: Q6, LocalQL6, legacy and Graph DISTANCE, mixed domains and a nonlinear
  graph; pass `--lammps-source /path/to/lammps`.
- `history`: global Q6/LocalQL6/CSS, WT hills, NPT and actual restart; pass the
  same `--fixture` as the probe. This validates ordinary global MetaD history,
  not online shared-family deposition.

Build profiles `kokkos` and `host` do not link Weave. Validation profile
`fallback` uses the Weave binary with `WEAVE_METAD_DEVICE_COMMIT=0`.
`al3sc.py --raw-product` checks the same local product used by the large fixed-bias
run; default mode checks the additional product switch.

Ordinary LAMMPS is a supported host profile. It may be built with
`PKG_KOKKOS=OFF`, `PKG_GPU=OFF` and without Weave; the plugin still requires CUDA
for its CV kernels. Build against that host's matching source, shared library,
integer sizes and MPI ABI. `--weave-source` and `--weave-library` are required
only for the `weave` build profile; `Kokkos_DIR` is supplied only for `kokkos`
and `weave`. Use a separate output directory for each profile:

```bash
python benchmarks/integration/build.py \
  --profile host --output "$PWD/build/plain-plugin" \
  --lammps-source /home/zqc/apps/lammps/lammps-22Jul2025 \
  --host-build "$PWD/build/plain-lammps-20261029/lammps" \
  --cuda /usr/local/cuda --mpi-prefix /home/zqc/apps/deepmd \
  --cmake /home/zqc/apps/deepmd/bin/cmake \
  --ninja /home/zqc/apps/deepmd/bin/ninja --arch 89 --jobs 4

# Pin the Python wrapper to this exact library in a fresh local package.
# This avoids another installed liblammps winning the Python executable's RPATH.
python - <<'PY'
from pathlib import Path
import shutil
package = Path('build/plain-python')
shutil.copytree('/home/zqc/apps/lammps/lammps-22Jul2025/python/lammps',
                package / 'lammps', ignore=shutil.ignore_patterns('__pycache__'))
(package / 'lammps/liblammps.so').symlink_to(
    Path('build/plain-lammps-20261029/lammps/liblammps.so').resolve())
PY

python benchmarks/integration/validate.py \
  --phase al3sc --profile host --np 1 2 \
  --build "$PWD/build/plain-plugin" \
  --output "$PWD/build/plain-validation" \
  --host-build "$PWD/build/plain-lammps-20261029/lammps" \
  --mpi-prefix /home/zqc/apps/deepmd \
  --python-package "$PWD/build/plain-python" \
  --assets /home/zqc/work/Al3Sc-kokkos-shared-20261028/assets
```

The `host` validator passes no `-k on`, `-sf kk` or Weave plugin. Host coordinates
are uploaded for CV evaluation; Host neighbor CSR is uploaded when its contents
change, and force/virial results return through the ordinary LAMMPS adapter.
LocalQL borrows this uploaded device CSR through the same interface as the Kokkos
path. Removing its unused pair-harmonic cache benefits both paths. This profile
does not provide CPU-only CV kernels or the Kokkos path's transfer performance.

The result directories keep command lines, exit codes, plugin SHA256, independent
energy references, finite differences and states for cross-backend comparison.
See [the consolidation record](../../docs/VERSION_CONSOLIDATION_20261028.zh.md)
and [validation scope](../../docs/INTEGRATION_VALIDATION_20261028.zh.md).

For production shared online WT, run `benchmarks/shared_frozen/online.py` with
an AlSc fixture, explicit plugin and LAMMPS arguments. See the
[shared manual](../../docs/manuals/SHARED_BIAS.zh.md).

On the recorded WSL OpenMPI 5.0.6 stack, `early_cuda.py --device 0 test.py ...`
initializes CUDA before Python imports MPI. It provides a sanitizer-compatible
startup while retaining CUDA-aware communication. Select each rank's device
explicitly; this is not a device mapping policy for arbitrary clusters. The
[environment diagnosis](../../docs/SCIENCE_ENV_20261028.zh.md) records the
isolated reproducer, unchanged installed libraries, and tested limitations.
