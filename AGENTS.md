# AGENTS.md

## Cursor Cloud specific instructions

This repo is **Metad-toy**, a CUDA + MPI LAMMPS metadynamics plugin. It builds to a single shared
object `fix_crystallize_plugin.so` that is loaded into LAMMPS via `plugin load`. See `README.md`
and `makecommand` for the canonical build invocation, and `docs/manuals/User_Manual.zh.md` for the
`fix metad` syntax.

### Hardware limitation (important)
- Every source file is CUDA (`src/**/*.cu`), and the plugin performs `cudaMalloc`/kernel launches
  during a run. **Running a simulation requires an NVIDIA GPU.**
- Cursor Cloud VMs have **no GPU** (`cudaGetDeviceCount` → "no CUDA-capable device is detected").
  The plugin still **compiles** and **loads into LAMMPS** here, but `run` aborts as `fix metad`
  begins GPU initialization. Build/lint/load can be validated; full metadynamics runs and the
  `test/**/run.sh` cases cannot be executed without GPU hardware.

### External dependencies (not in this repo, provisioned in the environment)
- CUDA Toolkit 12.6 at `/usr/local/cuda` (installed via NVIDIA apt repo). CUDA 12.0 is too old:
  the code uses `cuda::std::memcpy` (`<cuda/std/cstring>`), added in a later toolkit.
- OpenMPI (headers at `/usr/lib/x86_64-linux-gnu/openmpi/include`; wrappers `/usr/bin/mpicxx`,
  `/usr/bin/mpicc`).
- `gcc-12`/`g++-12`: nvcc's host compiler. The system default `g++` is 13; CUDA 12.6 accepts
  gcc-13 too, but g++-12 is the safe host compiler.
- **LAMMPS (stable_29Aug2024)** built once as a shared library. It is a large external dependency
  and is intentionally **not** in the startup update script (no build steps there). If it is
  missing (e.g. a cold VM without the snapshot), rebuild it once:
  ```bash
  git clone --depth 1 --branch stable_29Aug2024 https://github.com/lammps/lammps.git ~/apps/lammps-29Aug2024
  cd ~/apps/lammps-29Aug2024 && mkdir -p build && cd build
  cmake ../cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_MPI=ON -DBUILD_SHARED_LIBS=ON \
    -DLAMMPS_EXCEPTIONS=ON -DPKG_PLUGIN=ON -DPKG_MANYBODY=ON -DPKG_MEAM=ON \
    -DPKG_EXTRA-COMPUTE=ON -DPKG_MOLECULE=ON -DPKG_KSPACE=ON \
    -DCMAKE_CXX_COMPILER=mpicxx -DCMAKE_C_COMPILER=mpicc
  make -j"$(nproc)"
  ```
  This yields `~/apps/lammps-29Aug2024/build/liblammps.so` and the `lmp` binary. `PKG_PLUGIN` is
  required for `plugin load`; MEAM/MANYBODY/KSPACE cover the `test/` input files.

### Building the plugin
- The root `CMakeLists.txt` hardcodes `set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")`, so
  `/usr/local/cuda` must point at the real CUDA 12.6 toolkit (the NVIDIA deb sets this via
  `/etc/alternatives/cuda`; the update script also symlinks it as a fallback). Do not rely on
  `-DCMAKE_CUDA_COMPILER` alone — the in-file `set()` overrides it.
- `USE_ML_CVS` defaults ON and then hard-requires ONNX Runtime + cuDNN + a Conda env (`ORT_HOME`,
  `CONDA_PREFIX`). None of that is installed here, so **build with `-DUSE_ML_CVS=OFF`**. The ML CV
  path is unused scaffolding (`mlcvs_test()` is never called by the fix).
- `ENABLE_DEBUG` defaults ON (slow `-O0 -g -G` build). Pass `-DENABLE_DEBUG=OFF` for an optimized
  build.
- Full configure + build (run from repo root):
  ```bash
  export LAMMPS_SOURCE_DIR=~/apps/lammps-29Aug2024/src
  export LAMMPS_LIB_DIR=~/apps/lammps-29Aug2024/build
  export MPI_HOME=/usr/lib/x86_64-linux-gnu/openmpi
  export CLEAN_HOST_COMPILER=/usr/bin/g++-12
  rm -rf build && mkdir build && cd build
  cmake .. \
    -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
    -DCMAKE_CXX_COMPILER=/usr/bin/mpicxx -DCMAKE_C_COMPILER=/usr/bin/mpicc \
    -DCMAKE_CUDA_HOST_COMPILER=$CLEAN_HOST_COMPILER \
    -DLAMMPS_SOURCE_DIR=$LAMMPS_SOURCE_DIR -DLAMMPS_LIB_DIR=$LAMMPS_LIB_DIR \
    -DMPI_HOME=$MPI_HOME -DCLEAN_HOST_COMPILER=$CLEAN_HOST_COMPILER \
    -DUSE_ML_CVS=OFF -DCMAKE_CUDA_ARCHITECTURES=89
  make -j"$(nproc)"
  ```
  Note: `MPI_HOME` must be the multiarch OpenMPI prefix (has `include/mpi.h`); the mpi wrappers
  live in `/usr/bin`, not `$MPI_HOME/bin`, so pass the compilers by full path as above.
- `CMAKE_CUDA_ARCHITECTURES=89` targets Ada (RTX 40xx). nvcc cross-compiles this fine without a
  local GPU; change it to match the deployment GPU.

### Verifying the build without a GPU
- Load the freshly built plugin into LAMMPS (no GPU needed for load/registration):
  ```bash
  export LD_LIBRARY_PATH=~/apps/lammps-29Aug2024/build:/usr/local/cuda/lib64:$LD_LIBRARY_PATH
  printf 'plugin load /workspace/build/fix_crystallize_plugin.so\nplugin list\n' > /tmp/in.load
  ~/apps/lammps-29Aug2024/build/lmp -in /tmp/in.load
  ```
  Success shows `fix style plugin metad` and `compute style plugin metad/atom` registered.

### Tests
- There is no unit-test suite. `test/**/run.sh` are Slurm + `mpirun lmp` metadynamics jobs that
  require a GPU and pre-generated potential tables (`test/DISTANCE/potential/`, MEAM files in
  `test/potential_testlib/`); they cannot run in this GPU-less environment.
- `test/plot_hills.py` / `tools/plot_hills.py` are Python post-processors for `HILLS` output
  (need `numpy`/`matplotlib`).
