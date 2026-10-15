#!/usr/bin/env bash
set -uo pipefail

here=/home/zqc/work/Metad-toy/benchmarks/q_mpi_20260929
root=${here}/oldplugin_continuous
input=${here}/in.q46_mpi_continuous
lmp=/home/zqc/apps/deepmd/bin/lmp_mpi
mpirun=/home/zqc/apps/deepmd/bin/mpirun

mkdir -p "${root}"
for np in 1 2 4; do
  run_dir=${root}/np${np}
  mkdir -p "${run_dir}"
  cp "${input}" "${run_dir}/in.q46_mpi"
  (
    cd "${run_dir}"
    timeout 120s /usr/bin/time -f "wall_seconds=%e\nmax_rss_kb=%M" -o resource.txt \
      "${mpirun}" --bind-to none -np "${np}" -x CUDA_VISIBLE_DEVICES=0 \
      "${lmp}" -in in.q46_mpi -log log.lammps -screen screen.txt \
      > launcher.txt 2>&1
  )
  status=$?
  printf '%s\n' "${status}" > "${run_dir}/exit_status.txt"
done
