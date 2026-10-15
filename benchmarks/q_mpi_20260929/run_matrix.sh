#!/usr/bin/env bash
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
lmp=/home/zqc/apps/deepmd/bin/lmp_mpi
mpirun=/home/zqc/apps/deepmd/bin/mpirun

for np in 1 2 4; do
  run_dir="${here}/np${np}"
  mkdir -p "${run_dir}"
  cp "${here}/in.q46_mpi" "${run_dir}/in.q46_mpi"
  (
    cd "${run_dir}"
    /usr/bin/time -f "wall_seconds=%e\nmax_rss_kb=%M" -o resource.txt \
      "${mpirun}" --bind-to none -np "${np}" \
      -x CUDA_VISIBLE_DEVICES=0 \
      "${lmp}" -in in.q46_mpi -log log.lammps -screen screen.txt
  )
done

python3 "${here}/compare_mpi.py" | tee "${here}/comparison.csv"
