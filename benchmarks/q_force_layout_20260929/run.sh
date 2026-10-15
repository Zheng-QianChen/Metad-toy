#!/usr/bin/env bash
set -euo pipefail

project_root="${1:-/home/zqc/work/Metad-toy}"
repetitions="${2:-40}"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

/usr/local/cuda/bin/nvcc \
  -O3 -std=c++17 -arch=sm_89 -lineinfo \
  -I"${project_root}/include/CVsets" \
  "${here}/q_force_layout_bench.cu" \
  -o "${here}/q_force_layout_bench"

"${here}/q_force_layout_bench" "${repetitions}" | tee "${here}/results.csv"
