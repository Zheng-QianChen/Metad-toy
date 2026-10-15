# Q4/Q6 GPU force-layout benchmark

This is an isolated CUDA microbenchmark for the Steinhardt Q4/Q6 code in
`/home/zqc/work/Metad-toy`. It includes the project's real
`CV_Steinhardt_math.h`; the production files are not copied or modified.

The generated systems are periodically wrapped, slightly perturbed FCC cells:

- 11x11x11 cells: 5324 atoms
- 16x16x16 cells: 16384 atoms
- cutoff 0.82: about 12 directed neighbors per atom
- cutoff 1.30: about 42 directed neighbors per atom

Measured Q layouts:

- `center_original`: the current one-thread-per-center Q kernel structure
- `localq_style`: the LocalQ-like two-kernel structure: center-owned qlm loop,
  followed by a separate scalar q finalization kernel
- `edge_parallel`: one thread per directed edge, followed by a center CSR reduction

Measured derivative layouts:

- `original_gather`: current Q4/Q6 gather formula; no atomics
- `localq_atomic`: LocalQ-style center scatter; neighbor and final center atomics
- `center_split`: center scatter with owned center output; neighbor-only atomics
- `warp_center_split`: one warp per center; neighbor-only atomics
- `edge_atomic`: one thread per edge; atomics at both endpoints
- `edge_reverse_gather`: edge values plus forward/reverse CSR gather; no atomics
- `pair_fused_atomic`: half-list; one combined helper per undirected pair and
  atomics to both endpoints
- `pair_fused_gather`: half-list plus signed incidence CSR; no atomics

Timing uses CUDA events. Allocation, graph construction, host/device copies and
MPI are excluded. Required clears and combine kernels are included. Each result
is the median of seven batches; the CLI repetition count controls the number of
iterations in each batch.

`PIPELINE` rows time compatible Q and derivative layouts together. `FDIFF`
compares nine analytic gradient components against central coordinate finite
differences of the global mean Q.

This is a controlled kernel benchmark, not a timing of the complete LAMMPS fix.
It uses CSR storage instead of the production Q kernel's fixed-stride
`cutoff_Natoms` buffer. The force timings isolate `dcv/dx`: they omit the current
kernel's per-edge global `d_dYlm_dr` scratch traffic, per-atom virial writes,
group/ghost handling, MPI, neighbor rebuilding and host copies. The scatter
kernels deliberately call the project's combined gradient helper with its
neighbor coefficient set to zero; a generated source-only helper can remove
those zero-coefficient loads and may be faster.

Run under WSL:

```bash
./run.sh /home/zqc/work/Metad-toy 40
```
