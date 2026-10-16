# Q4/Q6 multi-MPI GPU test — 2026-09-29

> Historical report from 2026-09-29. Build failures and old-plugin results below describe that snapshot. For the current upload and validation scope, see [CHANGELOG](../../CHANGELOG.md) and [CV communication regression](../cv_communication/README.md).

## Conclusion

The current working tree cannot be tested as an executable because it does not compile. The available plugin binary predates the current Q/LQ refactor by about 20 hours.

The older plugin binary shows two distinct behaviors:

- A single continuous `run` passes with 1, 2, and 4 MPI ranks. At the final biased step, global Q4/Q6, bias energy, per-atom forces, and all six virial/pressure components agree across decompositions.
- Two `run` segments with `displace_atoms` between them pass on 1 rank but fail on 2 and 4 ranks during the first biased force evaluation of the second segment. The error is `STEIN_QL CV error: force is infinity` (the guard actually tests NaN).

This strongly isolates the old-plugin MPI failure to stale Steinhardt environment/neighbor caches across `run` boundaries, rather than the basic forward/reverse communication path used during a continuous run.

No production source file or existing build output was modified. All inputs, logs, and the isolated build tree are under `benchmarks/q_mpi_20260929`.

## Environment

- System: 5324 atoms, FCC 11×11×11 conventional cells, periodic box 44.55 Å per side.
- CVs: Q4.AVE and Q6.AVE, cutoff 3.30 Å, `cutoff_Natoms 12`.
- Neighbor list observed by LAMMPS: full list, 63,888 total entries, exactly 12 neighbors/atom.
- GPU: NVIDIA GeForce RTX 4080 SUPER, 16,376 MiB, driver 591.44.
- MPI: Open MPI 5.0.6 from `/home/zqc/apps/deepmd/bin/mpirun`.
- LAMMPS: 29 Aug 2024.
- Plugin tested: `/home/zqc/work/Metad-toy/build/fix_crystallize_plugin.so`.
- Plugin SHA-256: `1ad46b0e80394f53848d126745dca5ab74ac62a87d6f1c576bda3fdbbf014b69`.
- Plugin timestamp: 2026-09-28 15:24:26 +0800.
- Current `CVs_Steinhardt.cu` timestamp: 2026-09-29 11:45:00 +0800.

`reset_atoms id sort yes` is used before ID-based perturbations, so all MPI decompositions receive the same physical coordinates and atom IDs.

## Continuous single-run correctness test of the available plugin

Input: `in.q46_mpi_continuous`

The simulation starts at timestep 99999 and executes one `run 2`. A hill is deposited at step 100000. Deterministic velocities move the atoms before step 100001, producing nonzero bias forces and virial without crossing a `run` boundary.

Exit status:

- 1 rank: 0
- 2 ranks: 0
- 4 ranks: 0

Final step 100001, identical for all rank counts at the printed 17-digit precision:

- Q4 = `0.19094066586035247`
- Q6 = `0.57420819951512647`
- Bias energy = `0.99985400069006913`
- Per-atom coordinates: maximum difference versus 1 rank = 0
- Per-atom `fx`, `fy`, `fz`: maximum difference versus 1 rank = 0
- Total force L1 norm = `0.00975718104097642`
- Maximum absolute atomic force component = `2.436582301238482e-06`
- Net force components are at the `10^-17` to `10^-19` level.

The 1-rank pressure/virial output at step 100001 is:

`[-4.3472857697824502e-4, -1.1155310887657547e-4, 5.4628168585481448e-4, -7.3470610981803668e-6, -1.6796889824380747e-5, -2.1367291907307788e-5]`

The maximum difference among 1, 2, and 4 ranks is approximately `1.6e-18`.

An additional defect is visible at the initial thermo row (step 99999), before the first `post_force`: pressure/virial values are uninitialized garbage or NaN and depend on MPI rank count. From step 100000 onward they are finite and decomposition-independent.

Timing on one shared GPU:

- 1 rank: LAMMPS loop 0.159227 s; whole process 1.49 s.
- 2 ranks: LAMMPS loop 0.186181 s; whole process 1.64 s.
- 4 ranks: LAMMPS loop 0.376776 s; whole process 2.11 s.

These timings are correctness data only. All MPI ranks share one RTX 4080 SUPER, so they are not a multi-GPU scaling measurement. In this setup, additional ranks add CUDA-context and MPI overhead.

## Split-run failure test of the available plugin

Input: `in.q46_mpi_deterministic`

The first `run 1` deposits a hill at step 100000. `displace_atoms` then changes the coordinates, followed by a second `run 1`.

Exit status:

- 1 rank: 0, reaches step 100001.
- 2 ranks: 1, fails before producing step 100001.
- 4 ranks: 1, fails before producing step 100001.

Before entering the failing second segment, the completed step-100000 global results agree across decompositions:

- Q4 = `0.1909401491799663` (2-rank last-bit difference about `2e-17`).
- Q6 = `0.5742162379923087` (4-rank last-bit difference about `1.1e-16`).
- Bias energy = `0.099999940371662419`.

Both multi-rank cases fail with:

`STEIN_QL CV error: force is infinity, check your system or cv_value.`

The guard is reached after a previous CV has already had the opportunity to write NaN force, so the CV named by the final message is not necessarily the CV that generated the first NaN.

## Highest-confidence cause of the split-run failure

The available binary is older than the current source, but the control-flow pattern in the source explains the observed old-plugin failure:

1. `Steinhardt_env::neigh_lastcall_` is updated after flattening a neighbor list but is not reset when a new `run` rebinds neighbor lists.
2. The flatten-skip condition uses `neighbor->lastcall` and cached state, but not a NeighHub generation, the `NeighList*`, `nlocal`, or `nghost`.
3. The second run begins at the same timestep at which the first run ended. After `displace_atoms`, `last_update_step == ntimestep`, so the timestep-only environment refresh test can also reuse stale coordinates.
4. In multi-MPI runs, ghost local indices can change across setup/rebinding. Reusing flattened old localtags feeds the derivative kernel invalid or zero `q_l` and neighbor counts.
5. The derivative kernel computes `1/(q_l N_b)` without checking localtag range, `N_b > 0`, finite/nonzero `q_l`, or `r² > 0`, so stale indices become NaN instead of a source-level diagnostic.

Recommended repair order:

1. Add a NeighHub generation/epoch and include it, the `NeighList*`, `nlocal`, and `nghost` in the Steinhardt cache key; alternatively force-invalidate every Steinhardt environment from `Fix::init()`.
2. Force coordinate/environment refresh at every new `run`, even when the timestep has not advanced.
3. Add source checks in the derivative kernels for neighbor localtag range, positive neighbor count, finite/nonzero `q_l`, and positive `r²`.

## Current working tree build result

An isolated Release build with ML CVs disabled was attempted in `build_current`. Disabling ML does not remove Q/LQ, virial, neighbor, or MPI code. Compilation fails before a plugin is produced.

Primary blockers:

- `include/CVsets/CV_Steinhardt.h:326-327`: `Steinhardt_env<L>` is used as a template, but `Steinhardt_env` is not a template and `L` is undefined in that class.
- `src/cvs/CVs_Steinhardt.cu:244`: `req` is undefined in the base environment constructor.
- `src/cvs/CVs_Steinhardt.cu:648` and nearby: Q base environment code references LQ-only members such as `d_neigh_in_switching` and calculated-prefix arrays.
- `src/cvs/CVs_Steinhardt.cu:672-681`: the kernel launch does not match the signature at 1356-1371; three mask arguments are missing and subsequent arguments are shifted.
- `src/cvs/CVs_Steinhardt.cu:1390`: the Q environment kernel references `d_LQ_mask`, which is not in its parameter list or scope.

There is also an unsafe prefix/index layout in the current source that must be fixed after compilation is restored:

- `h_group_indices` is allocated for `nlocal` entries at line 408, but line 501 uploads `atom_all = nlocal + nghost` entries from it (host over-read).
- `h_group_numneigh` is allocated for `atom_all + 1`, but lines 541-547 fill only the first `group_count + 1` entries.
- Lines 548 and 550 upload/read through `atom_all`, including uninitialized prefix values.
- Lines 573-592 then interpret the same prefix as localtag-order data.

Choose one invariant and use it everywhere: either a `group_count + 1` prefix indexed by center ordinal, or an `atom_all + 1` prefix indexed by localtag. The present code mixes both.

## Artifacts

- `build_current/build.log`: complete current-source compiler output.
- `oldplugin_continuous/np{1,2,4}`: passing continuous-run logs and dumps.
- `oldplugin_deterministic/np{1,2,4}`: split-run pass/failure reproduction.
- `in.q46_mpi_continuous`: continuous-run test input.
- `in.q46_mpi_deterministic`: split-run test input.
