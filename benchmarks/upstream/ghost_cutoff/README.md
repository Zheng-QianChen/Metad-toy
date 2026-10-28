# LAMMPS custom ghost cutoff issue

状态：已于 2026-10-23 通过 WSL GitHub CLI 提交为 [LAMMPS #5262](https://github.com/lammps/lammps/issues/5262)。已回读核对，公开正文与用户删改后的 ISSUE.md 一致。

已提交标题：[BUG] Custom neighbor cutoff with REQ_GHOST leaves ghost-center rows at the pair-derived cutoff

正文：[ISSUE.md](ISSUE.md)。独立 CPU fix 探针：[ghost_cutoff_probe.cpp](ghost_cutoff_probe.cpp)。输入：[in.ghost_cutoff](in.ghost_cutoff)。
正文命令已直接运行：LAMMPS 29 Aug 2024、2 ranks，pair cutoff 4.0 时 ghost_missing=1，5.0 时 ghost_missing=0；两组 owned_missing=0。
当前 develop 只检查了源码。适配器 fallback 继续保留，不依赖上游修复。先请维护者确认请求语义，再考虑回归测试和 PR。
