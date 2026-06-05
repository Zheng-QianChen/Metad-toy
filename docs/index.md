# MetaD-toy: High-Performance Metadynamics Plugin for LAMMPS

Welcome to the official documentation of **MetaD-toy**, an advanced, non-intrusive metadynamics (MetaD) sampling plugin tailor-made for the LAMMPS molecular dynamics framework. 

This plugin is specifically engineered to accelerate the simulation of **solidification nucleation, phase transitions, and polymorph selection** in complex multi-component metallic systems (e.g., Al-based or W-based alloys).

---

## 🚀 Key Features

* **GPU Acceleration (CUDA Enabled)**: Highly parallelized compute kernels for local order parameters ($Q_4, Q_6$) and free-energy deposition grid calculations.
* **Smart Memory Topologies**: Supports both Dense Uniform Grids and **Sparse Hash Memory Maps** to prevent VRAM overflow during long-lifetime, high-dimensional simulations.
* **Decoupled Data Bus**: Built-in `compute metad/atom` infrastructure via a polymorphic routing design to achieve $O(N)$ data extraction throughput for post-processing.
* **Seamless i18n Handshake**: Full native integration with modern simulation analysis workflows.

---

## 🛠️ Quick Start

To mount the plugin in your LAMMPS simulation script (`in.file`), simply load the dynamic link library and define the `metad` fix:

```lammps
# 1. Load plugin
plugin load ${METAD_PLUGIN_PATH}/fix_crystallize_plugin.so

# 2. Initialize 2D Metadynamics
fix m all metad GAUSSIAN 0.003 0.05 10.0 PACE 1 CV_dim 2 ...
```