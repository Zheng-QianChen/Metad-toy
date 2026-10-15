#!/usr/bin/env python3
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


root = Path(__file__).resolve().parent
cases = []
current = None
for line in (root / "results.csv").read_text().splitlines():
    if line.startswith("CASE,"):
        fields = dict(part.split("=", 1) for part in line.split(",")[1:])
        current = {
            "label": f"N={fields['N']}\nnb={float(fields['avg_neighbors']):.0f}, Q{fields['L']}",
            "pipelines": {},
        }
        cases.append(current)
    elif line.startswith("PIPELINE,") and current is not None:
        match = re.match(r"PIPELINE,([^,]+),ms=([0-9.]+)", line)
        if match:
            current["pipelines"][match.group(1)] = float(match.group(2))

series = [
    ("LocalQ-style scatter", "localq_style_atomic"),
    ("Directed edge atomic", "center_q_edge_atomic"),
    ("Pair-fused atomic", "center_q_pair_fused_atomic"),
    ("Pair-fused gather", "center_q_pair_fused_gather"),
]

x = np.arange(len(cases))
width = 0.19
fig, ax = plt.subplots(figsize=(14, 6.5), constrained_layout=True)
for index, (display, key) in enumerate(series):
    values = [c["pipelines"]["original"] / c["pipelines"][key] for c in cases]
    bars = ax.bar(x + (index - 1.5) * width, values, width, label=display)
    if key.startswith("center_q_pair"):
        ax.bar_label(bars, labels=[f"{v:.2f}x" for v in values], padding=2, fontsize=8)

ax.axhline(1.0, color="black", linewidth=1, linestyle="--", label="Original = 1.0x")
ax.set_title("Steinhardt Q4/Q6 CUDA pipeline speedup vs. original center gather")
ax.set_xlabel("Periodic FCC test case (atom count, directed neighbors per atom, Q order)")
ax.set_ylabel("End-to-end Q + dCV/dx speedup (higher is faster)")
ax.set_xticks(x)
ax.set_xticklabels([c["label"] for c in cases])
ax.set_ylim(0.8, 2.2)
ax.grid(axis="y", alpha=0.25)
ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.13))
fig.text(
    0.01,
    0.005,
    "Source: RTX 4080 SUPER benchmark, 2026-09-29; CUDA events, median of 7 batches, 30 iterations/batch.",
    fontsize=8,
)
fig.savefig(root / "pipeline_speedup.png", dpi=180)
