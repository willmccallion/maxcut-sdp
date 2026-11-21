#!/usr/bin/env python3
"""Generate benchmark charts from CSV output of benchmark.sh."""

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
})
BLUE = "#2563eb"
GRAY = "#94a3b8"

csv_path = sys.argv[1] if len(sys.argv) > 1 else "results/gset.csv"
out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("charts")
out_dir.mkdir(parents=True, exist_ok=True)

rows = []
with open(csv_path) as f:
    for r in csv.DictReader(f):
        r["nodes"] = int(r["nodes"])
        r["time_s"] = float(r["time_s"])
        r["cut_value"] = float(r["cut_value"])
        r["best_known"] = int(r["best_known"])
        r["gap_pct"] = float(r["gap_pct"])
        rows.append(r)

labels = [r["graph"] for r in rows]
nodes = [r["nodes"] for r in rows]
cuts = [r["cut_value"] for r in rows]
bests = [r["best_known"] for r in rows]
gaps = [r["gap_pct"] for r in rows]
times = [r["time_s"] for r in rows]

fig, ax = plt.subplots(figsize=(12, 5))
x = range(len(labels))
w = 0.35
ax.bar([i - w/2 for i in x], bests, w, label="Best Known", color=GRAY, zorder=3)
ax.bar([i + w/2 for i in x], cuts, w, label="maxcut-sdp", color=BLUE, zorder=3)
ax.set_xticks(list(x))
ax.set_xticklabels(labels, rotation=45, ha="right")
ax.set_ylabel("Cut Value")
ax.set_title("maxcut-sdp vs Best Known on Gset Instances")
ax.legend()
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
fig.tight_layout()
fig.savefig(out_dir / "cut_comparison.png", dpi=150)
print(f"Saved {out_dir / 'cut_comparison.png'}")

fig, ax = plt.subplots(figsize=(10, 5.5))
ax.scatter(nodes, gaps, color=BLUE, s=70, zorder=4, edgecolors="white", linewidths=0.5)
for i, lbl in enumerate(labels):
    ax.annotate(lbl, (nodes[i], gaps[i]), textcoords="offset points", xytext=(6, 4),
                fontsize=9, color="#475569", zorder=5)
ax.set_xlabel("Nodes")
ax.set_ylabel("Gap to Best Known (%)")
ax.set_title("Optimality Gap vs Instance Size")
ax.set_ylim(bottom=-0.15)
fig.tight_layout()
fig.savefig(out_dir / "gap_to_best.png", dpi=150)
print(f"Saved {out_dir / 'gap_to_best.png'}")

fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(nodes, times, s=80, color=BLUE, zorder=3)
for i, lbl in enumerate(labels):
    ax.annotate(lbl, (nodes[i], times[i]), textcoords="offset points",
                xytext=(6, 4), fontsize=8, color="#64748b")
ax.set_xlabel("Nodes")
ax.set_ylabel("Time (s)")
ax.set_title("Solver Runtime vs Instance Size")
fig.tight_layout()
fig.savefig(out_dir / "scaling.png", dpi=150)
print(f"Saved {out_dir / 'scaling.png'}")

plt.close("all")
print("Done.")
