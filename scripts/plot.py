#!/usr/bin/env python3
"""Generate benchmark charts from CSV output of benchmark.sh."""

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ── Style ──────────────────────────────────────────────────────────────────
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
RED = "#ef4444"
GREEN = "#22c55e"
GREEN_LIGHT = "#bbf7d0"
GREEN_BAND = "#dcfce7"

# ── Load data ──────────────────────────────────────────────────────────────
csv_path = sys.argv[1] if len(sys.argv) > 1 else "results/gset.csv"
out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("charts")
out_dir.mkdir(parents=True, exist_ok=True)

rows = []
with open(csv_path) as f:
    for r in csv.DictReader(f):
        r["nodes"] = int(r["nodes"])
        r["edges"] = int(r["edges"])
        r["time_s"] = float(r["time_s"])
        r["cut_value"] = float(r["cut_value"])
        r["best_known"] = int(r["best_known"])
        r["gap_pct"] = float(r["gap_pct"])
        rows.append(r)

labels = [r["graph"] for r in rows]
nodes = [r["nodes"] for r in rows]
edges = [r["edges"] for r in rows]
cuts = [r["cut_value"] for r in rows]
bests = [r["best_known"] for r in rows]
gaps = [r["gap_pct"] for r in rows]
times = [r["time_s"] for r in rows]

# ── Chart 1: Cut value comparison (grouped bar) ───────────────────────────
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

# ── Chart 2: Gap with expected trend band ─────────────────────────────────
# Expected trend: gap grows roughly with sqrt(n) / density.
# Fit a simple model to define the "expected" band.
nodes_arr = np.array(nodes, dtype=float)
edges_arr = np.array(edges, dtype=float)
gaps_arr = np.array(gaps)

# Sort by node count for the trend line.
sort_idx = np.argsort(nodes_arr)
nodes_sorted = nodes_arr[sort_idx]
gaps_sorted = gaps_arr[sort_idx]

# Fit log-linear trend: gap ~ a * log(n) + b
log_n = np.log(nodes_arr)
coeffs = np.polyfit(log_n, gaps_arr, 1)
trend_fn = np.poly1d(coeffs)

# Generate smooth trend line.
n_smooth = np.linspace(min(nodes), max(nodes), 200)
trend_smooth = trend_fn(np.log(n_smooth))

# Band width: use residual std dev, with a minimum.
residuals = gaps_arr - trend_fn(log_n)
band_width = max(np.std(residuals), 0.3)

fig, ax = plt.subplots(figsize=(10, 5.5))

# Draw expected band.
ax.fill_between(n_smooth,
                np.maximum(trend_smooth - 1.5 * band_width, 0),
                trend_smooth + 1.5 * band_width,
                color=GREEN_BAND, alpha=0.8, label="Expected range", zorder=1)
ax.plot(n_smooth, np.maximum(trend_smooth, 0),
        color=GREEN, linewidth=1.5, linestyle="--", alpha=0.7, label="Trend", zorder=2)

# Plot actual gaps.
for i in range(len(rows)):
    in_band = abs(gaps[i] - trend_fn(np.log(nodes[i]))) <= 1.5 * band_width
    color = BLUE if in_band else RED
    ax.scatter(nodes[i], gaps[i], color=color, s=70, zorder=4, edgecolors="white", linewidths=0.5)
    offset_y = 0.12 if gaps[i] >= trend_fn(np.log(nodes[i])) else -0.25
    ax.annotate(labels[i], (nodes[i], gaps[i]),
                textcoords="offset points", xytext=(6, 4),
                fontsize=9, color="#475569", zorder=5)

ax.set_xlabel("Nodes")
ax.set_ylabel("Gap to Best Known (%)")
ax.set_title("Optimality Gap vs Instance Size")
ax.legend(loc="upper left")
ax.set_ylim(bottom=-0.15)
fig.tight_layout()
fig.savefig(out_dir / "gap_to_best.png", dpi=150)
print(f"Saved {out_dir / 'gap_to_best.png'}")

# ── Chart 3: Scaling — time vs nodes ─────────────────────────────────────
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
