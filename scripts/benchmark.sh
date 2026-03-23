#!/usr/bin/env bash
# Benchmark BM solver on Gset graphs.
# Outputs CSV to stdout; redirect to a file for plotting.
#
# Usage:
#   ./scripts/benchmark.sh > results/gset.csv

set -euo pipefail
cd "$(dirname "$0")/.."

BIN="./target/release/gw-maxcut"
cargo build --release 2>/dev/null

# Best-known values from literature (BLS, CirCut, DSDP, etc.)
declare -A BEST_KNOWN=(
    [G1]=11624   [G11]=564    [G14]=3064   [G22]=13359
    [G35]=7687   [G43]=6660   [G48]=6000   [G55]=10299
    [G58]=19293  [G60]=14188  [G63]=27047  [G67]=6950
    [G70]=9591   [G77]=9940   [G81]=14060
)

# Time budget tiers by node count.
time_budget() {
    local n=$1
    if   (( n <= 1000 )); then echo "--time 10"
    elif (( n <= 2000 )); then echo "--time 30"
    elif (( n <= 3000 )); then echo "--time 45"
    elif (( n <= 5000 )); then echo "--time 90"
    elif (( n <= 7000 )); then echo "--time 150"
    elif (( n <= 10000 )); then echo "--time 300"
    elif (( n <= 14000 )); then echo "--time 420"
    else                        echo "--time 600"
    fi
}

echo "graph,nodes,edges,solver,time_s,sdp_bound,cut_value,best_known,gap_pct"

for f in scripts/graphs/gset/G{1,11,14,22,35,43,48,55,58,60,63,67,70,77,81}; do
    [ -f "$f" ] || continue
    name=$(basename "$f")
    bk=${BEST_KNOWN[$name]:-0}

    # Read node count from first line to pick time budget.
    n=$(head -1 "$f" | awk '{print $1}')
    budget=$(time_budget "$n")

    >&2 echo "=== $name (n=$n) ==="
    output=$($BIN "$f" --solver bm --seed 42 $budget 2>&1)

    m=$(echo "$output" | grep "^Graph:" | sed 's/.*, \([0-9]*\) edges.*/\1/')
    sdp=$(echo "$output" | grep "SDP bound:" | awk '{print $3}')
    cut=$(echo "$output" | grep "GW cut value:" | awk '{print $4}')
    time_str=$(echo "$output" | grep "Total time:" | awk '{print $3}')

    # Convert time string to seconds.
    if echo "$time_str" | grep -q "ms$"; then
        time_s=$(echo "$time_str" | sed 's/ms$//' | awk '{printf "%.3f", $1 / 1000}')
    else
        time_s=$(echo "$time_str" | sed 's/s$//')
    fi

    # Gap to best known (%).
    gap=$(awk "BEGIN { printf \"%.2f\", 100 * (1 - ${cut%.000000} / $bk) }")

    echo "${name},${n},${m},bm,${time_s},${sdp},${cut},${bk},${gap}"
    >&2 echo "  cut=${cut} best_known=${bk} gap=${gap}% time=${time_s}s"
done
