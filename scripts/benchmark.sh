#!/usr/bin/env bash
# Benchmark the BM solver on Gset graphs.
# Outputs CSV to stdout; redirect to a file for plotting.
#
# Usage:
#   ./scripts/benchmark.sh > results/gset.csv

set -euo pipefail
cd "$(dirname "$0")/.."

BIN="./target/release/maxcut-sdp"
cargo build --release >&2

INSTANCES=(G1 G11 G14 G22 G35 G43 G48 G55 G58 G60 G63 G67 G70 G77 G81)

declare -A BEST_KNOWN
while read -r name value; do
    BEST_KNOWN[$name]=$value
done < <(grep -v '^#' scripts/best_known.txt)

# Time budget in seconds by node count.
time_budget() {
    local n=$1
    if   (( n <= 1000 )); then echo 10
    elif (( n <= 2000 )); then echo 30
    elif (( n <= 3000 )); then echo 45
    elif (( n <= 5000 )); then echo 90
    elif (( n <= 7000 )); then echo 150
    elif (( n <= 10000 )); then echo 300
    elif (( n <= 14000 )); then echo 420
    else                        echo 600
    fi
}

echo "graph,nodes,edges,solver,time_s,sdp_value,sdp_certified,cut_value,best_known,gap_pct"

for name in "${INSTANCES[@]}"; do
    f="scripts/graphs/gset/$name.txt"
    if [ ! -f "$f" ]; then
        >&2 echo "missing $f (run scripts/download.py gset)"
        continue
    fi
    bk=${BEST_KNOWN[$name]:?no best known value for $name in scripts/best_known.txt}
    read -r n m < "$f"
    budget=$(time_budget "$n")

    >&2 echo "=== $name (n=$n) ==="
    start=$(date +%s.%N)
    output=$("$BIN" "$f" --solver bm --seed 42 --time "$budget")
    end=$(date +%s.%N)
    time_s=$(awk -v s="$start" -v e="$end" 'BEGIN { printf "%.3f", e - s }')

    sdp=$(grep -E "^SDP (bound|relaxation):" <<< "$output" | awk '{print $3}')
    if grep -q "^SDP bound:" <<< "$output"; then certified=yes; else certified=no; fi
    cut=$(grep "^GW cut value:" <<< "$output" | awk '{print $4}')
    gap=$(awk -v cut="$cut" -v bk="$bk" 'BEGIN { printf "%.2f", 100 * (1 - cut / bk) }')

    echo "${name},${n},${m},bm,${time_s},${sdp},${certified},${cut},${bk},${gap}"
    >&2 echo "  cut=${cut} best_known=${bk} gap=${gap}% time=${time_s}s"
done
