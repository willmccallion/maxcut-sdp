#!/usr/bin/env python3
"""Download Max-Cut benchmark instances from BiqMac and Gset."""

import os
import sys
import urllib.request
import urllib.error

GRAPH_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "graphs")

# BiqMac Library  https://biqmac.aau.at/biqmaclib.html
# Format: first line "n m", then "i j w" (1-indexed) — our parse_biqmac.
BIQMAC_RUDY = "https://biqmac.aau.at/library/mac/rudy"
BIQMAC_ISING = "https://biqmac.aau.at/library/mac/ising"

RUDY_INSTANCES = {
    # g05_N.K = unweighted, ~50% density, N vertices, instance K
    "g05_60.0": 60,
    "g05_60.1": 60,
    "g05_80.0": 80,
    "g05_80.1": 80,
    "g05_100.0": 100,
    "g05_100.1": 100,
    # pm1s = ±1 weights, sparse
    "pm1s_40.0": 40,
    "pm1s_80.0": 80,
    "pm1s_100.0": 100,
    # pw = positive weights, varying density
    "pw01_100.0": 100,
    "pw05_100.0": 100,
    "pw09_100.0": 100,
    # w = general weights
    "w05_100.0": 100,
}

ISING_INSTANCES = {
    # 2D toroidal grid  n = k*k
    "t2g10_5555": 100,
    "t2g15_5555": 225,
    "t2g20_5555": 400,
    # 3D toroidal grid  n = k*k*k
    "t3g5_5555": 125,
    # Ising spin glass
    "ising2.5-100_5555": 100,
    "ising3.0-100_5555": 100,
}

# Gset  https://web.stanford.edu/~yyye/yyye/Gset/
# Larger instances (n=800–10000). Same sparse format (1-indexed).
GSET_BASE = "https://web.stanford.edu/~yyye/yyye/Gset"

GSET_INSTANCES = {
    "G1": 800,
    "G2": 800,
    "G3": 800,
    "G11": 800,
    "G14": 800,
    "G22": 2000,
    "G43": 1000,
    "G48": 3000,
    "G55": 5000,
}

# Known best cut values for verification.
KNOWN_BEST = {
    "g05_60.0": 536,
    "g05_60.1": 532,
    "g05_80.0": 934,
    "g05_80.1": 942,
    "g05_100.0": 1430,
    "G1": 11624,
    "G2": 11620,
    "G3": 11622,
    "G11": 564,
    "G14": 3064,
    "G22": 13359,
    "G43": 6660,
}


def download(url: str, dest: str) -> bool:
    """Download a URL to a local file. Returns True on success."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "gw-maxcut-bench/0.1"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = resp.read()
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(dest, "wb") as f:
            f.write(data)
        return True
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as e:
        print(f"FAILED ({e})", file=sys.stderr)
        return False


def fetch_set(label: str, base_url: str, instances: dict, subdir: str):
    """Download a set of instances."""
    print(f"\n=== {label} ({len(instances)} instances) ===")
    ok = fail = skip = 0
    for name, n in instances.items():
        dest = os.path.join(GRAPH_DIR, subdir, name + ".txt")
        if os.path.exists(dest):
            print(f"  [skip] {name:25s} n={n}")
            skip += 1
            continue
        url = f"{base_url}/{name}"
        print(f"  [GET]  {name:25s} n={n:>5} ... ", end="", flush=True)
        if download(url, dest):
            print("ok")
            ok += 1
        else:
            fail += 1
    print(f"  Result: {ok} downloaded, {skip} skipped, {fail} failed")


def list_downloaded():
    """List all downloaded graphs with sizes."""
    print("\n=== Downloaded instances ===")
    total = 0
    for root, _dirs, files in sorted(os.walk(GRAPH_DIR)):
        for f in sorted(files):
            if not f.endswith(".txt"):
                continue
            path = os.path.join(root, f)
            rel = os.path.relpath(path, GRAPH_DIR)
            with open(path) as fp:
                header = fp.readline().strip().split()
            if len(header) >= 2:
                n, m = header[0], header[1]
                stem = f.removesuffix(".txt")
                opt = KNOWN_BEST.get(stem, "")
                opt_str = f"  best={opt}" if opt else ""
                print(f"  {rel:45s}  n={n:>5s}  m={m:>7s}{opt_str}")
                total += 1
    if total == 0:
        print("  (none)")
    print(f"\nTotal: {total} instances in {GRAPH_DIR}/")


def main():
    import argparse

    p = argparse.ArgumentParser(description="Download Max-Cut benchmark graphs.")
    p.add_argument(
        "source",
        nargs="?",
        default="small",
        choices=["small", "rudy", "ising", "gset", "all"],
        help="small = rudy n<=100 (default), rudy = all rudy, all = everything",
    )
    p.add_argument("--list", action="store_true", help="List downloaded instances and exit")
    args = p.parse_args()

    if args.list:
        list_downloaded()
        return

    if args.source in ("small", "rudy", "all"):
        if args.source == "small":
            subset = {k: v for k, v in RUDY_INSTANCES.items() if v <= 100}
        else:
            subset = RUDY_INSTANCES
        fetch_set("BiqMac / Rudy", BIQMAC_RUDY, subset, "rudy")

    if args.source in ("small", "ising", "all"):
        if args.source == "small":
            subset = {k: v for k, v in ISING_INSTANCES.items() if v <= 100}
        else:
            subset = ISING_INSTANCES
        fetch_set("BiqMac / Ising", BIQMAC_ISING, subset, "ising")

    if args.source in ("gset", "all"):
        print("\n  NOTE: Gset graphs are large (n=800–5000).")
        print("  Dense SDP solver handles n<~100; larger needs Burer-Monteiro.\n")
        fetch_set("Gset", GSET_BASE, GSET_INSTANCES, "gset")

    list_downloaded()
    print("\nExample:")
    print("  cargo run --release -- --input scripts/graphs/rudy/g05_60.0.txt")


if __name__ == "__main__":
    main()
