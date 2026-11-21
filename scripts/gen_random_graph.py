#!/usr/bin/env python3
"""Generate a random sparse graph in Biq Mac format for benchmarking.

Usage: gen_random_graph.py N [P [SEED]]

Each pair of the N vertices is joined with probability P (default 5/N,
about five neighbours per vertex) by an edge of integer weight 1 to 10.
"""
import random
import sys


def gen_erdos_renyi(n, p, seed=42):
    random.seed(seed)
    edges = []
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            if random.random() < p:
                w = random.randint(1, 10)
                edges.append((i, j, w))
    return edges


if __name__ == "__main__":
    n = int(sys.argv[1])
    p = float(sys.argv[2]) if len(sys.argv) > 2 else 5.0 / n
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 42
    edges = gen_erdos_renyi(n, p, seed)
    print(f"{n} {len(edges)}")
    for i, j, w in edges:
        print(f"{i} {j} {w}")
