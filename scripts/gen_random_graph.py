"""Generate random sparse graphs in Biq Mac format for benchmarking."""
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
    p = float(sys.argv[2]) if len(sys.argv) > 2 else 5.0 / n  # ~5 neighbors per vertex
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 42
    edges = gen_erdos_renyi(n, p, seed)
    print(f"{n} {len(edges)}")
    for i, j, w in edges:
        print(f"{i} {j} {w}")
