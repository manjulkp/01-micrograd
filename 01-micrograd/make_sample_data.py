"""Generate a synthetic two-moons dataset, for trying train.py before drawing your own."""

import argparse
import math
import random

from micrograd.data import save_points


def make_moons(n_points=100, noise=0.1, seed=0):
    rng = random.Random(seed)
    points = []
    for i in range(n_points):
        angle = rng.uniform(0, math.pi)
        if i % 2 == 0:
            x, y, label = math.cos(angle), math.sin(angle), -1
        else:
            x, y, label = 1 - math.cos(angle), 0.5 - math.sin(angle), 1
        x += rng.gauss(0, noise)
        y += rng.gauss(0, noise)
        # Centre the moons so that they fit the drawing tool's -1 to 1 canvas.
        points.append(((x - 0.5) * 0.6, (y - 0.25) * 0.6, label))
    return points


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/sample_moons.csv")
    parser.add_argument("--points", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    save_points(args.out, make_moons(args.points, seed=args.seed))
    print(f"Wrote {args.points} points to {args.out}")
