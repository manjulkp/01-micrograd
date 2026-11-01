"""Read, write and split point datasets stored as CSV with columns x,y,label."""

import csv
import random
from pathlib import Path

VALID_LABELS = {-1, 1}


def save_points(path, points):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y", "label"])
        writer.writerows((f"{x:.4f}", f"{y:.4f}", label) for x, y, label in points)


def load_points(path):
    with Path(path).open(newline="") as f:
        points = [
            (float(row["x"]), float(row["y"]), int(row["label"]))
            for row in csv.DictReader(f)
        ]
    bad = {label for _, _, label in points} - VALID_LABELS
    if bad:
        raise ValueError(f"{path}: labels must be -1 or 1, found {sorted(bad)}")
    return points


def train_test_split(points, test_fraction=0.2, seed=0):
    """Split so that both classes appear in both sets, in their original ratio."""
    rng = random.Random(seed)
    train, test = [], []
    for label in sorted(VALID_LABELS):
        group = [p for p in points if p[2] == label]
        if len(group) < 2:
            raise ValueError(
                f"need at least 2 points of each class, found {len(group)} with label {label}"
            )
        rng.shuffle(group)
        n_test = max(1, round(len(group) * test_fraction))
        test += group[:n_test]
        train += group[n_test:]
    rng.shuffle(train)
    return train, test
