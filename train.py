"""Train a small MLP on a 2D point dataset and plot what it learned."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write the plot to a file without opening a window
import matplotlib.pyplot as plt

from micrograd.data import load_points, train_test_split
from micrograd.nn import MLP


def score(model, x, y):
    return model([x, y])


def compute_loss(model, points, alpha):
    """Return the hinge loss with L2 regularization, plus accuracy on these points."""
    scores = [score(model, x, y) for x, y, _ in points]
    # Hinge loss is zero for points on the correct side by a margin of at least 1.
    hinge = sum((1 - label * s).relu() for s, (_, _, label) in zip(scores, points))
    data_loss = hinge * (1.0 / len(points))
    l2 = alpha * sum(p * p for p in model.parameters())
    correct = sum((s.data > 0) == (label > 0) for s, (_, _, label) in zip(scores, points))
    return data_loss + l2, correct / len(points)


def accuracy(model, points):
    correct = sum((score(model, x, y).data > 0) == (label > 0) for x, y, label in points)
    return correct / len(points)


def train(model, points, steps, lr, alpha, log_every=10):
    history = []
    for step in range(steps):
        loss, acc = compute_loss(model, points, alpha)
        model.zero_grad()
        loss.backward()

        # Take large steps at first and smaller ones as training settles.
        step_size = lr * (1.0 - 0.9 * step / steps)
        for p in model.parameters():
            p.data -= step_size * p.grad

        history.append(loss.data)
        if step % log_every == 0 or step == steps - 1:
            print(f"step {step:4d}  loss {loss.data:.4f}  train accuracy {acc:.1%}")
    return history


def save_plot(model, train_points, test_points, history, path, resolution=50):
    all_points = train_points + test_points
    xs_all, ys_all = [p[0] for p in all_points], [p[1] for p in all_points]
    pad = 0.2
    x_min, x_max = min(xs_all) - pad, max(xs_all) + pad
    y_min, y_max = min(ys_all) - pad, max(ys_all) + pad
    grid_x = [x_min + (x_max - x_min) * i / (resolution - 1) for i in range(resolution)]
    grid_y = [y_min + (y_max - y_min) * i / (resolution - 1) for i in range(resolution)]
    grid = [[score(model, x, y).data for x in grid_x] for y in grid_y]

    fig, (ax_map, ax_loss) = plt.subplots(1, 2, figsize=(12, 5.2))

    ax_map.contourf(grid_x, grid_y, grid, levels=[-1e9, 0, 1e9], colors=["#c6dbef", "#fcd5ce"])
    ax_map.contour(grid_x, grid_y, grid, levels=[0], colors="#444444", linewidths=1)
    for points, marker, name in ((train_points, "o", "train"), (test_points, "^", "test")):
        ax_map.scatter(
            [p[0] for p in points],
            [p[1] for p in points],
            c=["tab:blue" if p[2] == -1 else "tab:red" for p in points],
            marker=marker,
            s=32,
            edgecolors="white",
            linewidths=0.5,
            label=name,
        )
    ax_map.set_title("Decision boundary")
    ax_map.legend(loc="upper right")

    ax_loss.plot(history, color="#333333")
    ax_loss.set(title="Training loss", xlabel="Step", ylabel="Loss")
    ax_loss.grid(alpha=0.25)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/my_points.csv")
    parser.add_argument("--hidden", type=int, nargs="+", default=[16, 16], help="hidden layer sizes")
    parser.add_argument("--activation", choices=["relu", "tanh"], default="relu")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=1e-4, help="L2 regularization strength")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--plot", default=None, help="default: outputs/<data file name>.png")
    args = parser.parse_args()

    if not Path(args.data).exists():
        parser.error(
            f"{args.data} not found. Draw it with draw_points.py, "
            "or run make_sample_data.py and pass --data data/sample_moons.csv"
        )

    train_points, test_points = train_test_split(load_points(args.data), seed=args.seed)
    model = MLP(2, [*args.hidden, 1], seed=args.seed, activation=args.activation)
    print(
        f"{len(train_points)} train points, {len(test_points)} test points, "
        f"{len(model.parameters())} parameters"
    )

    history = train(model, train_points, args.steps, args.lr, args.alpha)
    print(f"Train accuracy: {accuracy(model, train_points):.1%}")
    print(f"Test accuracy:  {accuracy(model, test_points):.1%}")

    plot_path = args.plot or f"outputs/{Path(args.data).stem}.png"
    save_plot(model, train_points, test_points, history, plot_path)
    print(f"Plot saved to {plot_path}")


if __name__ == "__main__":
    main()
