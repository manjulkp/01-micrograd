"""Draw your own two-class dataset with the mouse.

Click and drag to paint points. Your points are saved when you close the window.

Keys:  1 = blue class    2 = red class    u = undo last point    s = save now
"""

import argparse
import math
from pathlib import Path

import matplotlib

from micrograd.data import load_points, save_points

COLORS = {-1: "tab:blue", 1: "tab:red"}
CLASS_KEYS = {"1": -1, "2": 1}
MIN_SPACING = 0.03  # stops a slow drag from stacking points on top of each other


class PointPainter:
    def __init__(self, ax, path, points=()):
        self.ax = ax
        self.path = Path(path)
        self.points = list(points)
        self.label = -1
        self.painting = False
        self.scatter = ax.scatter([], [], s=25, edgecolors="white", linewidths=0.4)

        canvas = ax.figure.canvas
        canvas.mpl_connect("button_press_event", self.on_press)
        canvas.mpl_connect("button_release_event", self.on_release)
        canvas.mpl_connect("motion_notify_event", self.on_motion)
        canvas.mpl_connect("key_press_event", self.on_key)
        canvas.mpl_connect("close_event", self.on_close)
        self.redraw()

    def add_point(self, x, y):
        if self.points:
            last_x, last_y, last_label = self.points[-1]
            if last_label == self.label and math.dist((x, y), (last_x, last_y)) < MIN_SPACING:
                return
        self.points.append((x, y, self.label))
        self.redraw()

    def on_press(self, event):
        if event.inaxes is self.ax and event.button == 1:
            self.painting = True
            self.add_point(event.xdata, event.ydata)

    def on_release(self, event):
        self.painting = False

    def on_motion(self, event):
        if self.painting and event.inaxes is self.ax:
            self.add_point(event.xdata, event.ydata)

    def on_key(self, event):
        if event.key in CLASS_KEYS:
            self.label = CLASS_KEYS[event.key]
            self.redraw()
        elif event.key == "u" and self.points:
            self.points.pop()
            self.redraw()
        elif event.key == "s":
            self.save()

    def on_close(self, event):
        self.save()

    def save(self):
        if self.points:
            save_points(self.path, self.points)
            print(f"Saved {len(self.points)} points to {self.path}")

    def redraw(self):
        if self.points:
            self.scatter.set_offsets([(x, y) for x, y, _ in self.points])
            self.scatter.set_facecolors([COLORS[label] for _, _, label in self.points])
        else:
            self.scatter.set_offsets([(float("nan"), float("nan"))])
        counts = {c: sum(p[2] == c for p in self.points) for c in COLORS}
        painting = "blue" if self.label == -1 else "red"
        self.ax.set_title(
            f"Painting {painting}  |  blue: {counts[-1]}  red: {counts[1]}\n"
            "1/2 = change class   u = undo   s = save   close window = save and exit",
            fontsize=10,
        )
        self.ax.figure.canvas.draw_idle()


def create_axes():
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 7.5))
    ax.set(xlim=(-1, 1), ylim=(-1, 1), aspect="equal")
    ax.grid(alpha=0.25)
    return fig, ax


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/my_points.csv")
    args = parser.parse_args()

    # Turn off matplotlib's own shortcuts and toolbar so that they can't
    # interfere with painting or the keys above.
    matplotlib.rcParams["toolbar"] = "None"
    matplotlib.rcParams["keymap.save"] = []

    import matplotlib.pyplot as plt

    existing = load_points(args.out) if Path(args.out).exists() else []
    if existing:
        print(f"Continuing from {len(existing)} existing points in {args.out}")
    _, ax = create_axes()
    painter = PointPainter(ax, args.out, existing)
    plt.show()
