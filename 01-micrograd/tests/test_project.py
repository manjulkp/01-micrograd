import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest
from matplotlib.backend_bases import KeyEvent, MouseEvent

from draw_points import PointPainter, create_axes
from make_sample_data import make_moons
from micrograd.data import load_points, save_points, train_test_split
from micrograd.nn import MLP
from train import accuracy, train


def test_save_and_load_round_trip(tmp_path):
    path = tmp_path / "nested" / "points.csv"
    points = [(0.25, -0.5, 1), (0.75, 0.125, -1)]
    save_points(path, points)
    assert load_points(path) == points


def test_load_rejects_invalid_labels(tmp_path):
    path = tmp_path / "points.csv"
    path.write_text("x,y,label\n0.1,0.2,0\n")
    with pytest.raises(ValueError, match="labels must be -1 or 1"):
        load_points(path)


def test_split_keeps_both_classes_and_every_point():
    points = make_moons(50)
    train_points, test_points = train_test_split(points)
    assert sorted(train_points + test_points) == sorted(points)
    assert {p[2] for p in train_points} == {-1, 1}
    assert {p[2] for p in test_points} == {-1, 1}


def test_split_rejects_missing_class():
    with pytest.raises(ValueError, match="at least 2 points of each class"):
        train_test_split([(0.0, 0.0, 1), (0.1, 0.1, 1), (0.2, 0.2, 1)])


def test_training_lowers_loss_and_learns_moons():
    train_points, test_points = train_test_split(make_moons(60))
    model = MLP(2, [8, 8, 1], seed=0)
    history = train(model, train_points, steps=60, lr=1.0, alpha=1e-4, log_every=1000)
    assert history[-1] < history[0]
    assert accuracy(model, train_points) >= 0.9
    assert accuracy(model, test_points) >= 0.8


@pytest.fixture
def painter(tmp_path):
    fig, ax = create_axes()
    fig.canvas.draw()
    yield PointPainter(ax, tmp_path / "drawn.csv")
    plt.close(fig)


def mouse(painter, name, x, y):
    px, py = painter.ax.transData.transform((x, y))
    canvas = painter.ax.figure.canvas
    canvas.callbacks.process(name, MouseEvent(name, canvas, px, py, button=1))


def key(painter, value):
    canvas = painter.ax.figure.canvas
    canvas.callbacks.process("key_press_event", KeyEvent("key_press_event", canvas, value))


def test_painter_drag_switch_undo_and_save(painter):
    mouse(painter, "button_press_event", -0.5, -0.5)
    mouse(painter, "motion_notify_event", -0.3, -0.5)
    mouse(painter, "motion_notify_event", -0.295, -0.5)  # too close, so it is skipped
    mouse(painter, "button_release_event", -0.3, -0.5)
    mouse(painter, "motion_notify_event", 0.0, 0.0)  # not dragging, so it is ignored
    assert [p[2] for p in painter.points] == [-1, -1]

    key(painter, "2")
    mouse(painter, "button_press_event", 0.5, 0.5)
    mouse(painter, "button_release_event", 0.5, 0.5)
    mouse(painter, "button_press_event", -0.5, 0.5)
    mouse(painter, "button_release_event", -0.5, 0.5)
    assert [p[2] for p in painter.points] == [-1, -1, 1, 1]

    key(painter, "u")
    assert len(painter.points) == 3

    key(painter, "s")
    saved = load_points(painter.path)
    assert [p[2] for p in saved] == [-1, -1, 1]
    assert saved[2][:2] == pytest.approx((0.5, 0.5), abs=0.01)


def test_painter_does_not_write_an_empty_file(painter):
    key(painter, "s")
    assert not painter.path.exists()
