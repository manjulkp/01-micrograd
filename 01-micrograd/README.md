# Project 1: Micrograd autograd

A scalar autograd engine and small neural network written from scratch, following
Andrej Karpathy's micrograd. The network learns to separate two classes of 2D
points that you draw yourself.

## Run it

Run these from this folder:

```powershell
uv sync                                    # install dependencies
uv run python draw_points.py               # draw your points, then close the window
uv run python train.py                     # train and save a plot to outputs/
uv run pytest                              # run the tests
```

To try training before drawing anything, use the synthetic sample data:

```powershell
uv run python make_sample_data.py
uv run python train.py --data data/sample_moons.csv
```

### Drawing controls

| Input | Action |
| --- | --- |
| Click and drag | Paint points |
| `1` / `2` | Switch to the blue / red class |
| `u` | Undo the last point |
| `s` | Save now |
| Close the window | Save and exit |

### Training options

| Option | Default | Meaning |
| --- | --- | --- |
| `--data` | `data/my_points.csv` | Dataset to train on |
| `--hidden` | `16 16` | Hidden layer sizes |
| `--activation` | `relu` | `relu` or `tanh` |
| `--steps` | `100` | Training steps |
| `--lr` | `1.0` | Starting learning rate |
| `--alpha` | `0.0001` | L2 regularization strength |
| `--seed` | `0` | Random seed for the split and starting weights |

## Files

| File | What it does |
| --- | --- |
| `micrograd/engine.py` | `Value` class and backpropagation |
| `micrograd/nn.py` | `Neuron`, `Layer` and `MLP` |
| `micrograd/data.py` | CSV loading and saving, train/test split |
| `draw_points.py` | Mouse tool for drawing a dataset |
| `make_sample_data.py` | Synthetic two-moons data |
| `train.py` | Training loop and plot |
| `tests/` | Gradient checks, training and drawing-tool tests |
