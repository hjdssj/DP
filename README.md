# Adaptive Differential Privacy for MNIST

MNIST digit classification with differential privacy (DP) and adaptive gradient clipping. Built with PyTorch and [Opacus](https://github.com/pytorch/opacus).

## Scripts

| File | Description |
|------|-------------|
| `minst_baseline.py` | Basic MNIST training with DP-SGD using Opacus |
| `minst_adaptive.py` | MNIST training with fixed clipping threshold |
| `minst_adaptive_histogram.py` | MNIST with **adaptive clipping** based on MSE minimization + DP histogram visualization |

## Quick Start

```bash
# Activate virtual environment
source .venv/Scripts/activate  # Windows
# or: source .venv/bin/activate  # Linux

# Basic DP training
python minst_baseline.py -n 5

# Adaptive clipping with MSE optimization
python minst_adaptive_histogram.py -n 5 --adaptive-method mse_optimize --plot
```

## Adaptive Gradient Clipping

The key innovation in `minst_adaptive_histogram.py` is the **AdaptiveClipper** class that dynamically adjusts the clipping threshold `C` each epoch to minimize gradient estimation MSE:

```
MSE = Bias² + Variance

- Bias²: from scaling clipped gradients (high when C is small)
- Variance: from DP noise ∝ C² (high when C is large)
```

Two adaptation methods are available:

| Method | Strategy |
|--------|----------|
| `mse_optimize` | Grid search over percentiles to find C that minimizes estimated MSE |
| `percentile` | Set C to keep a target fraction of samples unclipped |

## Key Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `-n` | Number of epochs | 10 |
| `-b` | Batch size | 64 |
| `--lr` | Learning rate | 0.1 |
| `--sigma` | DP noise multiplier | 1.0 |
| `-c` | Initial clipping threshold | 1.0 |
| `--adaptive-method` | `mse_optimize` or `percentile` | `mse_optimize` |
| `--plot` | Enable histogram visualization | False |
| `--disable-dp` | Train without DP (baseline) | False |

## Output

Training generates:
- `run_results_*.pt` - Accuracy results
- `histogram_plots_*/` - Per-epoch gradient norm histograms (with `--plot`)
- `adaptive_results_*.pt` - C history, MSE decomposition (adaptive version)

## Model Architecture

SampleConvNet: 4-layer CNN
```
Conv1: 1→16 channels, 8×8, stride 2
Conv2: 16→32 channels, 4×4, stride 2
FC1: 512→32
FC2: 32→10
```

## Dependencies

- PyTorch
- Opacus (`pip install opacus`)
- torchvision
- numpy
- tqdm
- matplotlib (for visualization)
