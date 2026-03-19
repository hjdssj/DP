# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains MNIST digit classification training scripts with differential privacy support using [Opacus](https://github.com/pytorch/opacus). The project uses a simple convolutional neural network (SampleConvNet) to train on MNIST with optional differential privacy guarantees.

## Running the Scripts

Both scripts accept the same arguments. Run from the `d:\DP` directory using the virtual environment:

```bash
# Train with differential privacy (default)
python minst_adaptive.py

# Train without differential privacy (baseline)
python minst_adaptive.py --disable-dp

# Custom hyperparameters
python minst_adaptive.py -n 5 -b 128 --lr 0.05 --sigma 1.5 -c 1.0

# Save the trained model
python minst_adaptive.py --save-model
```

Key arguments:
- `-n, --epochs`: Number of training epochs (default: 10)
- `-b, --batch-size`: Batch size (default: 64)
- `--lr`: Learning rate (default: 0.1)
- `--sigma`: Noise multiplier for DP (default: 1.0)
- `-c, --max-per-sample-grad_norm`: Gradient clipping norm (default: 1.0)
- `--delta`: Target delta for DP (default: 1e-5)
- `--disable-dp`: Disable privacy and train with vanilla SGD
- `--save-model`: Save trained model weights to `mnist_cnn_*.pt`
- `--device`: Device to use, "cuda" or "cpu" (default: "cuda")
- `-r, --n-runs`: Number of runs to average (default: 1)

Results are saved to `run_results_*.pt` files containing accuracy per run.

## Architecture

**Model (SampleConvNet)**: A 4-layer CNN
- Conv1: 1→16 channels, 8×8 kernel, stride 2
- Conv2: 16→32 channels, 4×4 kernel, stride 2
- FC1: 512→32 features
- FC2: 32→10 (digit classes)

**Training**: SGD optimizer with optional Opacus PrivacyEngine wrapping for differential privacy. The privacy engine handles gradient clipping and noise addition.

**Data**: MNIST dataset normalized with mean=0.1307, std=0.3081. Downloaded to `../mnist` by default.

## Dependencies

- PyTorch
- Opacus (`from opacus import PrivacyEngine`)
- torchvision
- numpy
- tqdm

The `.venv` directory contains a pre-configured virtual environment.

## File Structure

- `minst_baseline.py`: Basic MNIST training with DP-SGD using Opacus
- `minst_adaptive.py`: MNIST training with fixed clipping threshold
- `minst_adaptive_histogram.py`: MNIST with adaptive clipping + DP histogram tracking
- `README.md`: Project documentation
- `run_results_*.pt`, `histogram_results_*.pt`, `adaptive_results_*.pt`: Saved results
