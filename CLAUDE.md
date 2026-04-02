# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains MNIST digit classification training scripts with differential privacy support using [Opacus](https://github.com/pytorch/opacus). The project uses a simple convolutional neural network (SampleConvNet) to train on MNIST with optional differential privacy guarantees.

## Running the Scripts

```bash
# Baseline DP-SGD (fixed C)
python minst_baseline.py -n 10 -b 64 --sigma 1.0 -c 0.4

# Adaptive clipping based on clipped ratio
python minst_adaptive_histogram.py -n 10 -b 64 --sigma 1.0 -c 0.4 --target-ratio 0.3 --plot

# Disable DP (baseline comparison)
python minst_baseline.py --disable-dp
```

Key arguments:
- `-n, --epochs`: Number of training epochs (default: 10)
- `-b, --batch-size`: Batch size (default: 64)
- `--lr`: Learning rate (default: 0.1)
- `--sigma`: Noise multiplier for DP (default: 1.0)
- `-c, --initial-c`: Initial clipping threshold (default: 1.0)
- `--target-ratio`: Target clipped sample ratio for adaptive C (default: 0.3 = 30%)
- `--delta`: Target delta for DP (default: 1e-5)
- `--disable-dp`: Disable privacy and train with vanilla SGD
- `--save-model`: Save trained model weights
- `--device`: Device to use, "cuda" or "cpu" (default: "cuda")
- `--plot`: Enable histogram visualization
- `-r, --n-runs`: Number of runs to average (default: 1)

Results saved to `run_results_*.pt` and `adaptive_histogram_results_*.pt`.

## Architecture

**Model (SampleConvNet)**: A 4-layer CNN
- Conv1: 1→16 channels, 8×8 kernel, stride 2
- Conv2: 16→32 channels, 4×4 kernel, stride 2
- FC1: 512→32 features
- FC2: 32→10 (digit classes)

**Training**: SGD optimizer with optional Opacus PrivacyEngine wrapping for differential privacy.

**Data**: MNIST dataset normalized with mean=0.1307, std=0.3081. Downloaded to `../mnist` by default.

## Dependencies

- PyTorch
- Opacus (`from opacus import PrivacyEngine`)
- torchvision
- numpy
- tqdm
- matplotlib (for visualization)

The `.venv` directory contains a pre-configured virtual environment.

## File Structure

- `minst_baseline.py`: Basic MNIST training with DP-SGD using Opacus (fixed C)
- `minst_adaptive.py`: Identical to baseline (fixed C)
- `minst_adaptive_histogram.py`: **Adaptive clipping based on clipped ratio** + histogram visualization
- `minst_adaptive_dp_manual.py`: Manual DP-SGD implementation (full control, less stable)
- `IMPLEMENTATION_NOTES.md`: Detailed notes on adaptive clipping challenges
- `histogram_plots_*/`: Generated histogram plots (when --plot enabled)
- `run_results_*.pt`, `adaptive_histogram_results_*.pt`: Saved results

## Adaptive Clipping Algorithm

The `minst_adaptive_histogram.py` uses a **clipped-ratio based** adaptive method:

1. Each epoch, track the fraction of samples that were clipped (norm ≥ C)
2. If clipped_ratio > target_ratio + tolerance → increase C
3. If clipped_ratio < target_ratio - tolerance → decrease C
4. Otherwise → keep C stable

```
target_ratio = 30%, tolerance = 5%

[0% ───────────────────────────────────────── 100%]
   │←─────── 减小C ───────→│←── 稳定 ──→│←─────── 增大C ───────→│
                            25%    30%    35%
```

## Key Findings

- **Optimal fixed C = 0.4** achieves ~94% accuracy with Opacus
- Adaptive clipping converges C ≈ 0.4 when target_ratio = 30%
- Adaptive accuracy: ~93.6% vs Fixed C=0.4: ~94.1% (nearly identical)
- Clipped ratio adjustment is stable and effective
- Per-sample gradient norms follow heavy-tailed distribution
