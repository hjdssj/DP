# Adaptive Differential Privacy for MNIST

MNIST digit classification with differential privacy (DP) and adaptive gradient clipping. Built with PyTorch and [Opacus](https://github.com/pytorch/opacus).

## Scripts

| File | Description |
|------|-------------|
| `minst_baseline.py` | Basic MNIST training with DP-SGD using Opacus (fixed C) |
| `minst_adaptive_histogram.py` | **Adaptive clipping** based on clipped ratio + histogram visualization |
| `minst_adaptive_dp_manual.py` | Manual DP-SGD implementation (full control, less stable) |

## Quick Start

```bash
# Activate virtual environment
source .venv/Scripts/activate  # Windows
# or: source .venv/bin/activate  # Linux

# Baseline DP-SGD (fixed C = 0.4, optimal)
python minst_baseline.py -n 10 -b 64 --sigma 1.0 -c 0.4

# Adaptive clipping (auto-adjusts C based on clipped ratio)
python minst_adaptive_histogram.py -n 10 -b 64 --sigma 1.0 -c 0.4 --target-ratio 0.3 --plot

# Without DP (baseline)
python minst_baseline.py --disable-dp
```

## Adaptive Clipping

The `minst_adaptive_histogram.py` implements **clipped-ratio based adaptive C**:

```
C 调整逻辑：

如果 clipped_ratio > target_ratio + tolerance → 增大C
如果 clipped_ratio < target_ratio - tolerance → 减小C
如果在容忍范围内 → C不变

target_ratio = 30%, tolerance = 5%

[0% ───────────────────────────────────────── 100%]
   │←─────── 太小 ───────→│←── 稳定 ──→│←─────── 太大 ───────→│
                            25%    30%    35%
```

**核心思想：**
- C太小 → 太多样本被裁剪 → bias大、noise小
- C太大 → 太少样本被裁剪 → bias小、noise大
- 目标：维持约30%样本被裁剪（bias-variance平衡）

## Key Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `-n` | Number of epochs | 10 |
| `-b` | Batch size | 64 |
| `--lr` | Learning rate | 0.1 |
| `--sigma` | DP noise multiplier | 1.0 |
| `-c, --initial-c` | Initial clipping threshold | 1.0 |
| `--target-ratio` | Target clipped ratio (0.0-1.0) | 0.3 |
| `--plot` | Enable histogram visualization | False |
| `--disable-dp` | Train without DP | False |

## Visualization

With `--plot`, generates:

```
histogram_plots_*/
├── epoch_002.png  # Per-epoch gradient norm distribution
├── epoch_004.png  # Left: log scale, Right: zoomed (norm < 0.3)
├── epoch_006.png
├── ...
└── run_1_summary.png  # C history + clipped ratio over epochs
```

## Results

| Method | C | Accuracy | ε |
|--------|---|---------|-----|
| Baseline (fixed C=0.4) | 0.4 | **94.08%** | 0.50 |
| Adaptive (target 30%) | 0.40 | 93.95% | 0.50 |

Adaptive clipping converges C ≈ 0.4 (optimal), achieving nearly identical accuracy.

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
