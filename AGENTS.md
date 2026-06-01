# Repository Guidelines

## Project Structure & Module Organization

This repository is a Python research workspace for differentially private SGD and adaptive gradient clipping experiments. Core scripts live at the repository root, grouped by dataset or experiment family: `minst_*.py` for MNIST, `fashion_*.py` / `fashion_mnist_*.py` for Fashion-MNIST, and `cifar10_*.py` for CIFAR-10. Shared adaptive clipping logic is in `adaptive_clipping.py`, while batch orchestration is in `run_all_experiments.py`.

Generated artifacts are also stored at the root or in result folders. Treat `run_results_*.pt`, `adaptive_histogram_results_*.pt`, `ckpt_*_done.pt`, `experiment_results*/`, `histogram_plots*/`, and `cifar10_nodp_results/` as experiment outputs. Notebooks (`*.ipynb`) are exploratory companions to the scripts.

## Build, Test, and Development Commands

Activate the local environment before running experiments:

```bash
source .venv/Scripts/activate
```

Common commands:

```bash
python minst_baseline.py -n 10 -b 64 --sigma 1.0 -c 0.4
python minst_adaptive_histogram.py -n 10 -b 64 --sigma 1.0 -c 0.4 --target-ratio 0.3 --plot
python minst_baseline.py --disable-dp
python run_all_experiments.py
```

Use `--help` on individual scripts to confirm supported flags before adding new parameters.

## Coding Style & Naming Conventions

Use standard Python style with 4-space indentation, clear function names, and explicit argparse options. Follow the existing filename pattern: `<dataset>_<method>_<purpose>.py`, for example `cifar10_resnet_dp.py` or `fashion_mnist_epsilon_tradeoff.py`. Keep experiment outputs named with enough parameters to reproduce the run, matching existing names such as `run_results_mnist_0.1_1.0_0.4_64_20.pt`.

Prefer small, script-local changes unless logic is reused across datasets; place reusable clipping utilities in `adaptive_clipping.py`.

## Testing Guidelines

There is no formal test suite in this repository. Validate changes with short smoke runs before full experiments, for example:

```bash
python minst_baseline.py -n 1 -b 64 --sigma 1.0 -c 0.4
python minst_adaptive_histogram.py -n 1 -b 64 --sigma 1.0 -c 0.4
```

For plotting changes, run with `--plot` and inspect the generated `histogram_plots*/run_1_summary.png`.

## Commit & Pull Request Guidelines

Recent commits use short imperative messages, such as `Add histogram-based adaptive gradient clipping with Opacus`. Keep commits focused on one experiment, fix, or documentation update.

Pull requests should include the purpose, commands run, key metrics or output files produced, and any changed assumptions about privacy parameters. Include screenshots only when plots or notebook visualizations change.

## Security & Configuration Tips

Do not commit private datasets, large downloaded archives, or machine-specific environment changes. Keep privacy settings explicit in commands and filenames: include `sigma`, clipping threshold `c`, batch size, and epoch count when saving new results.
