#!/usr/bin/env python3
"""Run full epsilon-Accuracy trade-off sweeps with resume support.

The sweep covers fixed clipping and MSE + DP histogram for MNIST,
Fashion-MNIST, and CIFAR-10. Existing result files are skipped.
"""

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"


@dataclass(frozen=True)
class Experiment:
    dataset: str
    method: str
    name: str
    script: str
    args: tuple[str, ...]
    output: str


def fmt_sigma(sigma):
    return str(sigma)


def mnist_experiments():
    sigmas = [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]
    experiments = []
    for sigma in sigmas:
        sigma_s = fmt_sigma(sigma)
        common = ("-n", "10", "-b", "64", "--sigma", sigma_s, "--device", "cuda")
        experiments.append(
            Experiment(
                dataset="mnist",
                method="fixed",
                name=f"mnist_fixed_sigma{sigma_s}",
                script="minst_adaptive_histogram.py",
                args=(*common, "--mode", "fixed", "-c", "1.0"),
                output=f"adaptive_histogram_results_mnist_adaptive_fixed_0.1_{sigma_s}_1.0_64_10_hist0.pt",
            )
        )
        experiments.append(
            Experiment(
                dataset="mnist",
                method="mse",
                name=f"mnist_mse_dp_hist_sigma{sigma_s}",
                script="minst_adaptive_histogram.py",
                args=(
                    *common,
                    "--mode", "mse",
                    "--use-dp-histogram",
                    "--epsilon-hist", "0.025",
                ),
                output=f"adaptive_histogram_results_mnist_adaptive_mse_0.1_{sigma_s}_1.0_64_10_hist0.025.pt",
            )
        )
    return experiments


def fashion_experiments():
    # Reuse the sigma grid from the existing Fashion-MNIST epsilon sweep.
    sigmas = [3.383908, 1.929262, 1.212828, 0.856756, 0.647915, 0.503971]
    experiments = []
    for sigma in sigmas:
        sigma_s = fmt_sigma(sigma)
        common = (
            "-n", "20",
            "-b", "2048",
            "--max-physical-batch-size", "32",
            "--sigma", sigma_s,
            "--device", "cuda",
        )
        experiments.append(
            Experiment(
                dataset="fashion",
                method="fixed",
                name=f"fashion_fixed_sigma{sigma_s}",
                script="fashionmnist_resnet18_dp_baseline.py",
                args=(*common, "--mode", "fixed", "-c", "1.0"),
                output=f"adaptive_histogram_results_fashion_resnet18_fixed_0.05_{sigma_s}_1.0_2048_20_hist0.pt",
            )
        )
        experiments.append(
            Experiment(
                dataset="fashion",
                method="mse",
                name=f"fashion_mse_dp_hist_sigma{sigma_s}",
                script="fashionmnist_resnet18_dp_baseline.py",
                args=(
                    *common,
                    "--mode", "mse",
                    "--use-dp-histogram",
                    "--epsilon-hist", "0.025",
                ),
                output=f"adaptive_histogram_results_fashion_resnet18_mse_0.05_{sigma_s}_1.0_2048_20_hist0.025.pt",
            )
        )
    return experiments


def cifar_experiments():
    sigmas = [1.5, 1.2, 1.0, 0.8, 0.6, 0.5]
    experiments = []
    for sigma in sigmas:
        sigma_s = fmt_sigma(sigma)
        common = (
            "-n", "20",
            "-b", "1024",
            "--max-physical-batch-size", "32",
            "--sigma", sigma_s,
            "--device", "cuda",
        )
        experiments.append(
            Experiment(
                dataset="cifar",
                method="fixed",
                name=f"cifar_fixed_sigma{sigma_s}",
                script="cifar10_resnet_dp.py",
                args=(*common, "--mode", "fixed", "-c", "1.0"),
                output=f"adaptive_histogram_results_cifar10_fixed_0.1_{sigma_s}_1.0_1024_20_hist0.pt",
            )
        )
        experiments.append(
            Experiment(
                dataset="cifar",
                method="mse",
                name=f"cifar_mse_dp_hist_sigma{sigma_s}",
                script="cifar10_resnet_dp.py",
                args=(
                    *common,
                    "--mode", "mse",
                    "--use-dp-histogram",
                    "--epsilon-hist", "0.025",
                ),
                output=f"adaptive_histogram_results_cifar10_mse_0.1_{sigma_s}_1.0_1024_20_hist0.025.pt",
            )
        )
    return experiments


def all_experiments():
    return [*mnist_experiments(), *fashion_experiments(), *cifar_experiments()]


def filtered_experiments(dataset, method):
    experiments = all_experiments()
    if dataset != "all":
        experiments = [exp for exp in experiments if exp.dataset == dataset]
    if method != "all":
        experiments = [exp for exp in experiments if exp.method == method]
    return experiments


def build_command(experiment, python_exe):
    return [python_exe, str(ROOT / experiment.script), *experiment.args]


def run_experiment(experiment, python_exe, force=False, dry_run=False):
    output_path = RESULTS_DIR / experiment.output
    if output_path.exists() and not force:
        print(f"[skip] {experiment.name}")
        print(f"       existing: {output_path}")
        return True

    command = build_command(experiment, python_exe)
    print(f"[run]  {experiment.name}")
    print(f"       output: {output_path}")
    print(f"       cmd: {' '.join(command)}")

    if dry_run:
        return True

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(command, cwd=ROOT.parent)
    if completed.returncode != 0:
        print(f"[fail] {experiment.name} exited with code {completed.returncode}")
        return False
    if not output_path.exists():
        print(f"[warn] expected output not found: {output_path}")
        return False

    print(f"[done] {experiment.name}")
    return True


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run full trade-off sweeps with resume support.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", choices=["all", "mnist", "fashion", "cifar"],
                        default="all")
    parser.add_argument("--method", choices=["all", "fixed", "mse"], default="all")
    parser.add_argument("--python", default=sys.executable,
                        help="Python executable used to launch experiment scripts")
    parser.add_argument("--list", action="store_true",
                        help="List selected experiments and completion status")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without running experiments")
    parser.add_argument("--force", action="store_true",
                        help="Rerun experiments even if expected outputs exist")
    return parser.parse_args()


def main():
    args = parse_args()
    experiments = filtered_experiments(args.dataset, args.method)

    if args.list:
        for exp in experiments:
            status = "done" if (RESULTS_DIR / exp.output).exists() else "pending"
            print(f"{status:7} {exp.dataset:7} {exp.method:5} {exp.name:36} {RESULTS_DIR / exp.output}")
        return

    print(f"Selected experiments: {len(experiments)}")
    print(f"Results directory: {RESULTS_DIR}")
    for exp in experiments:
        ok = run_experiment(exp, args.python, force=args.force, dry_run=args.dry_run)
        if not ok:
            print("Stopped. Re-run this script after fixing the issue to resume.")
            sys.exit(1)

    print("Selected trade-off sweep experiments are complete.")


if __name__ == "__main__":
    main()
