#!/usr/bin/env python3
"""Run representative DP histogram experiments with resume support.

The runner skips an experiment when its expected result file already exists.
Run it again after interruption to continue from the remaining experiments.
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
    name: str
    script: str
    args: tuple[str, ...]
    output: str


EXPERIMENTS = [
    Experiment(
        name="mnist_fixed_sigma08_b64_e10",
        script="minst_adaptive_histogram.py",
        args=(
            "-n", "10",
            "-b", "64",
            "--sigma", "0.8",
            "--mode", "fixed",
            "-c", "1.0",
            "--device", "cuda",
        ),
        output="adaptive_histogram_results_mnist_adaptive_fixed_0.1_0.8_1.0_64_10_hist0.pt",
    ),
    Experiment(
        name="mnist_fixed_sigma10_b64_e10",
        script="minst_adaptive_histogram.py",
        args=(
            "-n", "10",
            "-b", "64",
            "--sigma", "1.0",
            "--mode", "fixed",
            "-c", "1.0",
            "--device", "cuda",
        ),
        output="adaptive_histogram_results_mnist_adaptive_fixed_0.1_1.0_1.0_64_10_hist0.pt",
    ),
    Experiment(
        name="mnist_fixed_sigma15_b64_e10",
        script="minst_adaptive_histogram.py",
        args=(
            "-n", "10",
            "-b", "64",
            "--sigma", "1.5",
            "--mode", "fixed",
            "-c", "1.0",
            "--device", "cuda",
        ),
        output="adaptive_histogram_results_mnist_adaptive_fixed_0.1_1.5_1.0_64_10_hist0.pt",
    ),
    Experiment(
        name="fashion_mse_dp_hist025_bs2048_e20",
        script="fashionmnist_resnet18_dp_baseline.py",
        args=(
            "-n", "20",
            "-b", "2048",
            "--max-physical-batch-size", "32",
            "--sigma", "1.0",
            "--mode", "mse",
            "--use-dp-histogram",
            "--epsilon-hist", "0.025",
            "--device", "cuda",
        ),
        output="adaptive_histogram_results_fashion_resnet18_mse_0.05_1.0_1.0_2048_20_hist0.025.pt",
    ),
    Experiment(
        name="fashion_fixed_bs2048_e20",
        script="fashionmnist_resnet18_dp_baseline.py",
        args=(
            "-n", "20",
            "-b", "2048",
            "--max-physical-batch-size", "32",
            "--sigma", "1.0",
            "--mode", "fixed",
            "-c", "1.0",
            "--device", "cuda",
        ),
        output="adaptive_histogram_results_fashion_resnet18_fixed_0.05_1.0_1.0_2048_20_hist0.pt",
    ),
    Experiment(
        name="cifar10_fixed_bs1024_e20",
        script="cifar10_resnet_dp.py",
        args=(
            "-n", "20",
            "-b", "1024",
            "--max-physical-batch-size", "32",
            "--sigma", "1.0",
            "--mode", "fixed",
            "-c", "1.0",
            "--device", "cuda",
        ),
        output="adaptive_histogram_results_cifar10_fixed_0.1_1.0_1.0_1024_20_hist0.pt",
    ),
    Experiment(
        name="cifar10_mse_dp_hist025_bs1024_e20",
        script="cifar10_resnet_dp.py",
        args=(
            "-n", "20",
            "-b", "1024",
            "--max-physical-batch-size", "32",
            "--sigma", "1.0",
            "--mode", "mse",
            "--use-dp-histogram",
            "--epsilon-hist", "0.025",
            "--device", "cuda",
        ),
        output="adaptive_histogram_results_cifar10_mse_0.1_1.0_1.0_1024_20_hist0.025.pt",
    ),
]


def build_command(experiment, python_exe):
    script_path = ROOT / experiment.script
    return [python_exe, str(script_path), *experiment.args]


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
        print(f"[warn] {experiment.name} finished, but expected output was not found:")
        print(f"       {output_path}")
        return False

    print(f"[done] {experiment.name}")
    return True


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run representative DP histogram experiments with resume support.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--python", default=sys.executable,
                        help="Python executable used to launch experiment scripts")
    parser.add_argument("--force", action="store_true",
                        help="Rerun experiments even if expected result files exist")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print planned commands without running them")
    parser.add_argument("--list", action="store_true",
                        help="List experiments and expected outputs")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.list:
        for exp in EXPERIMENTS:
            status = "done" if (RESULTS_DIR / exp.output).exists() else "pending"
            print(f"{status:7} {exp.name:36} {RESULTS_DIR / exp.output}")
        return

    print(f"Results directory: {RESULTS_DIR}")
    for exp in EXPERIMENTS:
        ok = run_experiment(
            exp,
            python_exe=args.python,
            force=args.force,
            dry_run=args.dry_run,
        )
        if not ok:
            print("Stopped. Re-run this script after fixing the issue to resume.")
            sys.exit(1)

    print("All representative experiments are complete.")


if __name__ == "__main__":
    main()
