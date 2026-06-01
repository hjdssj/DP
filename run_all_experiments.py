#!/usr/bin/env python3
"""
Automated experiment runner for DP-SGD adaptive clipping comparison.

Supports two datasets and two sweep modes:
  --dataset mnist    : MNIST with SampleConvNet (default)
  --dataset fashion  : Fashion-MNIST with ResNet18
  --sweep_mode sigma   : iterate over a list of noise multipliers
  --sweep_mode epsilon : specify target epsilon values, auto-compute sigma

Usage:
    python run_all_experiments.py                                # MNIST, sigma mode
    python run_all_experiments.py --dataset fashion              # Fashion-MNIST, sigma mode
    python run_all_experiments.py --dataset fashion --sweep_mode epsilon
    python run_all_experiments.py --sweep_mode epsilon --epsilons 1 2 4 8
    python run_all_experiments.py --max_tasks 3                  # debug: run 3 tasks
"""

import argparse
import csv
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time


# ---------------------------------------------------------------------------
# Dataset-specific configuration
# ---------------------------------------------------------------------------

SIGMA_LIST = [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]
EPSILON_LIST = [1, 2, 4, 8, 16, 32]
METHODS = ["baseline", "ratio", "mse"]

DATASET_CONFIGS = {
    "mnist": {
        "n_train": 60000,
        "batch_size": 64,
        "lr": 0.1,
        "fixed_c": 0.4,
        "initial_c": 1.0,
        "target_ratio": 0.2,
        "delta": 1e-5,
        "epochs": 20,
        "timeout": 1800,
        "max_physical_batch_size": None,
    },
    "fashion": {
        "n_train": 60000,
        "batch_size": 2048,
        "lr": 0.05,
        "fixed_c": 1.0,
        "initial_c": 1.0,
        "target_ratio": 0.2,
        "delta": None,          # auto 1/N
        "epochs": 20,
        "timeout": 3600,        # ResNet18 is slower
        "max_physical_batch_size": 32,
    },
}

DEVICE = "cuda"


# ---------------------------------------------------------------------------
# Epsilon <-> Sigma conversion
# ---------------------------------------------------------------------------

def compute_epsilon(sigma, steps, sample_rate, delta):
    """Compute epsilon using RDP accountant.

    Uses the math directly instead of RDPAccountant to avoid the
    sample_rate < 0.5 constraint in the vectorized history format.
    """
    from opacus.accountants.analysis import rdp as rdp_analysis

    alphas = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))

    rdp = rdp_analysis.compute_rdp(
        q=sample_rate,
        noise_multiplier=sigma,
        steps=steps,
        orders=alphas,
    )
    eps, _ = rdp_analysis.get_privacy_spent(
        orders=alphas,
        rdp=rdp,
        delta=delta,
    )
    return eps


def sigma_for_epsilon(target_eps, steps, sample_rate, delta):
    """Binary search for the sigma that yields a target epsilon."""
    lo, hi = 0.01, 100.0
    for _ in range(100):
        mid = (lo + hi) / 2
        eps = compute_epsilon(mid, steps, sample_rate, delta)
        if eps > target_eps:
            lo = mid
        else:
            hi = mid

    sigma = (lo + hi) / 2
    actual_eps = compute_epsilon(sigma, steps, sample_rate, delta)
    return sigma, actual_eps


def build_sigma_map(epsilon_list, cfg):
    """Map each target epsilon to the corresponding sigma.

    Args:
        cfg: dataset config dict (needs n_train, batch_size, delta, epochs)
    """
    delta = cfg["delta"] or (1.0 / cfg["n_train"])
    steps = cfg["epochs"] * (cfg["n_train"] // cfg["batch_size"])
    sample_rate = cfg["batch_size"] / cfg["n_train"]

    sigma_map = {}
    print(f"Computing sigma for target epsilons (steps={steps}, q={sample_rate:.6f}):")
    for target_eps in epsilon_list:
        sig, actual = sigma_for_epsilon(target_eps, steps, sample_rate, delta)
        sigma_map[target_eps] = (sig, actual)
        print(f"  target_eps={target_eps:>5} -> sigma={sig:.4f} (actual eps={actual:.4f})")

    return sigma_map


# ---------------------------------------------------------------------------
# Result persistence
# ---------------------------------------------------------------------------

def load_results(path):
    """Load existing results from JSON, returning [] if file missing/corrupt."""
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        print(f"  [WARN] Corrupted results file, starting fresh: {path}")
        return []


def atomic_write_json(path, data):
    """Write JSON atomically: write to temp file, then replace."""
    dir_name = os.path.dirname(path) or "."
    fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(tmp_path, path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


def export_csv(results, csv_path):
    """Export results list to CSV."""
    if not results:
        return
    fieldnames = ["dataset", "method", "sigma", "target_epsilon", "epsilon",
                  "accuracy", "final_C", "status", "error_message"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------

def parse_metrics_from_output(output):
    """Extract epsilon, accuracy, and final_C from training script stdout.

    Handles both MNIST output format (ε = X.XX) and Fashion-MNIST format
    (eps = X.XX).
    """
    metrics = {"epsilon": None, "accuracy": None, "final_C": None}

    # Epsilon: match both "ε = X.XX" and "eps = X.XX", take LAST occurrence
    eps_matches = re.findall(r"(?:ε|eps)\s*=\s*([\d.]+)", output)
    if eps_matches:
        metrics["epsilon"] = float(eps_matches[-1])

    # Accuracy: take the LAST occurrence
    acc_matches = re.findall(
        r"Accuracy:\s*\d+/\d+\s*\(\s*([\d.]+)%\s*\)", output
    )
    if acc_matches:
        metrics["accuracy"] = float(acc_matches[-1]) / 100.0

    # Final C: take the LAST occurrence
    c_matches = re.findall(r"Final adaptive C:\s*([\d.]+)", output)
    if c_matches:
        metrics["final_C"] = float(c_matches[-1])

    return metrics


# ---------------------------------------------------------------------------
# Task ID and result lookup
# ---------------------------------------------------------------------------

def make_task_id(dataset, method, sigma, target_epsilon=None):
    """Unique identifier for a task."""
    ds_prefix = "fashion" if dataset == "fashion" else "mnist"
    if target_epsilon is not None:
        return f"{ds_prefix}_{method}_eps{target_epsilon}_sigma{sigma:.4f}"
    return f"{ds_prefix}_{method}_sigma{sigma}"


def find_existing_result(results, dataset, method, sigma):
    """Check if a successful result already exists for (dataset, method, sigma)."""
    for r in results:
        if (r.get("dataset") == dataset
                and r["method"] == method
                and abs(r["sigma"] - sigma) < 1e-6
                and r["status"] == "success"):
            return r
    return None


# ---------------------------------------------------------------------------
# Command construction
# ---------------------------------------------------------------------------

def build_command(dataset, method, sigma, cfg, python_executable=sys.executable):
    """Build the subprocess command list for a given dataset, method and sigma."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    delta = cfg["delta"] or (1.0 / cfg["n_train"])

    if dataset == "mnist":
        if method == "baseline":
            script = os.path.join(script_dir, "minst_baseline.py")
            cmd = [
                python_executable, script,
                "-n", str(cfg["epochs"]),
                "-b", str(cfg["batch_size"]),
                "--lr", str(cfg["lr"]),
                "--sigma", str(sigma),
                "-c", str(cfg["fixed_c"]),
                "--delta", str(delta),
                "--device", DEVICE,
            ]
        elif method in ("ratio", "mse"):
            script = os.path.join(script_dir, "minst_adaptive_histogram.py")
            cmd = [
                python_executable, script,
                "--mode", method,
                "-n", str(cfg["epochs"]),
                "-b", str(cfg["batch_size"]),
                "--lr", str(cfg["lr"]),
                "--sigma", str(sigma),
                "-c", str(cfg["initial_c"]),
                "--delta", str(delta),
                "--device", DEVICE,
            ]
            if method == "ratio":
                cmd.extend(["--target-ratio", str(cfg["target_ratio"])])
        else:
            raise ValueError(f"Unknown method: {method}")

    elif dataset == "fashion":
        # Fashion-MNIST uses a single script with --mode for all methods
        script = os.path.join(script_dir, "fashionmnist_resnet18_dp_baseline.py")
        mode = "fixed" if method == "baseline" else method
        cmd = [
            python_executable, script,
            "--mode", mode,
            "-n", str(cfg["epochs"]),
            "-b", str(cfg["batch_size"]),
            "--lr", str(cfg["lr"]),
            "--sigma", str(sigma),
            "--device", DEVICE,
        ]
        if method == "baseline":
            cmd.extend(["-c", str(cfg["fixed_c"])])
        else:
            cmd.extend(["--initial-c", str(cfg["initial_c"])])
            if method == "ratio":
                cmd.extend(["--target-ratio", str(cfg["target_ratio"])])
        if cfg.get("max_physical_batch_size"):
            cmd.extend(["--max-physical-batch-size",
                        str(cfg["max_physical_batch_size"])])
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    return cmd


# ---------------------------------------------------------------------------
# Single task execution
# ---------------------------------------------------------------------------

def run_task(dataset, method, sigma, cfg, log_dir, python_executable,
             target_epsilon=None):
    """Run a single (dataset, method, sigma) experiment as a subprocess."""
    task_id = make_task_id(dataset, method, sigma, target_epsilon)
    log_path = os.path.join(log_dir, f"{task_id}.log")
    cmd = build_command(dataset, method, sigma, cfg, python_executable)

    print(f"  Command: {' '.join(cmd)}")
    print(f"  Log: {log_path}")

    result = {
        "dataset": dataset,
        "method": method,
        "sigma": round(sigma, 6),
        "target_epsilon": target_epsilon,
        "task_id": task_id,
        "epsilon": None,
        "accuracy": None,
        "final_C": None,
        "status": "pending",
        "error_message": None,
    }

    # For baseline, final_C is the fixed C
    if method == "baseline":
        result["final_C"] = cfg["fixed_c"]

    timeout = cfg["timeout"]

    try:
        with open(log_path, "w", encoding="utf-8") as log_file:
            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=timeout,
            )
            output = proc.stdout
            log_file.write(output)
            log_file.flush()

        # Check return code
        if proc.returncode != 0:
            result["status"] = "failed"
            tail = output.strip().split("\n")[-3:] if output else ["(no output)"]
            result["error_message"] = f"Exit code {proc.returncode}: " + " | ".join(tail)
            return result

        # Parse metrics
        metrics = parse_metrics_from_output(output)
        result["epsilon"] = metrics["epsilon"]
        result["accuracy"] = metrics["accuracy"]
        if metrics["final_C"] is not None:
            result["final_C"] = metrics["final_C"]

        # Validate
        if metrics["accuracy"] is None:
            result["status"] = "failed"
            result["error_message"] = "Could not parse accuracy from output"
        elif metrics["epsilon"] is None:
            result["status"] = "failed"
            result["error_message"] = "Could not parse epsilon from output"
        else:
            result["status"] = "success"

    except subprocess.TimeoutExpired:
        result["status"] = "failed"
        result["error_message"] = f"Timeout after {timeout}s"
    except FileNotFoundError as e:
        result["status"] = "failed"
        result["error_message"] = f"Script not found: {e}"
    except Exception as e:
        result["status"] = "failed"
        result["error_message"] = f"{type(e).__name__}: {e}"

    return result


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Automated DP-SGD experiment runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Dataset
    parser.add_argument(
        "--dataset", type=str, default="mnist",
        choices=["mnist", "fashion"],
        help="Dataset to train on",
    )

    # Sweep mode
    parser.add_argument(
        "--sweep_mode", type=str, default="epsilon",
        choices=["sigma", "epsilon"],
        help="Sweep mode: 'sigma' iterates over SIGMA_LIST, "
             "'epsilon' computes sigma for each target epsilon",
    )
    parser.add_argument(
        "--epsilons", type=float, nargs="+", default=None,
        help="Target epsilon values (only used with --sweep_mode epsilon). "
             "Default: [1, 2, 4, 8, 16, 32]",
    )

    # Execution control
    parser.add_argument(
        "--resume", action="store_true", default=True,
        help="Skip tasks that already have successful results (default: on)",
    )
    parser.add_argument(
        "--no-resume", dest="resume", action="store_false",
        help="Disable resume, rerun all",
    )
    parser.add_argument(
        "--force", action="store_true", default=False,
        help="Ignore existing results and rerun everything",
    )
    parser.add_argument(
        "--max_tasks", type=int, default=None,
        help="Limit number of tasks to run (for debugging)",
    )

    # Output
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Directory for results.json, results.csv, and logs/. "
             "Default: experiment_results_{dataset}",
    )

    # Runtime
    parser.add_argument(
        "--python", type=str, default=sys.executable,
        help="Python interpreter to use for subprocess calls",
    )
    parser.add_argument(
        "--device", type=str, default=DEVICE,
        help="Device override (cuda or cpu)",
    )
    parser.add_argument(
        "--fixed_c", type=float, default=None,
        help="Override fixed C for baseline method",
    )
    parser.add_argument(
        "--batch_size", type=int, default=None,
        help="Override batch size",
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Override number of epochs",
    )
    parser.add_argument(
        "--retry", type=int, default=1,
        help="Number of retries for failed tasks (0 = no retry)",
    )
    args = parser.parse_args()

    # Build dataset config (copy to avoid mutating the global)
    cfg = dict(DATASET_CONFIGS[args.dataset])

    # Apply overrides
    import run_all_experiments as _self
    _self.DEVICE = args.device
    if args.fixed_c is not None:
        cfg["fixed_c"] = args.fixed_c
    if args.batch_size is not None:
        cfg["batch_size"] = args.batch_size
    if args.epochs is not None:
        cfg["epochs"] = args.epochs

    force = args.force or not args.resume

    # Set up output directories
    output_dir = args.output_dir or f"experiment_results_{args.dataset}"
    os.makedirs(output_dir, exist_ok=True)
    log_dir = os.path.join(output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    results_path = os.path.join(output_dir, "results.json")
    csv_path = os.path.join(output_dir, "results.csv")

    # Load existing results
    results = load_results(results_path)
    print(f"Loaded {len(results)} existing results from {results_path}")

    # ---- Build task list based on sweep mode ----
    if args.sweep_mode == "epsilon":
        epsilon_list = args.epsilons if args.epsilons else EPSILON_LIST
        sigma_map = build_sigma_map(epsilon_list, cfg)

        # Task list: (method, sigma, target_epsilon)
        all_tasks = []
        for target_eps in epsilon_list:
            sig, _ = sigma_map[target_eps]
            for method in METHODS:
                all_tasks.append((method, sig, target_eps))
    else:
        # sigma mode: no target_epsilon
        all_tasks = []
        for method in METHODS:
            for sigma in SIGMA_LIST:
                all_tasks.append((method, sigma, None))

    # Filter: skip completed tasks unless --force
    tasks_to_run = []
    for method, sigma, target_eps in all_tasks:
        if not force and find_existing_result(results, args.dataset, method, sigma):
            tid = make_task_id(args.dataset, method, sigma, target_eps)
            print(f"  SKIP {tid} (already completed)")
        else:
            tasks_to_run.append((method, sigma, target_eps))

    # If --force, clear old successful results for this dataset
    if force:
        results = [r for r in results
                   if not (r.get("dataset") == args.dataset
                           and r["status"] == "success")]

    # Apply max_tasks limit
    if args.max_tasks is not None:
        tasks_to_run = tasks_to_run[:args.max_tasks]

    total = len(tasks_to_run)
    if total == 0:
        print("All tasks already completed. Use --force to rerun.")
        export_csv(results, csv_path)
        print(f"Results CSV: {csv_path}")
        _print_summary(results)
        return

    # Print experiment header
    if args.sweep_mode == "epsilon":
        eps_str = ", ".join(str(e) for e in epsilon_list)
        sweep_info = f"  Target epsilons: [{eps_str}]"
    else:
        sweep_info = f"  Sigma values: {SIGMA_LIST}"

    print(f"\n{'='*60}")
    print(f"  Experiment Runner: {total} tasks to run")
    print(f"  Dataset: {args.dataset}")
    print(f"  Sweep mode: {args.sweep_mode}")
    print(f"  Methods: {METHODS}")
    print(sweep_info)
    print(f"  Epochs: {cfg['epochs']}, Batch: {cfg['batch_size']}, LR: {cfg['lr']}")
    print(f"  Baseline fixed C: {cfg['fixed_c']}, Adaptive initial C: {cfg['initial_c']}")
    print(f"  Device: {DEVICE}, Timeout: {cfg['timeout']}s per task")
    if cfg.get("max_physical_batch_size"):
        print(f"  Max physical batch size: {cfg['max_physical_batch_size']}")
    print(f"  Retry: {args.retry}")
    print(f"{'='*60}\n")

    # Run tasks
    completed = 0
    failed = 0
    skipped = len(all_tasks) - total

    for idx, (method, sigma, target_eps) in enumerate(tasks_to_run, 1):
        task_id = make_task_id(args.dataset, method, sigma, target_eps)
        print(f"\n[{idx}/{total}] {task_id}")

        result = run_task(args.dataset, method, sigma, cfg, log_dir,
                          args.python, target_eps)

        # Retry logic
        retry_count = 0
        while result["status"] == "failed" and retry_count < args.retry:
            retry_count += 1
            print(f"  RETRY {retry_count}/{args.retry} for {task_id}...")
            time.sleep(2)
            result = run_task(args.dataset, method, sigma, cfg, log_dir,
                              args.python, target_eps)

        # Update counters
        if result["status"] == "success":
            completed += 1
            eps_str = f"eps={result['epsilon']:.2f}" if result['epsilon'] else "eps=?"
            acc_str = f"{result['accuracy']*100:.2f}%" if result['accuracy'] else "?"
            c_str = f"C={result['final_C']:.4f}" if result['final_C'] else "C=?"
            print(f"  OK  {eps_str} | {acc_str} | {c_str}")
        else:
            failed += 1
            print(f"  FAILED: {result['error_message']}")

        # Persist immediately: remove old entry for same (dataset, method, sigma), then append
        results = [
            r for r in results
            if not (r.get("dataset") == args.dataset
                    and r["method"] == method
                    and abs(r["sigma"] - sigma) < 1e-6)
        ]
        results.append(result)
        atomic_write_json(results_path, results)
        export_csv(results, csv_path)

    # Final summary
    print(f"\n{'='*60}")
    print(f"  DONE: {completed} succeeded, {failed} failed, {skipped} skipped")
    print(f"  Results: {results_path}")
    print(f"  CSV:     {csv_path}")
    print(f"  Logs:    {log_dir}/")
    print(f"{'='*60}")

    _print_summary(results)


def _print_summary(results):
    """Print a formatted summary table, sorted by epsilon when available."""
    if not results:
        return

    has_target_eps = any(r.get("target_epsilon") is not None for r in results)

    if has_target_eps:
        method_order = {m: i for i, m in enumerate(METHODS)}

        def sort_key(r):
            te = r.get("target_epsilon") or 0
            mo = method_order.get(r["method"], 99)
            ds = r.get("dataset", "mnist")
            return (ds, te, mo)

        sorted_results = sorted(results, key=sort_key)

        print("\n--- Results Summary (epsilon sweep) ---")
        print(f"{'Dataset':<9} {'Method':<10} {'TargetEps':<10} {'Sigma':<8} "
              f"{'ActualEps':<10} {'Accuracy':<10} {'Final C':<8} {'Status':<8}")
        print("-" * 80)
        for r in sorted_results:
            ds = r.get("dataset", "mnist")[:8]
            te = f"{r['target_epsilon']}" if r.get("target_epsilon") is not None else "-"
            sig = f"{r['sigma']:.4f}"
            eps = f"{r['epsilon']:.2f}" if r.get("epsilon") else "-"
            acc = f"{r['accuracy']*100:.2f}%" if r.get("accuracy") else "-"
            c = f"{r['final_C']:.4f}" if r.get("final_C") else "-"
            print(f"{ds:<9} {r['method']:<10} {te:<10} {sig:<8} "
                  f"{eps:<10} {acc:<10} {c:<8} {r['status']:<8}")
    else:
        sorted_results = sorted(
            results,
            key=lambda x: (x.get("dataset", "mnist"),
                           METHODS.index(x["method"]) if x["method"] in METHODS else 99,
                           x["sigma"]),
        )

        print("\n--- Results Summary (sigma sweep) ---")
        print(f"{'Dataset':<9} {'Method':<10} {'Sigma':<8} {'Epsilon':<10} "
              f"{'Accuracy':<10} {'Final C':<8} {'Status':<8}")
        print("-" * 68)
        for r in sorted_results:
            ds = r.get("dataset", "mnist")[:8]
            eps = f"{r['epsilon']:.2f}" if r.get("epsilon") else "-"
            acc = f"{r['accuracy']*100:.2f}%" if r.get("accuracy") else "-"
            c = f"{r['final_C']:.4f}" if r.get("final_C") else "-"
            print(f"{ds:<9} {r['method']:<10} {r['sigma']:<8.2f} {eps:<10} "
                  f"{acc:<10} {c:<8} {r['status']:<8}")


if __name__ == "__main__":
    main()
