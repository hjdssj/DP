#!/usr/bin/env python3
"""Generate epsilon-Accuracy trade-off figures from validated results."""

from pathlib import Path

import matplotlib.pyplot as plt
import torch


FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)
RESULTS_DIR = Path("dp_histogram_mainline") / "results"
DATASET_ORDER = ["MNIST", "Fashion-MNIST", "CIFAR-10"]
METHOD_STYLES = {
    "Fixed C": {"color": "#4C78A8", "marker": "o", "linestyle": "-", "label": "Fixed C"},
    "MSE + DP hist": {
        "color": "#F58518",
        "marker": "s",
        "linestyle": "-",
        "label": "MSE + DP hist",
    },
}
MAINLINE_EPOCHS = {
    "MNIST": 10,
    "Fashion-MNIST": 20,
    "CIFAR-10": 20,
}
MAINLINE_SIGMAS = {
    "MNIST": {0.5, 0.8, 1.0, 1.2, 1.5, 2.0},
    "Fashion-MNIST": {3.383908, 1.929262, 1.212828, 0.856756, 0.647915, 0.503971},
    "CIFAR-10": {1.5, 1.2, 1.0, 0.8, 0.6, 0.5},
}


def pct(values):
    return [v * 100 if v <= 1 else v for v in values]


def savefig(name):
    path = FIG_DIR / name
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


def plot_latest_mainline():
    mnist_fixed_eps = [0.8234, 0.5034, 0.2714]
    mnist_fixed_acc = [92.84, 92.01, 84.96]
    mnist_mse_eps = [1.0734, 0.7534, 0.5214]
    mnist_mse_acc = [94.91, 94.02, 92.93]

    cifar_fixed_eps = [3.7820, 6.1836]
    cifar_fixed_acc = [49.89, 55.60]
    cifar_mse_eps = [4.2820, 7.4336]
    cifar_mse_acc = [51.87, 55.05]

    fashion_fixed_eps = [5.1224]
    fashion_fixed_acc = [83.10]
    fashion_mse_eps = [5.6224]
    fashion_mse_acc = [84.34]

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8))

    axes[0].plot(mnist_fixed_eps, mnist_fixed_acc, "o-", label="Fixed C")
    axes[0].plot(mnist_mse_eps, mnist_mse_acc, "s-", label="MSE + DP hist")
    axes[0].set_title("MNIST")

    axes[1].scatter(fashion_fixed_eps, fashion_fixed_acc, marker="o", label="Fixed C")
    axes[1].scatter(fashion_mse_eps, fashion_mse_acc, marker="s", label="MSE + DP hist")
    axes[1].set_title("Fashion-MNIST")

    axes[2].plot(cifar_fixed_eps, cifar_fixed_acc, "o-", label="Fixed C")
    axes[2].plot(cifar_mse_eps, cifar_mse_acc, "s-", label="MSE + DP hist")
    axes[2].set_title("CIFAR-10")

    for ax in axes:
        ax.set_xlabel("Total epsilon")
        ax.set_ylabel("Accuracy (%)")
        ax.grid(True, alpha=0.3)
        ax.legend()

    savefig("latest_dp_hist_tradeoff.png")


def plot_mnist_sweep():
    eps = [0.76, 1.43, 3.19, 6.74, 13.82, 28.09]
    baseline = [93.67, 95.01, 95.77, 95.64, 96.14, 95.93]
    ratio = [89.33, 93.84, 94.77, 95.99, 96.49, 96.92]
    mse = [95.15, 95.61, 95.74, 96.33, 96.84, 97.10]

    plt.figure(figsize=(6, 4))
    plt.plot(eps, baseline, "o-", label="Fixed C")
    plt.plot(eps, ratio, "^-", label="Ratio")
    plt.plot(eps, mse, "s-", label="MSE")
    plt.xscale("log")
    plt.xlabel("Actual epsilon")
    plt.ylabel("Accuracy (%)")
    plt.title("MNIST epsilon-Accuracy sweep")
    plt.grid(True, alpha=0.3)
    plt.legend()
    savefig("mnist_epsilon_accuracy_sweep.png")


def plot_fashion_bs2048_sweep():
    eps = [0.91, 1.81, 3.60, 7.10, 14.14, 28.45]
    baseline = [80.13, 82.06, 83.06, 82.72, 82.73, 82.95]
    ratio = [78.77, 81.92, 83.65, 85.47, 86.82, 86.85]
    mse = [75.19, 78.35, 83.18, 86.37, 88.07, 88.43]

    plt.figure(figsize=(6, 4))
    plt.plot(eps, baseline, "o-", label="Fixed C")
    plt.plot(eps, ratio, "^-", label="Ratio")
    plt.plot(eps, mse, "s-", label="MSE")
    plt.xscale("log")
    plt.xlabel("Actual epsilon")
    plt.ylabel("Accuracy (%)")
    plt.title("Fashion-MNIST bs=2048 epsilon-Accuracy sweep")
    plt.grid(True, alpha=0.3)
    plt.legend()
    savefig("fashion_bs2048_epsilon_accuracy_sweep.png")


def plot_cifar_dcsgd_e():
    paper_eps = [0.91, 1.79, 3.44, 6.89, 13.94, 28.17]
    paper_acc = [36.95, 41.17, 45.64, 47.47, 48.85, 51.51]
    ours_eps = [8.00, 16.00, 32.00]
    ours_acc = [45.27, 49.61, 48.33]
    main_eps = [3.7820, 4.2820, 6.1836, 7.4336]
    main_acc = [49.89, 51.87, 55.60, 55.05]

    plt.figure(figsize=(6.5, 4.2))
    plt.plot(paper_eps, paper_acc, "o-", label="DC-SGD-E paper")
    plt.plot(ours_eps, ours_acc, "^-", label="DC-SGD-E ours")
    plt.scatter(main_eps, main_acc, marker="s", label="Current mainline")
    plt.xscale("log")
    plt.xlabel("Epsilon")
    plt.ylabel("Accuracy (%)")
    plt.title("CIFAR-10 epsilon-Accuracy comparison")
    plt.grid(True, alpha=0.3)
    plt.legend()
    savefig("cifar10_epsilon_accuracy_comparison.png")


def infer_dataset(path):
    name = path.name
    if "mnist_adaptive" in name:
        return "MNIST"
    if "fashion_resnet18" in name:
        return "Fashion-MNIST"
    if "cifar10" in name:
        return "CIFAR-10"
    return None


def load_mainline_points():
    points = []
    for path in RESULTS_DIR.glob("adaptive_histogram_results_*.pt"):
        # Prefer the new naming scheme with explicit histogram budget.
        if "_hist" not in path.stem:
            continue
        dataset = infer_dataset(path)
        if dataset is None:
            continue
        data = torch.load(path, map_location="cpu", weights_only=False)
        args = data.get("args", {})
        mode = data.get("mode")
        if mode not in {"fixed", "mse"}:
            continue
        acc = (data.get("run_results") or [None])[0]
        eps = data.get("epsilon_total")
        if acc is None or eps is None:
            continue
        points.append({
            "dataset": dataset,
            "method": "Fixed C" if mode == "fixed" else "MSE + DP hist",
            "epsilon": float(eps),
            "accuracy": float(acc) * 100.0,
            "sigma": float(args.get("sigma", 0.0)),
            "epochs": int(args.get("epochs", 0)),
            "batch": int(args.get("batch_size", 0)),
            "c_history": data.get("c_history", []),
            "clipped_ratio_history": data.get("clipped_ratio_history", []),
            "mse_history": data.get("mse_history", []),
            "bias_history": data.get("bias_history", []),
            "var_history": data.get("var_history", []),
            "epsilon_sgd": float(data.get("epsilon_sgd", 0.0) or 0.0),
            "epsilon_hist_total": float(data.get("epsilon_hist_total", 0.0) or 0.0),
        })
    return points


def filter_primary_sweep(points):
    """Keep the thesis mainline sweep and exclude supplemental runs."""
    primary = []
    for p in points:
        dataset = p["dataset"]
        if p["epochs"] != MAINLINE_EPOCHS.get(dataset):
            continue
        if not any(abs(p["sigma"] - sigma) < 1e-6 for sigma in MAINLINE_SIGMAS[dataset]):
            continue
        primary.append(p)
    return primary


def grouped_method_points(points, dataset, method):
    method_points = [
        p
        for p in points
        if p["dataset"] == dataset and p["method"] == method
    ]
    method_points.sort(key=lambda p: p["epsilon"])
    return method_points


def plot_full_mainline_from_results():
    points = filter_primary_sweep(load_mainline_points())
    if not points:
        print("No mainline result .pt files found; skipping full DP histogram plot.")
        return

    datasets = DATASET_ORDER
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8))
    for ax, dataset in zip(axes, datasets):
        subset = [p for p in points if p["dataset"] == dataset]
        for method, marker in [("Fixed C", "o"), ("MSE + DP hist", "s")]:
            method_points = grouped_method_points(subset, dataset, method)
            if not method_points:
                continue
            ax.plot(
                [p["epsilon"] for p in method_points],
                [p["accuracy"] for p in method_points],
                marker + "-",
                label=method,
            )
        ax.set_title(dataset)
        ax.set_xlabel("Total epsilon")
        ax.set_ylabel("Accuracy (%)")
        ax.grid(True, alpha=0.3)
        ax.legend()
    savefig("full_dp_hist_tradeoff_sweeps.png")


def plot_full_mainline_paper():
    points = filter_primary_sweep(load_mainline_points())
    if not points:
        print("No primary sweep result .pt files found; skipping paper DP histogram plot.")
        return

    fig, axes = plt.subplots(3, 1, figsize=(6.8, 8.2), sharex=False)
    handles = []
    labels = []
    for ax, dataset in zip(axes, DATASET_ORDER):
        for method, style in METHOD_STYLES.items():
            method_points = grouped_method_points(points, dataset, method)
            if not method_points:
                continue
            line = ax.plot(
                [p["epsilon"] for p in method_points],
                [p["accuracy"] for p in method_points],
                color=style["color"],
                marker=style["marker"],
                linestyle=style["linestyle"],
                linewidth=1.7,
                markersize=4.2,
                label=style["label"],
            )[0]
            if style["label"] not in labels:
                handles.append(line)
                labels.append(style["label"])
        ax.set_xscale("log")
        ax.set_title(dataset, loc="left", fontsize=11, fontweight="bold")
        ax.set_xlabel("Total epsilon (log scale)")
        ax.set_ylabel("Accuracy (%)")
        ax.grid(True, which="major", alpha=0.28)
        ax.grid(True, which="minor", alpha=0.10)
        ax.tick_params(axis="both", labelsize=9)

    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 1.01),
    )
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    path = FIG_DIR / "full_dp_hist_tradeoff_sweeps_paper.png"
    plt.savefig(path, dpi=240, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


def plot_individual_mainline_paper():
    points = filter_primary_sweep(load_mainline_points())
    if not points:
        print("No primary sweep result .pt files found; skipping individual paper plots.")
        return

    filenames = {
        "MNIST": "mnist_dp_hist_tradeoff_paper.png",
        "Fashion-MNIST": "fashion_dp_hist_tradeoff_paper.png",
        "CIFAR-10": "cifar10_dp_hist_tradeoff_paper.png",
    }
    for dataset in DATASET_ORDER:
        plt.figure(figsize=(5.8, 3.8))
        for method, style in METHOD_STYLES.items():
            method_points = grouped_method_points(points, dataset, method)
            if not method_points:
                continue
            plt.plot(
                [p["epsilon"] for p in method_points],
                [p["accuracy"] for p in method_points],
                color=style["color"],
                marker=style["marker"],
                linestyle=style["linestyle"],
                linewidth=1.8,
                markersize=4.5,
                label=style["label"],
            )
        plt.xscale("log")
        plt.xlabel("Total epsilon (log scale)")
        plt.ylabel("Accuracy (%)")
        plt.title(f"{dataset} DP histogram trade-off")
        plt.grid(True, which="major", alpha=0.30)
        plt.grid(True, which="minor", alpha=0.12)
        plt.legend(frameon=False)
        savefig(filenames[dataset])


def plot_accuracy_delta_paper():
    points = filter_primary_sweep(load_mainline_points())
    if not points:
        print("No primary sweep result .pt files found; skipping accuracy delta plot.")
        return

    fig, axes = plt.subplots(3, 1, figsize=(6.8, 8.0), sharex=False)
    for ax, dataset in zip(axes, DATASET_ORDER):
        fixed = {
            p["sigma"]: p["accuracy"]
            for p in points
            if p["dataset"] == dataset and p["method"] == "Fixed C"
        }
        mse = {
            p["sigma"]: p["accuracy"]
            for p in points
            if p["dataset"] == dataset and p["method"] == "MSE + DP hist"
        }
        sigmas = sorted(set(fixed) & set(mse), reverse=True)
        deltas = [mse[s] - fixed[s] for s in sigmas]
        colors = ["#54A24B" if d >= 0 else "#E45756" for d in deltas]
        labels = [f"{s:.3g}" for s in sigmas]
        ax.bar(labels, deltas, color=colors, width=0.62)
        ax.axhline(0, color="#333333", linewidth=0.9)
        ax.set_title(dataset, loc="left", fontsize=11, fontweight="bold")
        ax.set_xlabel("Noise multiplier sigma")
        ax.set_ylabel("Accuracy gain (pp)")
        ax.grid(True, axis="y", alpha=0.25)
        for idx, delta in enumerate(deltas):
            va = "bottom" if delta >= 0 else "top"
            offset = 0.22 if delta >= 0 else -0.22
            ax.text(
                idx,
                delta + offset,
                f"{delta:+.2f}",
                ha="center",
                va=va,
                fontsize=8,
            )
        lower = min(deltas + [0])
        upper = max(deltas + [0])
        pad = max((upper - lower) * 0.18, 0.6)
        ax.set_ylim(lower - pad, upper + pad)
    plt.tight_layout()
    path = FIG_DIR / "dp_hist_accuracy_delta_paper.png"
    plt.savefig(path, dpi=240, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


def plot_final_c_vs_sigma_paper():
    points = filter_primary_sweep(load_mainline_points())
    mse_points = [p for p in points if p["method"] == "MSE + DP hist"]
    if not mse_points:
        print("No MSE primary sweep points found; skipping final C plot.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(12.8, 3.4))
    for ax, dataset in zip(axes, DATASET_ORDER):
        dataset_points = [p for p in mse_points if p["dataset"] == dataset and p["c_history"]]
        dataset_points.sort(key=lambda p: p["sigma"], reverse=True)
        sigmas = [p["sigma"] for p in dataset_points]
        final_cs = [p["c_history"][-1] for p in dataset_points]
        ax.plot(
            [f"{s:.3g}" for s in sigmas],
            final_cs,
            color="#F58518",
            marker="s",
            linewidth=1.8,
            markersize=4.5,
        )
        ax.set_title(dataset, fontsize=10, fontweight="bold")
        ax.set_xlabel("Noise multiplier sigma")
        ax.set_ylabel("Final clipping threshold C")
        ax.grid(True, axis="y", alpha=0.28)
    savefig("dp_hist_final_c_vs_sigma_paper.png")


def plot_c_history_representative_paper():
    points = filter_primary_sweep(load_mainline_points())
    targets = {
        "MNIST": 1.5,
        "Fashion-MNIST": 0.647915,
        "CIFAR-10": 1.0,
    }
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 3.4))
    for ax, dataset in zip(axes, DATASET_ORDER):
        target_sigma = targets[dataset]
        candidates = [
            p for p in points
            if p["dataset"] == dataset
            and p["method"] == "MSE + DP hist"
            and abs(p["sigma"] - target_sigma) < 1e-6
            and p["c_history"]
        ]
        if not candidates:
            ax.set_visible(False)
            continue
        point = candidates[0]
        epochs = list(range(0, len(point["c_history"])))
        ax.plot(
            epochs,
            point["c_history"],
            color="#F58518",
            marker="s",
            linewidth=1.7,
            markersize=3.8,
        )
        ax.set_title(f"{dataset}, sigma={target_sigma:g}", fontsize=10, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Clipping threshold C")
        ax.grid(True, alpha=0.28)
    savefig("dp_hist_c_history_representative_paper.png")


def plot_mse_decomposition_representative_paper():
    points = filter_primary_sweep(load_mainline_points())
    targets = {
        "MNIST": 1.5,
        "Fashion-MNIST": 0.647915,
        "CIFAR-10": 1.0,
    }
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 3.4))
    for ax, dataset in zip(axes, DATASET_ORDER):
        target_sigma = targets[dataset]
        candidates = [
            p for p in points
            if p["dataset"] == dataset
            and p["method"] == "MSE + DP hist"
            and abs(p["sigma"] - target_sigma) < 1e-6
            and p["bias_history"]
            and p["var_history"]
        ]
        if not candidates:
            ax.set_visible(False)
            continue
        point = candidates[0]
        epochs = list(range(1, len(point["bias_history"]) + 1))
        ax.plot(
            epochs,
            point["bias_history"],
            color="#54A24B",
            marker="o",
            linewidth=1.5,
            markersize=3.2,
            label="Bias term",
        )
        ax.plot(
            epochs,
            point["var_history"],
            color="#E45756",
            marker="^",
            linewidth=1.5,
            markersize=3.2,
            label="Variance term",
        )
        ax.set_title(f"{dataset}, sigma={target_sigma:g}", fontsize=10, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Estimated term")
        ax.grid(True, alpha=0.28)
        ax.legend(frameon=False, fontsize=8)
    savefig("dp_hist_mse_decomposition_representative_paper.png")


def plot_privacy_budget_composition_paper():
    points = filter_primary_sweep(load_mainline_points())
    mse_points = [p for p in points if p["method"] == "MSE + DP hist"]
    if not mse_points:
        print("No MSE primary sweep points found; skipping privacy budget plot.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(12.8, 3.4))
    for ax, dataset in zip(axes, DATASET_ORDER):
        dataset_points = [p for p in mse_points if p["dataset"] == dataset]
        dataset_points.sort(key=lambda p: p["sigma"], reverse=True)
        labels = [f"{p['sigma']:.3g}" for p in dataset_points]
        eps_sgd = [p["epsilon_sgd"] for p in dataset_points]
        eps_hist = [p["epsilon_hist_total"] for p in dataset_points]
        ax.bar(labels, eps_sgd, color="#4C78A8", width=0.62, label="DP-SGD")
        ax.bar(labels, eps_hist, bottom=eps_sgd, color="#F58518", width=0.62, label="DP histogram")
        ax.set_title(dataset, fontsize=10, fontweight="bold")
        ax.set_xlabel("Noise multiplier sigma")
        ax.set_ylabel("Total epsilon")
        ax.grid(True, axis="y", alpha=0.28)
        ax.legend(frameon=False, fontsize=8)
    savefig("dp_hist_privacy_budget_composition_paper.png")


def main():
    plot_full_mainline_from_results()
    plot_full_mainline_paper()
    plot_individual_mainline_paper()
    plot_accuracy_delta_paper()
    plot_final_c_vs_sigma_paper()
    plot_c_history_representative_paper()
    plot_mse_decomposition_representative_paper()
    plot_privacy_budget_composition_paper()
    plot_latest_mainline()
    plot_mnist_sweep()
    plot_fashion_bs2048_sweep()
    plot_cifar_dcsgd_e()


if __name__ == "__main__":
    main()
