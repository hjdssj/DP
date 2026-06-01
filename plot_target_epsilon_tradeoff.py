from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


DATA_PATH = Path("target_epsilon_method_comparison.csv")
OUT_DIR = Path("figures")

DATASETS = ["MNIST", "Fashion-MNIST", "CIFAR-10"]
METHODS = ["fixed_dp", "ratio", "mse_naive", "mse_stable"]
METHOD_LABELS = {
    "fixed_dp": "Fixed DP",
    "ratio": "Ratio",
    "mse_naive": "Naive MSE",
    "mse_stable": "Stable MSE",
}
METHOD_COLORS = {
    "fixed_dp": "#4C78A8",
    "ratio": "#F58518",
    "mse_naive": "#54A24B",
    "mse_stable": "#E45756",
}
METHOD_MARKERS = {
    "fixed_dp": "o",
    "ratio": "s",
    "mse_naive": "^",
    "mse_stable": "D",
}
METHOD_LINESTYLES = {
    "fixed_dp": "-",
    "ratio": "-",
    "mse_naive": "-",
    "mse_stable": "-",
}
Y_LIMITS = {
    "MNIST": (88, 98),
    "Fashion-MNIST": (74, 90),
    "CIFAR-10": (30, 62),
}


def setup_matplotlib():
    plt.rcParams.update(
        {
            "font.family": ["Arial", "Microsoft YaHei", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "axes.linewidth": 0.9,
            "axes.edgecolor": "#333333",
            "axes.labelcolor": "#222222",
            "xtick.labelsize": 20,
            "ytick.labelsize": 20,
            "xtick.color": "#222222",
            "ytick.color": "#222222",
            "grid.color": "#D9D9D9",
            "grid.linewidth": 0.7,
            "legend.frameon": True,
            "legend.framealpha": 0.95,
            "legend.edgecolor": "#DDDDDD",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def read_data():
    df = pd.read_csv(DATA_PATH)
    for col in ["target_epsilon", "actual_epsilon", "accuracy_percent"]:
        df[col] = pd.to_numeric(df[col])
    return df


def plot_method_lines(ax, subset):
    for method in METHODS:
        method_df = subset[subset["method"] == method].sort_values("actual_epsilon")
        ax.plot(
            method_df["actual_epsilon"],
            method_df["accuracy_percent"],
            label=METHOD_LABELS[method],
            color=METHOD_COLORS[method],
            marker=METHOD_MARKERS[method],
            linestyle=METHOD_LINESTYLES[method],
            linewidth=2.0,
            markersize=5.2,
            markeredgewidth=0,
        )


def style_axis(ax, dataset, show_ylabel=True):
    ax.set_title(dataset, fontsize=28, fontweight="bold", pad=18)
    ax.set_xlabel("Actual epsilon", fontsize=24, fontweight="bold")
    if show_ylabel:
        ax.set_ylabel("Accuracy (%)", fontsize=24, fontweight="bold")
    else:
        ax.set_ylabel("")
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 4, 8, 16, 32])
    ax.set_xticklabels(["1", "2", "4", "8", "16", "32"])
    ax.set_xlim(0.75, 34)
    ax.set_ylim(*Y_LIMITS[dataset])
    ax.grid(True, which="major", axis="both", alpha=0.85)
    for tick_label in ax.get_xticklabels() + ax.get_yticklabels():
        tick_label.set_fontweight("bold")
    ax.legend(loc="lower right", fontsize=14)


def save_figure(fig, stem):
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(OUT_DIR / f"{stem}.{ext}", bbox_inches="tight")


def save_single_dataset_figures(df):
    filenames = {
        "MNIST": "target_epsilon_tradeoff_mnist",
        "Fashion-MNIST": "target_epsilon_tradeoff_fashion_mnist",
        "CIFAR-10": "target_epsilon_tradeoff_cifar10",
    }

    for dataset in DATASETS:
        fig, ax = plt.subplots(figsize=(6.2, 4.2))
        subset = df[df["dataset"] == dataset]
        plot_method_lines(ax, subset)
        style_axis(ax, dataset, show_ylabel=True)
        fig.tight_layout()
        save_figure(fig, filenames[dataset])
        plt.close(fig)


def save_summary_figure(df):
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.2), sharex=False, sharey=False)
    for index, (ax, dataset) in enumerate(zip(axes, DATASETS)):
        subset = df[df["dataset"] == dataset]
        plot_method_lines(ax, subset)
        style_axis(ax, dataset, show_ylabel=(index == 0))

    handles, labels = axes[0].get_legend_handles_labels()
    for ax in axes:
        ax.get_legend().remove()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        bbox_to_anchor=(0.5, -0.04),
        fontsize=14,
        frameon=True,
        framealpha=0.95,
        edgecolor="#DDDDDD",
    )
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    save_figure(fig, "target_epsilon_tradeoff_all_datasets")
    plt.close(fig)


def main():
    setup_matplotlib()
    OUT_DIR.mkdir(exist_ok=True)
    df = read_data()
    save_single_dataset_figures(df)
    save_summary_figure(df)

    print("Saved matplotlib trade-off figures:")
    for name in [
        "target_epsilon_tradeoff_mnist",
        "target_epsilon_tradeoff_fashion_mnist",
        "target_epsilon_tradeoff_cifar10",
        "target_epsilon_tradeoff_all_datasets",
    ]:
        print(f"  {OUT_DIR / (name + '.png')}")
        print(f"  {OUT_DIR / (name + '.pdf')}")
        print(f"  {OUT_DIR / (name + '.svg')}")


if __name__ == "__main__":
    main()
