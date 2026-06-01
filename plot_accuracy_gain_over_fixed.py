from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


DATA_PATH = Path("target_epsilon_method_comparison.csv")
OUT_DIR = Path("figures")

DATASETS = ["MNIST", "Fashion-MNIST", "CIFAR-10"]
METHODS = ["ratio", "mse_naive", "mse_stable"]
METHOD_LABELS = {
    "ratio": "Ratio",
    "mse_naive": "Naive MSE",
    "mse_stable": "Stable MSE",
}
METHOD_COLORS = {
    "ratio": "#F58518",
    "mse_naive": "#54A24B",
    "mse_stable": "#E45756",
}
METHOD_MARKERS = {
    "ratio": "s",
    "mse_naive": "^",
    "mse_stable": "D",
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


def read_gain_data():
    df = pd.read_csv(DATA_PATH)
    for col in ["target_epsilon", "actual_epsilon", "accuracy_percent"]:
        df[col] = pd.to_numeric(df[col])

    fixed = (
        df[df["method"] == "fixed_dp"][
            ["dataset", "target_epsilon", "actual_epsilon", "accuracy_percent"]
        ]
        .rename(columns={"accuracy_percent": "fixed_accuracy"})
        .copy()
    )
    adaptive = df[df["method"].isin(METHODS)].copy()
    merged = adaptive.merge(
        fixed,
        on=["dataset", "target_epsilon", "actual_epsilon"],
        how="left",
    )
    merged["accuracy_gain"] = merged["accuracy_percent"] - merged["fixed_accuracy"]
    return merged


def plot_method_lines(ax, subset):
    for method in METHODS:
        method_df = subset[subset["method"] == method].sort_values("actual_epsilon")
        ax.plot(
            method_df["actual_epsilon"],
            method_df["accuracy_gain"],
            label=METHOD_LABELS[method],
            color=METHOD_COLORS[method],
            marker=METHOD_MARKERS[method],
            linestyle="-",
            linewidth=2.0,
            markersize=5.2,
            markeredgewidth=0,
        )


def style_axis(ax, dataset, show_ylabel=True):
    ax.set_title(dataset, fontsize=12, fontweight="semibold", pad=9)
    ax.set_xlabel("Actual epsilon", fontsize=10)
    if show_ylabel:
        ax.set_ylabel("Accuracy gain over Fixed DP (%)", fontsize=10)
    else:
        ax.set_ylabel("")
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 4, 8, 16, 32])
    ax.set_xticklabels(["1", "2", "4", "8", "16", "32"])
    ax.set_xlim(0.75, 34)
    ax.set_ylim(-8.5, 7.0)
    ax.axhline(0, color="#444444", linewidth=1.2, linestyle="-", zorder=0)
    ax.grid(True, which="major", axis="both", alpha=0.85)
    ax.legend(loc="upper left", fontsize=8.5)


def save_figure(fig, stem):
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(OUT_DIR / f"{stem}.{ext}", bbox_inches="tight")


def save_single_dataset_figures(df):
    filenames = {
        "MNIST": "accuracy_gain_over_fixed_mnist",
        "Fashion-MNIST": "accuracy_gain_over_fixed_fashion_mnist",
        "CIFAR-10": "accuracy_gain_over_fixed_cifar10",
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
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.2), sharex=False, sharey=True)
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
        ncol=3,
        bbox_to_anchor=(0.5, -0.04),
        fontsize=9,
        frameon=True,
        framealpha=0.95,
        edgecolor="#DDDDDD",
    )
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    save_figure(fig, "accuracy_gain_over_fixed_all_datasets")
    plt.close(fig)


def main():
    setup_matplotlib()
    OUT_DIR.mkdir(exist_ok=True)
    df = read_gain_data()
    save_single_dataset_figures(df)
    save_summary_figure(df)

    print("Saved accuracy gain figures:")
    for name in [
        "accuracy_gain_over_fixed_mnist",
        "accuracy_gain_over_fixed_fashion_mnist",
        "accuracy_gain_over_fixed_cifar10",
        "accuracy_gain_over_fixed_all_datasets",
    ]:
        print(f"  {OUT_DIR / (name + '.png')}")
        print(f"  {OUT_DIR / (name + '.pdf')}")
        print(f"  {OUT_DIR / (name + '.svg')}")


if __name__ == "__main__":
    main()
