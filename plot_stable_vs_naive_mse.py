from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


DATA_PATH = Path("target_epsilon_method_comparison.csv")
OUT_DIR = Path("figures")
DATASETS = ["MNIST", "Fashion-MNIST", "CIFAR-10"]
EPSILONS = [1, 2, 4, 8, 16, 32]
POS_COLOR = "#E45756"
NEG_COLOR = "#54A24B"
Y_LIMITS = (-3, 7)


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
    for col in ["target_epsilon", "accuracy_percent"]:
        df[col] = pd.to_numeric(df[col])

    naive = (
        df[df["method"] == "mse_naive"][
            ["dataset", "target_epsilon", "accuracy_percent"]
        ]
        .rename(columns={"accuracy_percent": "naive_accuracy"})
    )
    stable = (
        df[df["method"] == "mse_stable"][
            ["dataset", "target_epsilon", "accuracy_percent"]
        ]
        .rename(columns={"accuracy_percent": "stable_accuracy"})
    )
    out = stable.merge(naive, on=["dataset", "target_epsilon"], how="inner")
    out["accuracy_delta"] = out["stable_accuracy"] - out["naive_accuracy"]
    return out


def style_axis(ax, dataset, show_ylabel=True):
    ax.set_title(dataset, fontsize=28, fontweight="bold", pad=18)
    ax.set_xlabel("Target epsilon", fontsize=24, fontweight="bold")
    if show_ylabel:
        ax.set_ylabel("Stable MSE - Naive MSE (%)", fontsize=20, fontweight="bold")
    else:
        ax.set_ylabel("")
    ax.set_ylim(*Y_LIMITS)
    ax.set_yticks([-3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7])
    ax.grid(True, which="major", axis="y", alpha=0.85)
    ax.axhline(0, color="#333333", linewidth=1.25, zorder=3)
    for tick_label in ax.get_xticklabels() + ax.get_yticklabels():
        tick_label.set_fontweight("bold")


def plot_bars(ax, subset):
    values = []
    for eps in EPSILONS:
        row = subset[subset["target_epsilon"] == eps]
        values.append(float(row.iloc[0]["accuracy_delta"]))

    colors = [POS_COLOR if value >= 0 else NEG_COLOR for value in values]
    bars = ax.bar(
        range(len(EPSILONS)),
        values,
        color=colors,
        edgecolor="white",
        linewidth=0.8,
        width=0.62,
        zorder=2,
    )
    ax.set_xticks(range(len(EPSILONS)))
    ax.set_xticklabels([str(eps) for eps in EPSILONS])

    for bar, value in zip(bars, values):
        if abs(value) < 0.01:
            continue
        va = "bottom" if value >= 0 else "top"
        offset = 0.12 if value >= 0 else -0.12
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + offset,
            f"{value:+.2f}",
            ha="center",
            va=va,
            fontsize=16,
            color="#222222",
        )


def add_legend(ax):
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=POS_COLOR, label="Stable better"),
        plt.Rectangle((0, 0), 1, 1, color=NEG_COLOR, label="Naive better"),
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=14)


def save_figure(fig, stem):
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(OUT_DIR / f"{stem}.{ext}", bbox_inches="tight")


def save_single_dataset_figures(df):
    filenames = {
        "MNIST": "stable_vs_naive_mse_mnist",
        "Fashion-MNIST": "stable_vs_naive_mse_fashion_mnist",
        "CIFAR-10": "stable_vs_naive_mse_cifar10",
    }
    for dataset in DATASETS:
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        subset = df[df["dataset"] == dataset]
        plot_bars(ax, subset)
        style_axis(ax, dataset, show_ylabel=True)
        add_legend(ax)
        fig.tight_layout()
        save_figure(fig, filenames[dataset])
        plt.close(fig)


def save_summary_figure(df):
    fig, axes = plt.subplots(1, 3, figsize=(15.4, 4.2), sharex=False, sharey=True)
    for index, (ax, dataset) in enumerate(zip(axes, DATASETS)):
        subset = df[df["dataset"] == dataset]
        plot_bars(ax, subset)
        style_axis(ax, dataset, show_ylabel=(index == 0))

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=POS_COLOR, label="Stable better"),
        plt.Rectangle((0, 0), 1, 1, color=NEG_COLOR, label="Naive better"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.5, -0.04),
        fontsize=14,
        frameon=True,
        framealpha=0.95,
        edgecolor="#DDDDDD",
    )
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    save_figure(fig, "stable_vs_naive_mse_all_datasets")
    plt.close(fig)


def main():
    setup_matplotlib()
    OUT_DIR.mkdir(exist_ok=True)
    df = read_data()
    save_single_dataset_figures(df)
    save_summary_figure(df)

    print("Saved Stable-vs-Naive MSE figures:")
    for name in [
        "stable_vs_naive_mse_mnist",
        "stable_vs_naive_mse_fashion_mnist",
        "stable_vs_naive_mse_cifar10",
        "stable_vs_naive_mse_all_datasets",
    ]:
        print(f"  {OUT_DIR / (name + '.png')}")
        print(f"  {OUT_DIR / (name + '.pdf')}")
        print(f"  {OUT_DIR / (name + '.svg')}")


if __name__ == "__main__":
    main()
