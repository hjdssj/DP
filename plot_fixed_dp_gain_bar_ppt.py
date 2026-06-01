from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


DATA_PATH = Path("target_epsilon_method_comparison.csv")
OUT_DIR = Path("ppt_assets") / "fixed_dp_gain_bar"

DATASETS = ["MNIST", "Fashion-MNIST", "CIFAR-10"]
METHODS = ["ratio", "mse_naive", "mse_stable"]
METHOD_LABELS = {
    "ratio": "Ratio",
    "mse_naive": "Naive MSE",
    "mse_stable": "Stable MSE",
}
METHOD_COLORS = {
    "ratio": "#63A65F",
    "mse_naive": "#4E79A7",
    "mse_stable": "#1F6B45",
}
EPSILONS = [1, 2, 4, 8, 16, 32]
Y_LIMITS = (-8, 7)


def setup_matplotlib():
    plt.rcParams.update(
        {
            "font.family": ["Arial", "Microsoft YaHei", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "axes.linewidth": 0.9,
            "axes.edgecolor": "#333333",
            "axes.labelcolor": "#222222",
            "xtick.labelsize": 18,
            "ytick.labelsize": 18,
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

    fixed = (
        df[df["method"] == "fixed_dp"][
            ["dataset", "target_epsilon", "accuracy_percent"]
        ]
        .rename(columns={"accuracy_percent": "fixed_accuracy"})
    )
    gains = df[df["method"].isin(METHODS)].merge(
        fixed,
        on=["dataset", "target_epsilon"],
        how="left",
    )
    gains["accuracy_gain"] = gains["accuracy_percent"] - gains["fixed_accuracy"]
    return gains


def style_axis(ax, dataset, show_ylabel=True):
    ax.set_title(dataset, fontsize=26, fontweight="bold", pad=16)
    ax.set_xlabel("Target epsilon", fontsize=22, fontweight="bold")
    if show_ylabel:
        ax.set_ylabel("Accuracy gain (%)", fontsize=20, fontweight="bold")
    else:
        ax.set_ylabel("")
    ax.set_ylim(*Y_LIMITS)
    ax.set_yticks([-8, -6, -4, -2, 0, 2, 4, 6])
    ax.grid(True, which="major", axis="y", alpha=0.85)
    ax.axhline(0, color="#6B7280", linewidth=1.25, zorder=3)
    for tick_label in ax.get_xticklabels() + ax.get_yticklabels():
        tick_label.set_fontweight("bold")


def plot_bars(ax, subset, width=0.24):
    x = list(range(len(EPSILONS)))
    offsets = [-width, 0, width]

    for method, offset in zip(METHODS, offsets):
        values = []
        for eps in EPSILONS:
            row = subset[
                (subset["method"] == method) & (subset["target_epsilon"] == eps)
            ]
            values.append(float(row.iloc[0]["accuracy_gain"]))
        ax.bar(
            [pos + offset for pos in x],
            values,
            width=width,
            label=METHOD_LABELS[method],
            color=METHOD_COLORS[method],
            edgecolor="white",
            linewidth=0.8,
            zorder=2,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([str(eps) for eps in EPSILONS])


def save_figure(fig, stem):
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(OUT_DIR / f"{stem}.{ext}", bbox_inches="tight")


def save_single_dataset_figures(df):
    filenames = {
        "MNIST": "fixed_dp_gain_bar_mnist",
        "Fashion-MNIST": "fixed_dp_gain_bar_fashion_mnist",
        "CIFAR-10": "fixed_dp_gain_bar_cifar10",
    }
    for dataset in DATASETS:
        fig, ax = plt.subplots(figsize=(12.6, 4.8))
        subset = df[df["dataset"] == dataset]
        plot_bars(ax, subset, width=0.16)
        style_axis(ax, dataset, show_ylabel=True)
        ax.legend(loc="upper left", fontsize=13)
        fig.tight_layout()
        save_figure(fig, filenames[dataset])
        plt.close(fig)


def save_summary_figure(df):
    fig, axes = plt.subplots(1, 3, figsize=(20.0, 4.8), sharex=False, sharey=True)
    for index, (ax, dataset) in enumerate(zip(axes, DATASETS)):
        subset = df[df["dataset"] == dataset]
        plot_bars(ax, subset)
        style_axis(ax, dataset, show_ylabel=(index == 0))

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, -0.04),
        fontsize=14,
        frameon=True,
        framealpha=0.95,
        edgecolor="#DDDDDD",
    )
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    save_figure(fig, "fixed_dp_gain_bar_all_datasets")
    plt.close(fig)


def main():
    setup_matplotlib()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = read_data()
    save_single_dataset_figures(df)
    save_summary_figure(df)

    print("Saved PPT fixed-DP gain bar figures:")
    for name in [
        "fixed_dp_gain_bar_mnist",
        "fixed_dp_gain_bar_fashion_mnist",
        "fixed_dp_gain_bar_cifar10",
        "fixed_dp_gain_bar_all_datasets",
    ]:
        print(f"  {OUT_DIR / (name + '.png')}")
        print(f"  {OUT_DIR / (name + '.pdf')}")
        print(f"  {OUT_DIR / (name + '.svg')}")


if __name__ == "__main__":
    main()
