import os
import pickle
import zipfile

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def load_torch_plain_dict(path):
    """Load a torch.save dict that contains only pickle-serializable objects."""
    with zipfile.ZipFile(path) as zf:
        data_name = next(name for name in zf.namelist() if name.endswith("data.pkl"))
        return pickle.loads(zf.read(data_name))


def main():
    result_path = (
        "dp_histogram_mainline/results/"
        "adaptive_histogram_results_mnist_adaptive_mse_0.1_2.0_1.0_64_10_hist0.025.pt"
    )
    out_dir = "ppt_assets"
    os.makedirs(out_dir, exist_ok=True)

    data = load_torch_plain_dict(result_path)
    epochs = np.arange(1, len(data["mse_history"]) + 1)
    c_history = np.asarray(data["c_history"][1:], dtype=float)
    mse = np.asarray(data["mse_history"], dtype=float)
    bias = np.asarray(data["bias_history"], dtype=float)
    var = np.asarray(data["var_history"], dtype=float)

    plt.rcParams.update(
        {
            "font.size": 13,
            "axes.titlesize": 17,
            "axes.labelsize": 14,
            "legend.fontsize": 12,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "font.family": "DejaVu Sans",
        }
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.2, 5.6))
    fig.patch.set_facecolor("white")
    for ax in (ax1, ax2):
        ax.set_facecolor("#fbfcfe")
        ax.grid(True, color="#94a3b8", alpha=0.25, linewidth=1)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    ax1.plot(epochs, mse, color="#0f766e", marker="o", linewidth=3, label="MSE")
    ax1.plot(epochs, bias, color="#f97316", marker="s", linewidth=2.5, label="Bias")
    ax1.plot(epochs, var, color="#6366f1", marker="^", linewidth=2.5, label="Variance")
    ax1.set_title("Real MSE Decomposition During Training")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Objective value")
    ax1.legend(frameon=False)

    ax2.plot(epochs, c_history, color="#2563eb", marker="o", linewidth=3, label="Selected C")
    ax2.set_title("Selected Clipping Threshold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Clipping threshold C")
    ax2.legend(frameon=False)

    subtitle = (
        "MNIST MSE controller, "
        f"sigma={data['args']['sigma']}, batch={data['args']['batch_size']}, "
        f"epsilon_hist={data['epsilon_hist_per_epoch']}"
    )
    fig.suptitle(subtitle, y=1.02, fontsize=13, color="#475569")
    fig.tight_layout()

    png_path = os.path.join(out_dir, "real_mse_history_mnist_sigma2.png")
    svg_path = os.path.join(out_dir, "real_mse_history_mnist_sigma2.svg")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    print(png_path)
    print(svg_path)


if __name__ == "__main__":
    main()
