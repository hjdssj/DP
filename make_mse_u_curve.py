import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def main():
    out_dir = "ppt_assets"
    os.makedirs(out_dir, exist_ok=True)

    c = np.linspace(0.15, 4.5, 400)

    # Conceptual curves: clipping bias decreases with C, noise variance grows as C^2.
    bias = 2.7 * np.exp(-1.35 * c) + 0.08
    variance = 0.085 * c**2
    mse = bias + variance

    c_star = c[np.argmin(mse)]
    mse_star = mse.min()

    plt.rcParams.update(
        {
            "font.size": 14,
            "axes.titlesize": 18,
            "axes.labelsize": 15,
            "legend.fontsize": 12,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "font.family": "DejaVu Sans",
        }
    )

    fig, ax = plt.subplots(figsize=(10.8, 6.0))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#fbfcfe")

    ax.plot(c, bias, color="#f97316", linewidth=3, label="Clipping loss / Bias(C)")
    ax.plot(c, variance, color="#6366f1", linewidth=3, label="Noise variance")
    ax.plot(c, mse, color="#0f766e", linewidth=4, label="MSE(C)")

    ax.axvline(c_star, color="#0f766e", linestyle="--", linewidth=2)
    ax.scatter([c_star], [mse_star], s=90, color="#0f766e", zorder=5)
    ax.annotate(
        r"Chosen threshold $C^*$",
        xy=(c_star, mse_star),
        xytext=(c_star + 0.35, mse_star + 0.55),
        arrowprops=dict(arrowstyle="->", color="#0f766e", lw=2),
        color="#0f766e",
        fontsize=14,
        fontweight="bold",
    )

    ax.annotate(
        "Small C:\nlarge clipping loss",
        xy=(0.55, bias[np.searchsorted(c, 0.55)]),
        xytext=(0.45, 2.45),
        arrowprops=dict(arrowstyle="->", color="#f97316", lw=2),
        color="#9a3412",
        fontsize=13,
        ha="center",
    )

    ax.annotate(
        "Large C:\nlarge noise variance",
        xy=(4.0, variance[np.searchsorted(c, 4.0)]),
        xytext=(3.55, 2.35),
        arrowprops=dict(arrowstyle="->", color="#6366f1", lw=2),
        color="#3730a3",
        fontsize=13,
        ha="center",
    )

    ax.set_title("MSE Objective for Choosing the Clipping Threshold", pad=14)
    ax.set_xlabel("Clipping threshold C")
    ax.set_ylabel("Approximate cost")
    ax.set_xlim(0, 4.7)
    ax.set_ylim(0, 3.1)
    ax.grid(True, color="#94a3b8", alpha=0.25, linewidth=1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False)

    fig.tight_layout(rect=[0, 0.06, 1, 1])
    png_path = os.path.join(out_dir, "mse_u_curve_concept.png")
    svg_path = os.path.join(out_dir, "mse_u_curve_concept.svg")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    print(png_path)
    print(svg_path)


if __name__ == "__main__":
    main()
