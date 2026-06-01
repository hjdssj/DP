import argparse
import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description="Plot same-epoch MSE(C) candidate curve from saved experiment results."
    )
    parser.add_argument("result", help="Path to .pt result file with mse_curve_history")
    parser.add_argument("--epoch", type=int, default=-1,
                        help="1-based epoch index to plot; default: last recorded epoch")
    parser.add_argument("--out-dir", default="ppt_assets")
    parser.add_argument("--prefix", default=None)
    parser.add_argument("--plot-histogram", action="store_true",
                        help="Also plot raw/noisy gradient-norm histograms for the selected epoch")
    parser.add_argument("--hist-quantile", type=float, default=None,
                        help="Trim histogram right tail by raw-count quantile, e.g. 0.99")
    parser.add_argument("--hist-max", type=float, default=None,
                        help="Trim histogram bins above this gradient-norm value")
    args = parser.parse_args()

    import torch

    data = torch.load(args.result, map_location="cpu", weights_only=False)
    curves = data.get("mse_curve_history")
    if not curves:
        raise SystemExit(
            "This result file does not contain mse_curve_history. "
            "Run an MSE experiment after updating adaptive_clipping.py."
        )

    epoch = len(curves) if args.epoch == -1 else args.epoch
    if epoch < 1 or epoch > len(curves):
        raise SystemExit(f"epoch must be in [1, {len(curves)}], got {epoch}")

    curve = curves[epoch - 1]
    c = np.asarray(curve["candidates"], dtype=float)
    bias = np.asarray(curve["bias"], dtype=float)
    var = np.asarray(curve["variance"], dtype=float)
    mse = np.asarray(curve["mse"], dtype=float)
    c_star = float(curve.get("optimal_c_raw", c[np.argmin(mse)]))
    mse_star = float(np.interp(c_star, c, mse))

    os.makedirs(args.out_dir, exist_ok=True)
    base = args.prefix or os.path.splitext(os.path.basename(args.result))[0]
    base = f"{base}_epoch{epoch:03d}_mse_curve"

    plt.rcParams.update(
        {
            "font.size": 13,
            "axes.titlesize": 18,
            "axes.labelsize": 14,
            "legend.fontsize": 12,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "font.family": "DejaVu Sans",
        }
    )

    fig, ax = plt.subplots(figsize=(7.4, 7.4))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#fbfcfe")
    if hasattr(ax, "set_box_aspect"):
        ax.set_box_aspect(1)

    ax.plot(c, bias, color="#f97316", linewidth=3, label="Bias(C)")
    ax.plot(c, var, color="#6366f1", linewidth=3, label="Variance(C)")
    ax.plot(c, mse, color="#0f766e", linewidth=4, label="MSE(C)")
    ax.axvline(c_star, color="#0f766e", linestyle="--", linewidth=2)
    ax.scatter([c_star], [mse_star], s=90, color="#0f766e", zorder=5)
    ax.annotate(
        rf"selected raw $C^*={c_star:.3g}$",
        xy=(c_star, mse_star),
        xytext=(c_star + 0.05 * (c.max() - c.min()), mse_star + 0.12 * (mse.max() - mse.min())),
        arrowprops=dict(arrowstyle="->", color="#0f766e", lw=2),
        color="#0f766e",
        fontsize=13,
        fontweight="bold",
    )

    ax.set_title(f"Candidate Threshold Objective in Epoch {epoch}")
    ax.set_xlabel("Candidate clipping threshold C")
    ax.set_ylabel("Objective value")
    ax.grid(True, color="#94a3b8", alpha=0.25, linewidth=1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False)

    subtitle_parts = []
    if "args" in data:
        run_args = data["args"]
        if "sigma" in run_args:
            subtitle_parts.append(f"sigma={run_args['sigma']}")
        if "batch_size" in run_args:
            subtitle_parts.append(f"batch={run_args['batch_size']}")
    if curve.get("histogram_query_is_dp"):
        subtitle_parts.append("DP noisy histogram")
    if subtitle_parts:
        fig.suptitle(", ".join(subtitle_parts), y=1.02, fontsize=13, color="#475569")

    fig.tight_layout(rect=[0, 0.06, 1, 1])
    png_path = os.path.join(args.out_dir, f"{base}.png")
    svg_path = os.path.join(args.out_dir, f"{base}.svg")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    print(png_path)
    print(svg_path)

    if args.plot_histogram:
        plot_histograms(
            curve,
            args.out_dir,
            base,
            hist_quantile=args.hist_quantile,
            hist_max=args.hist_max,
        )


def plot_histograms(curve, out_dir, base, hist_quantile=None, hist_max=None):
    required = ["bin_edges", "raw_counts"]
    missing = [key for key in required if key not in curve]
    if missing:
        raise SystemExit(
            "This result file does not contain saved histogram counts. "
            "Re-run the training script after updating adaptive_clipping.py. "
            f"Missing keys: {', '.join(missing)}"
        )

    bin_edges = np.asarray(curve["bin_edges"], dtype=float)
    raw_counts = np.asarray(curve["raw_counts"], dtype=float)
    noisy_counts = np.asarray(
        curve.get("noisy_counts", curve.get("used_counts", raw_counts)),
        dtype=float,
    )
    bin_edges, raw_counts, noisy_counts, trim_note = trim_histogram_tail(
        bin_edges,
        raw_counts,
        noisy_counts,
        hist_quantile=hist_quantile,
        hist_max=hist_max,
    )
    c_star = float(curve.get("optimal_c_raw", np.nan))
    previous_c = float(curve.get("previous_c", np.nan))

    plot_single_histogram(
        bin_edges,
        raw_counts,
        title=with_trim_note("Raw Gradient-Norm Histogram", trim_note),
        color="#F97316",
        out_dir=out_dir,
        stem=f"{base}_raw_histogram",
        label="Raw counts",
    )
    plot_single_histogram(
        bin_edges,
        noisy_counts,
        title=with_trim_note("DP Noisy Gradient-Norm Histogram", trim_note),
        color="#1F6B45",
        out_dir=out_dir,
        stem=f"{base}_noisy_histogram",
        label="Noisy counts",
    )
    plot_comparison_histogram(
        bin_edges,
        raw_counts,
        noisy_counts,
        c_star=c_star,
        previous_c=previous_c,
        out_dir=out_dir,
        stem=f"{base}_histogram_with_c",
        trim_note=trim_note,
    )


def trim_histogram_tail(bin_edges, raw_counts, noisy_counts, hist_quantile=None, hist_max=None):
    if hist_quantile is not None and not (0 < hist_quantile <= 1):
        raise SystemExit("--hist-quantile must be in (0, 1], for example 0.99")

    keep_until = len(raw_counts)
    trim_note_parts = []

    if hist_quantile is not None and hist_quantile < 1:
        total = float(np.sum(np.maximum(raw_counts, 0.0)))
        if total > 0:
            cumulative = np.cumsum(np.maximum(raw_counts, 0.0)) / total
            keep_until = min(keep_until, int(np.searchsorted(cumulative, hist_quantile)) + 1)
            trim_note_parts.append(f"trimmed at raw {hist_quantile:.1%} quantile")

    if hist_max is not None:
        if hist_max <= bin_edges[0]:
            raise SystemExit("--hist-max must be larger than the first histogram edge")
        keep_by_max = int(np.searchsorted(bin_edges[1:], hist_max, side="right"))
        keep_until = min(keep_until, max(1, keep_by_max))
        trim_note_parts.append(f"x <= {hist_max:g}")

    keep_until = max(1, min(keep_until, len(raw_counts)))
    return (
        bin_edges[: keep_until + 1],
        raw_counts[:keep_until],
        noisy_counts[:keep_until],
        ", ".join(trim_note_parts),
    )


def with_trim_note(title, trim_note):
    return f"{title} ({trim_note})" if trim_note else title


def setup_histogram_axis(ax):
    ax.set_xlabel("Gradient norm")
    ax.set_ylabel("Count")
    ax.grid(True, axis="y", color="#94a3b8", alpha=0.25, linewidth=1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_single_histogram(bin_edges, counts, title, color, out_dir, stem, label):
    widths = np.diff(bin_edges)
    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#fbfcfe")
    ax.bar(
        bin_edges[:-1],
        counts,
        width=widths,
        align="edge",
        color=color,
        edgecolor="white",
        linewidth=0.5,
        alpha=0.88,
        label=label,
    )
    ax.set_title(title)
    setup_histogram_axis(ax)
    ax.legend(frameon=False)
    fig.tight_layout()
    save_histogram_outputs(fig, out_dir, stem)
    plt.close(fig)


def plot_comparison_histogram(
    bin_edges, raw_counts, noisy_counts, c_star, previous_c, out_dir, stem, trim_note=""
):
    widths = np.diff(bin_edges)
    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#fbfcfe")
    ax.bar(
        bin_edges[:-1],
        raw_counts,
        width=widths,
        align="edge",
        color="#F97316",
        edgecolor="white",
        linewidth=0.5,
        alpha=0.42,
        label="Raw counts",
    )
    ax.bar(
        bin_edges[:-1],
        noisy_counts,
        width=widths,
        align="edge",
        color="#1F6B45",
        edgecolor="white",
        linewidth=0.5,
        alpha=0.72,
        label="DP noisy counts",
    )
    if np.isfinite(c_star):
        ax.axvline(c_star, color="#0F172A", linestyle="--", linewidth=2.2,
                   label=rf"selected $C^*={c_star:.3g}$")
    if np.isfinite(previous_c):
        ax.axvline(previous_c, color="#64748B", linestyle=":", linewidth=2.0,
                   label=rf"previous $C={previous_c:.3g}$")
    ax.set_title(with_trim_note("Gradient-Norm Histogram with Clipping Threshold", trim_note))
    setup_histogram_axis(ax)
    ax.legend(frameon=False)
    fig.tight_layout()
    save_histogram_outputs(fig, out_dir, stem)
    plt.close(fig)


def save_histogram_outputs(fig, out_dir, stem):
    png_path = os.path.join(out_dir, f"{stem}.png")
    svg_path = os.path.join(out_dir, f"{stem}.svg")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    print(png_path)
    print(svg_path)


if __name__ == "__main__":
    main()
