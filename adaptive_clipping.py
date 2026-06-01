"""
Shared adaptive gradient clipping module for DP-SGD.

Contains:
  - GradientHistogram: tracks per-sample gradient norm distribution
  - compute_mse_from_histogram: MSE(C) = bias + variance via histogram
  - find_optimal_c_mse: grid search for MSE-minimizing C
  - AdaptiveClipController: dual-mode (ratio / mse) C update controller

Used by:
  - minst_adaptive_histogram.py
  - fashionmnist_resnet18_dp_baseline.py
  - (any future DP-SGD script with adaptive clipping)
"""

import numpy as np


class GradientHistogram:
    """Tracks TRUE (pre-clip) gradient norm distribution.

    grad_sample is read before optimizer.step(), so norms are unclipped.
    We compare against current_c to compute the real clipped ratio.
    """

    def __init__(self, bin_min=0.0, bin_max=2.0, num_bins=50):
        self.bin_min = bin_min
        self.bin_max = bin_max
        self.bin_edges = np.linspace(bin_min, bin_max, num_bins + 1)
        self.bin_centers = (self.bin_edges[:-1] + self.bin_edges[1:]) / 2
        self.num_bins = num_bins
        self.current_c = bin_max / 2
        self.reset()

    def set_bin_max(self, bin_max):
        self.bin_max = bin_max
        self.bin_edges = np.linspace(self.bin_min, bin_max, self.num_bins + 1)
        self.bin_centers = (self.bin_edges[:-1] + self.bin_edges[1:]) / 2

    def set_current_c(self, c):
        self.current_c = c

    def reset(self):
        self.counts = np.zeros(self.num_bins)
        self.total_samples = 0
        self.clipped_count = 0

    def add_batch(self, grad_norms):
        """Add TRUE per-sample gradient norms (before Opacus clipping)."""
        norms = grad_norms.cpu().detach().numpy()
        self.total_samples += len(norms)

        # Count samples that would be clipped by current C
        self.clipped_count += int(np.sum(norms >= self.current_c))

        # Bin all norms
        indices = np.clip(
            np.searchsorted(self.bin_edges[1:], norms), 0, self.num_bins - 1
        )
        for idx in indices:
            self.counts[idx] += 1

    def get_clipped_ratio(self):
        if self.total_samples == 0:
            return 0.0
        return self.clipped_count / self.total_samples

    def get_stats(self):
        if self.total_samples == 0:
            return {}
        mean = np.sum(self.bin_centers * self.counts) / self.total_samples
        return {
            'total_samples': self.total_samples,
            'clipped_ratio': self.get_clipped_ratio(),
            'mean': mean,
            'std': np.sqrt(
                np.sum(((self.bin_centers - mean) ** 2) * self.counts) / self.total_samples
            ),
        }


# ---------------------------------------------------------------------------
# MSE-based optimal C computation using histogram approximation
# ---------------------------------------------------------------------------

def compute_mse_from_histogram(C, histogram, sigma, d, n):
    """Compute MSE(C) using histogram approximation.

    MSE(C) = bias(C) + variance(C)

    where:
        bias(C)     = (1/N) Σ_i max(||g_i|| - C, 0)²
        variance(C) = σ² C² d / n²

    The bias term is approximated by discrete summation over histogram bins:
        bias(C) ≈ (1/N) Σ_b count[b] · max(bin_center[b] - C, 0)²

    Args:
        C: Candidate clipping threshold
        histogram: GradientHistogram with accumulated norm distribution
        sigma: DP noise multiplier
        d: Model parameter dimension (sum of all parameter sizes)
        n: Batch size

    Returns:
        dict with keys 'mse', 'bias', 'variance'
    """
    N = histogram.total_samples
    if N == 0:
        return {'mse': float('inf'), 'bias': 0.0, 'variance': 0.0}

    excess = np.maximum(histogram.bin_centers - C, 0.0)
    bias = float(np.sum(histogram.counts * excess ** 2) / N)

    variance = (sigma ** 2) * (C ** 2) * d / (n ** 2)

    return {'mse': bias + variance, 'bias': bias, 'variance': variance}


def find_optimal_c_mse(histogram, sigma, d, n, min_c=0.05, max_c=None,
                       num_coarse=200, num_fine=50):
    """Find C that minimizes MSE(C) via two-phase grid search.

    Phase 1: coarse search over [min_c, max_c] with num_coarse points
    Phase 2: fine search around the best candidate with num_fine points

    Args:
        histogram: GradientHistogram with accumulated norm distribution
        sigma: DP noise multiplier
        d: Model parameter dimension
        n: Batch size
        min_c: Lower bound for candidate C
        max_c: Upper bound for candidate C (default: histogram.bin_max)
        num_coarse: Number of candidates in coarse search
        num_fine: Number of candidates in fine search

    Returns:
        Optimal C value (float)
    """
    if max_c is None:
        max_c = histogram.bin_max

    if histogram.total_samples == 0:
        return min_c

    # Phase 1: coarse grid search
    candidates = np.linspace(min_c, max_c, num_coarse)
    best_c = min_c
    best_mse = float('inf')

    for c in candidates:
        result = compute_mse_from_histogram(c, histogram, sigma, d, n)
        if result['mse'] < best_mse:
            best_mse = result['mse']
            best_c = c

    # Phase 2: fine search around the best candidate
    fine_lo = max(min_c, best_c * 0.7)
    fine_hi = min(max_c, best_c * 1.3)
    if fine_hi <= fine_lo:
        return best_c

    fine_candidates = np.linspace(fine_lo, fine_hi, num_fine)
    for c in fine_candidates:
        result = compute_mse_from_histogram(c, histogram, sigma, d, n)
        if result['mse'] < best_mse:
            best_mse = result['mse']
            best_c = c

    return float(best_c)


# ---------------------------------------------------------------------------
# Adaptive clip controller
# ---------------------------------------------------------------------------

class AdaptiveClipController:
    """Adaptive clipping threshold controller supporting two modes.

    Modes:
    - 'ratio': Adjust C based on observed clipping ratio vs target
    - 'mse':   Adjust C by minimizing MSE(C) = bias + variance
    """

    def __init__(self, mode='ratio', initial_c=1.0,
                 # ratio mode parameters
                 target_ratio=0.2, tolerance=0.1, adjustment_speed=0.15,
                 # mse mode parameters
                 sigma=1.0, d=26010, batch_size=64,
                 # common parameters
                 min_c=0.1, max_c=10.0, smoothing=0.5):
        self.mode = mode
        self.c = initial_c
        self.min_c = min_c
        self.max_c = max_c
        self.smoothing = smoothing

        # ratio mode
        self.target_ratio = target_ratio
        self.tolerance = tolerance
        self.adjustment_speed = adjustment_speed

        # mse mode
        self.sigma = sigma
        self.d = d
        self.batch_size = batch_size

        # tracking
        self.c_history = [initial_c]
        self.clipped_ratio_history = []
        self.mse_history = []
        self.bias_history = []
        self.var_history = []

    def update(self, histogram):
        """Update C based on histogram statistics.

        For 'ratio' mode: uses clipped_ratio from histogram
        For 'mse' mode: finds C that minimizes MSE(C)

        Returns:
            (new_c, old_c) tuple
        """
        old_c = self.c
        clipped_ratio = histogram.get_clipped_ratio()
        self.clipped_ratio_history.append(clipped_ratio)

        if self.mode == 'ratio':
            new_c = self._update_ratio(clipped_ratio)
        elif self.mode == 'mse':
            new_c = self._update_mse(histogram)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        self.c = new_c
        self.c_history.append(new_c)
        return new_c, old_c

    def _update_ratio(self, clipped_ratio):
        """Ratio-based C adjustment."""
        if clipped_ratio > self.target_ratio + self.tolerance:
            excess = clipped_ratio - (self.target_ratio + self.tolerance)
            adjustment = 1 + self.adjustment_speed * (1 + excess * 2)
            raw_c = self.c * adjustment
        elif clipped_ratio < self.target_ratio - self.tolerance:
            deficit = (self.target_ratio - self.tolerance) - clipped_ratio
            adjustment = 1 - self.adjustment_speed * (1 + deficit * 2)
            raw_c = self.c * adjustment
        else:
            raw_c = self.c

        raw_c = max(self.min_c, min(self.max_c, raw_c))
        return self.smoothing * self.c + (1 - self.smoothing) * raw_c

    def _update_mse(self, histogram):
        """MSE-minimizing C adjustment."""
        optimal_c = find_optimal_c_mse(
            histogram, self.sigma, self.d, self.batch_size,
            min_c=self.min_c, max_c=self.max_c,
        )

        mse_result = compute_mse_from_histogram(
            optimal_c, histogram, self.sigma, self.d, self.batch_size
        )
        self.mse_history.append(mse_result['mse'])
        self.bias_history.append(mse_result['bias'])
        self.var_history.append(mse_result['variance'])

        new_c = self.smoothing * self.c + (1 - self.smoothing) * optimal_c
        return max(self.min_c, min(self.max_c, new_c))

    def get_c(self):
        return self.c
