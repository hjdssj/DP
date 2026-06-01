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

    def get_noisy_counts(self, epsilon_hist, rng=None, clip_negative=True):
        """Return a DP-noisy copy of histogram counts.

        The histogram query has L1 sensitivity 1 because each sample
        contributes to exactly one bin. Adding independent Laplace noise with
        scale 1 / epsilon_hist gives epsilon_hist-DP for one epoch query.
        """
        if epsilon_hist <= 0:
            raise ValueError("epsilon_hist must be positive for DP histogram")
        if self.total_samples == 0:
            return np.zeros_like(self.counts, dtype=float)

        rng = rng if rng is not None else np.random
        noisy_counts = self.counts.astype(float) + rng.laplace(
            0.0, 1.0 / epsilon_hist, size=self.num_bins
        )
        if clip_negative:
            noisy_counts = np.maximum(noisy_counts, 0.0)
        return noisy_counts

    def get_noisy_clipped_ratio(self, epsilon_hist, rng=None):
        """Return a DP-noisy clipped ratio for ratio-based updates."""
        if epsilon_hist <= 0:
            raise ValueError("epsilon_hist must be positive for DP histogram")
        if self.total_samples == 0:
            return 0.0

        rng = rng if rng is not None else np.random
        noisy_count = self.clipped_count + rng.laplace(0.0, 1.0 / epsilon_hist)
        noisy_count = max(0.0, min(float(self.total_samples), float(noisy_count)))
        return noisy_count / self.total_samples

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

def compute_mse_from_counts(C, bin_centers, counts, sigma, d, n, total_samples=None):
    """Compute MSE(C) using histogram approximation.

    MSE(C) = bias(C) + variance(C)

    where:
        bias(C)     = (1/N) Σ_i max(||g_i|| - C, 0)²
        variance(C) = σ² C² d / n²

    The bias term is approximated by discrete summation over histogram bins:
        bias(C) ≈ (1/N) Σ_b count[b] · max(bin_center[b] - C, 0)²

    Args:
        C: Candidate clipping threshold
        bin_centers: Centers of histogram bins
        counts: Histogram counts, optionally DP-noisy
        sigma: DP noise multiplier
        d: Model parameter dimension (sum of all parameter sizes)
        n: Batch size
        total_samples: Optional denominator for the bias term. If omitted,
            the sum of counts is used, which is appropriate for noisy counts.

    Returns:
        dict with keys 'mse', 'bias', 'variance'
    """
    counts = np.maximum(np.asarray(counts, dtype=float), 0.0)
    bin_centers = np.asarray(bin_centers, dtype=float)
    N = float(total_samples) if total_samples is not None else float(np.sum(counts))
    if N <= 0:
        return {'mse': float('inf'), 'bias': 0.0, 'variance': 0.0}

    excess = np.maximum(bin_centers - C, 0.0)
    bias = float(np.sum(counts * excess ** 2) / N)

    variance = (sigma ** 2) * (C ** 2) * d / (n ** 2)

    return {'mse': bias + variance, 'bias': bias, 'variance': variance}


def compute_mse_from_histogram(C, histogram, sigma, d, n):
    """Compute MSE(C) using plaintext histogram counts."""
    return compute_mse_from_counts(
        C, histogram.bin_centers, histogram.counts, sigma, d, n,
        total_samples=histogram.total_samples,
    )


def estimate_clipped_ratio_from_counts(bin_centers, counts, C):
    """Approximate clipped ratio from histogram counts without a new query."""
    counts = np.maximum(np.asarray(counts, dtype=float), 0.0)
    total = float(np.sum(counts))
    if total <= 0:
        return 0.0
    clipped = float(np.sum(counts[np.asarray(bin_centers) >= C]))
    return max(0.0, min(1.0, clipped / total))


def find_optimal_c_mse(histogram, sigma, d, n, min_c=0.05, max_c=None,
                       num_coarse=200, num_fine=50, counts=None,
                       total_samples=None):
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
        counts: Optional counts array. Pass DP-noisy counts for private
            histogram-based MSE.
        total_samples: Optional denominator for the bias term.

    Returns:
        Optimal C value (float)
    """
    if max_c is None:
        max_c = histogram.bin_max

    if histogram.total_samples == 0 and counts is None:
        return min_c

    counts = histogram.counts if counts is None else counts
    total_samples = histogram.total_samples if total_samples is None else total_samples

    # Phase 1: coarse grid search
    candidates = np.linspace(min_c, max_c, num_coarse)
    best_c = min_c
    best_mse = float('inf')

    for c in candidates:
        result = compute_mse_from_counts(
            c, histogram.bin_centers, counts, sigma, d, n,
            total_samples=total_samples,
        )
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
        result = compute_mse_from_counts(
            c, histogram.bin_centers, counts, sigma, d, n,
            total_samples=total_samples,
        )
        if result['mse'] < best_mse:
            best_mse = result['mse']
            best_c = c

    return float(best_c)


def compute_mse_curve_from_counts(bin_centers, counts, sigma, d, n,
                                  min_c=0.05, max_c=10.0, num_points=300,
                                  total_samples=None):
    """Compute Bias(C), Variance(C), and MSE(C) over a candidate C grid."""
    candidates = np.linspace(min_c, max_c, num_points)
    bias_values = []
    var_values = []
    mse_values = []

    for c in candidates:
        result = compute_mse_from_counts(
            c, bin_centers, counts, sigma, d, n,
            total_samples=total_samples,
        )
        bias_values.append(result['bias'])
        var_values.append(result['variance'])
        mse_values.append(result['mse'])

    return {
        'candidates': candidates.tolist(),
        'bias': bias_values,
        'variance': var_values,
        'mse': mse_values,
    }


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
                 # DP histogram parameters
                 use_dp_histogram=False, epsilon_hist=1.0, rng=None,
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

        # DP histogram
        self.use_dp_histogram = use_dp_histogram
        self.epsilon_hist = epsilon_hist
        self.rng = rng if rng is not None else np.random
        self.epsilon_hist_spent = 0.0
        self.histogram_query_count = 0
        self.last_update_info = {}

        # tracking
        self.c_history = [initial_c]
        self.clipped_ratio_history = []
        self.mse_history = []
        self.bias_history = []
        self.var_history = []
        self.mse_curve_history = []

    def update(self, histogram):
        """Update C based on histogram statistics.

        For 'ratio' mode: uses clipped_ratio from histogram
        For 'mse' mode: finds C that minimizes MSE(C)

        Returns:
            (new_c, old_c) tuple
        """
        old_c = self.c
        self.last_update_info = {
            'use_dp_histogram': self.use_dp_histogram,
            'epsilon_hist': self.epsilon_hist if self.use_dp_histogram else 0.0,
        }

        if self.mode == 'ratio':
            if self.use_dp_histogram:
                clipped_ratio = histogram.get_noisy_clipped_ratio(
                    self.epsilon_hist, rng=self.rng
                )
                self._record_histogram_query()
                self.last_update_info['clipped_ratio_source'] = 'noisy'
            else:
                clipped_ratio = histogram.get_clipped_ratio()
                self.last_update_info['clipped_ratio_source'] = 'true'

            self.clipped_ratio_history.append(clipped_ratio)
            self.last_update_info['reported_clipped_ratio'] = clipped_ratio
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
        raw_counts = histogram.counts.astype(float).copy()
        raw_total_samples = int(histogram.total_samples)
        if self.use_dp_histogram:
            counts = histogram.get_noisy_counts(self.epsilon_hist, rng=self.rng)
            total_samples = None
            self._record_histogram_query()
            clipped_ratio = estimate_clipped_ratio_from_counts(
                histogram.bin_centers, counts, self.c
            )
            self.last_update_info['clipped_ratio_source'] = 'estimated_from_noisy_counts'
        else:
            counts = histogram.counts
            total_samples = histogram.total_samples
            clipped_ratio = histogram.get_clipped_ratio()
            self.last_update_info['clipped_ratio_source'] = 'true'

        self.clipped_ratio_history.append(clipped_ratio)
        self.last_update_info['reported_clipped_ratio'] = clipped_ratio

        optimal_c = find_optimal_c_mse(
            histogram, self.sigma, self.d, self.batch_size,
            min_c=self.min_c, max_c=self.max_c,
            counts=counts, total_samples=total_samples,
        )

        mse_curve = compute_mse_curve_from_counts(
            histogram.bin_centers, counts, self.sigma, self.d, self.batch_size,
            min_c=self.min_c, max_c=self.max_c, num_points=300,
            total_samples=total_samples,
        )
        mse_curve['optimal_c_raw'] = float(optimal_c)
        mse_curve['previous_c'] = float(self.c)
        mse_curve['histogram_total_samples'] = int(histogram.total_samples)
        mse_curve['histogram_query_is_dp'] = bool(self.use_dp_histogram)
        mse_curve['bin_edges'] = histogram.bin_edges.tolist()
        mse_curve['bin_centers'] = histogram.bin_centers.tolist()
        mse_curve['raw_counts'] = raw_counts.tolist()
        mse_curve['raw_total_samples'] = raw_total_samples
        mse_curve['used_counts'] = np.asarray(counts, dtype=float).tolist()
        if self.use_dp_histogram:
            mse_curve['noisy_counts'] = np.asarray(counts, dtype=float).tolist()
        self.mse_curve_history.append(mse_curve)

        mse_result = compute_mse_from_counts(
            optimal_c, histogram.bin_centers, counts, self.sigma, self.d,
            self.batch_size, total_samples=total_samples,
        )
        self.mse_history.append(mse_result['mse'])
        self.bias_history.append(mse_result['bias'])
        self.var_history.append(mse_result['variance'])

        new_c = self.smoothing * self.c + (1 - self.smoothing) * optimal_c
        return max(self.min_c, min(self.max_c, new_c))

    def get_c(self):
        return self.c

    def _record_histogram_query(self):
        self.histogram_query_count += 1
        self.epsilon_hist_spent += self.epsilon_hist
