//! Histogram-based data structures for O(n·p) CART splitting.
//!
//! Instead of sorting per feature per node (O(n·p·log n)), histogram CART:
//! 1. Quantizes features to 256 bins once (O(n·p·log n) amortized)
//! 2. Builds per-node histograms in O(n·p) per level
//! 3. Scans 256 bins per feature for best split (negligible)
//!
//! Net: O(n·p) per node instead of O(n·p·log n). 3-5x faster at 100K+ rows.

// ---------------------------------------------------------------------------
// Feature quantization
// ---------------------------------------------------------------------------

/// Maps continuous f64 values to discrete u8 bin indices (0..n_bins-1).
/// Uses quantile-based bin edges for roughly equal-frequency bins.
pub(crate) struct Quantizer {
    /// edges[i] is the upper bound of bin i. value <= edges[0] → bin 0,
    /// edges[i-1] < value <= edges[i] → bin i, value > edges[last] → last bin.
    edges: Vec<f64>,
    /// Actual number of bins (≤ MAX_BINS, fewer for low-cardinality features).
    pub(crate) n_bins: usize,
}

pub(crate) const MAX_BINS: usize = 256;

/// Bin index reserved for NaN values.
/// Bins 0..254 are for quantile edges; bin 255 always holds NaN samples.
pub(crate) const NAN_BIN: u8 = 255;

impl Quantizer {
    /// Build quantizer from a column of feature values.
    /// NaN values are filtered out before computing edges — they always get NAN_BIN (255).
    pub(crate) fn from_column(values: &[f64], max_bins: usize) -> Self {
        let max_bins = max_bins.min(MAX_BINS);
        let n = values.len();
        if n == 0 {
            return Self {
                edges: Vec::new(),
                n_bins: 1,
            };
        }

        // Filter NaN before sorting — NaN values always go to NAN_BIN, not into edges.
        let mut sorted: Vec<f64> = values.iter().copied().filter(|v| !v.is_nan()).collect();
        if sorted.is_empty() {
            // All NaN column: 1 bin (the NaN bin is separate, handled by bin_of)
            return Self {
                edges: Vec::new(),
                n_bins: 1,
            };
        }
        sorted.sort_unstable_by(|a, b| a.total_cmp(b));
        let n = sorted.len(); // re-bind to non-NaN count

        // Never add an edge at the max value — it creates an empty last bin.
        let max_val = sorted[n - 1];
        let edges = if n <= max_bins {
            // Exact mode: n ≤ max_bins so we can use every unique value as a
            // split candidate. This gives exact CART splits (same as sklearn)
            // for small datasets instead of approximate quantile positions.
            let mut edges = Vec::with_capacity(n);
            for &v in &sorted {
                if v >= max_val {
                    break;
                }
                if edges.last().map_or(true, |&last: &f64| last < v) {
                    edges.push(v);
                }
            }
            edges
        } else {
            // Quantile mode: large datasets — pick max_bins quantile edges,
            // skipping duplicates.
            let mut edges = Vec::with_capacity(max_bins);
            for i in 1..max_bins {
                let idx = (i * n / max_bins).min(n - 1);
                let edge = sorted[idx];
                if edge >= max_val {
                    continue;
                }
                if edges.last().map_or(true, |&last: &f64| last < edge) {
                    edges.push(edge);
                }
            }
            edges
        };

        let n_bins = edges.len() + 1;
        Self { edges, n_bins }
    }

    /// Map a value to its bin index. NaN values always return NAN_BIN (255).
    #[inline]
    pub(crate) fn bin_of(&self, value: f64) -> u8 {
        if value.is_nan() {
            return NAN_BIN;
        }
        // partition_point returns first index where edge >= value is false,
        // i.e., first edge that is NOT less than value.
        self.edges.partition_point(|&edge| edge < value) as u8
    }

    /// Reconstruct the threshold for splitting at bin `b`.
    /// left = bins 0..=b, right = bins (b+1)..
    /// Returns edges[b] — all values with bin_of(x) <= b satisfy x <= edges[b].
    #[inline]
    pub(crate) fn threshold_of(&self, bin: u8) -> f64 {
        let b = bin as usize;
        if b < self.edges.len() {
            self.edges[b]
        } else {
            *self.edges.last().unwrap_or(&0.0)
        }
    }
}

// ---------------------------------------------------------------------------
// Quantized feature matrix
// ---------------------------------------------------------------------------

/// Column-major matrix of u8 bin indices, one per (row, feature).
pub(crate) struct QuantizedMatrix {
    data: Vec<u8>,
    pub(crate) nrows: usize,
    pub(crate) ncols: usize,
    quantizers: Vec<Quantizer>,
}

impl QuantizedMatrix {
    /// Quantize from column-major f64 data (same layout as ColMajorMatrix).
    pub(crate) fn from_col_data(
        col_data: &[f64],
        nrows: usize,
        ncols: usize,
        max_bins: usize,
    ) -> Self {
        let mut data = Vec::with_capacity(nrows * ncols);
        let mut quantizers = Vec::with_capacity(ncols);

        for col in 0..ncols {
            let start = col * nrows;
            let col_slice = &col_data[start..start + nrows];
            let q = Quantizer::from_column(col_slice, max_bins);
            for &v in col_slice {
                data.push(q.bin_of(v));
            }
            quantizers.push(q);
        }

        Self {
            data,
            nrows,
            ncols,
            quantizers,
        }
    }

    /// Get bin index for (row, col) without bounds check.
    #[inline(always)]
    pub(crate) unsafe fn get_unchecked(&self, row: usize, col: usize) -> u8 {
        unsafe { *self.data.get_unchecked(col * self.nrows + row) }
    }

    /// Threshold for splitting feature `col` at bin `bin`.
    #[inline]
    pub(crate) fn threshold_for(&self, col: usize, bin: u8) -> f64 {
        self.quantizers[col].threshold_of(bin)
    }

    /// Number of bins for feature `col`.
    #[inline]
    pub(crate) fn n_bins_for(&self, col: usize) -> usize {
        self.quantizers[col].n_bins
    }
}

// ---------------------------------------------------------------------------
// GBT histogram (gradient/hessian per bin, for lossguide growth)
// ---------------------------------------------------------------------------

/// Single bin in a GBT histogram — AoS layout keeps g and h adjacent (16 bytes, one cache line
/// holds 4 bins). This halves cache-line writes during scatter vs SoA (g_bins + h_bins 2 KB apart).
#[derive(Clone, Copy, Default)]
pub(crate) struct GBTHistBin {
    pub(crate) g: f64,
    pub(crate) h: f64,
}

/// Per-feature gradient/hessian histogram for GBT lossguide growth.
///
/// Always MAX_BINS (256) slots. Slots 0..max_bin-1 are data bins;
/// slot 255 (NAN_BIN) accumulates samples with NaN for this feature.
#[derive(Clone)]
pub(crate) struct GBTHistogram {
    pub(crate) bins: [GBTHistBin; MAX_BINS],
}

impl GBTHistogram {
    #[inline]
    pub(crate) fn new() -> Self {
        Self { bins: [GBTHistBin::default(); MAX_BINS] }
    }
}

// ---------------------------------------------------------------------------
// Entropy helper
// ---------------------------------------------------------------------------

/// Shannon entropy from weighted class counts (base-e log).
/// Any log base gives equivalent split rankings, so base-e avoids the ln(2) constant.
fn entropy_from_counts(counts: &[f64], total_w: f64) -> f64 {
    if total_w < 1e-15 {
        return 0.0;
    }
    let mut h = 0.0_f64;
    for &c in counts {
        if c > 1e-15 {
            let p = c / total_w;
            h -= p * p.ln();
        }
    }
    h
}

// ---------------------------------------------------------------------------
// Classification histogram
// ---------------------------------------------------------------------------

/// Per-feature histogram for Gini-based classification splitting.
///
/// Stores weighted class counts and sample counts per bin.
/// Layout: `class_w[bin * n_classes + class]` = weighted count.
/// NaN samples are tracked separately (not part of the bin array).
pub(crate) struct ClfHistogram {
    class_w: Vec<f64>,
    bin_counts: Vec<usize>,
    n_bins: usize,
    n_classes: usize,
    /// Weighted class counts for NaN samples (length = n_classes).
    nan_class_w: Vec<f64>,
    /// Number of NaN samples.
    nan_count: usize,
}

impl ClfHistogram {
    pub(crate) fn new(n_bins: usize, n_classes: usize) -> Self {
        Self {
            class_w: vec![0.0; n_bins * n_classes],
            bin_counts: vec![0; n_bins],
            n_bins,
            n_classes,
            nan_class_w: vec![0.0; n_classes],
            nan_count: 0,
        }
    }

    /// Reset all bins to zero for reuse across features.
    pub(crate) fn clear(&mut self, n_bins: usize) {
        let needed = n_bins * self.n_classes;
        if self.class_w.len() < needed {
            self.class_w.resize(needed, 0.0);
        }
        self.class_w[..needed].fill(0.0);
        if self.bin_counts.len() < n_bins {
            self.bin_counts.resize(n_bins, 0);
        }
        self.bin_counts[..n_bins].fill(0);
        self.n_bins = n_bins;
        self.nan_class_w.fill(0.0);
        self.nan_count = 0;
    }

    /// Accumulate one sample. NAN_BIN samples go to separate NaN accumulators.
    #[inline]
    pub(crate) fn add(&mut self, bin: u8, class: usize, weight: f64) {
        if bin == NAN_BIN {
            self.nan_class_w[class] += weight;
            self.nan_count += 1;
            return;
        }
        let b = bin as usize;
        self.class_w[b * self.n_classes + class] += weight;
        self.bin_counts[b] += 1;
    }

    /// Compute right child histogram = parent - left child (histogram subtraction trick).
    /// Caller must ensure `left` contains a proper subset of `parent`'s samples.
    /// Avoids scanning right-child samples entirely — O(bins × k) instead of O(n_right × p).
    pub(crate) fn subtract(parent: &Self, left: &Self) -> Self {
        debug_assert_eq!(parent.n_bins, left.n_bins);
        debug_assert_eq!(parent.n_classes, left.n_classes);
        let nb = parent.n_bins;
        let k = parent.n_classes;
        let mut out = Self::new(nb, k);
        for i in 0..(nb * k) {
            // max(0.0) guards against fp rounding giving tiny negatives
            out.class_w[i] = (parent.class_w[i] - left.class_w[i]).max(0.0);
        }
        for b in 0..nb {
            out.bin_counts[b] = parent.bin_counts[b].saturating_sub(left.bin_counts[b]);
        }
        // NaN accumulators
        for c in 0..k {
            out.nan_class_w[c] = (parent.nan_class_w[c] - left.nan_class_w[c]).max(0.0);
        }
        out.nan_count = parent.nan_count.saturating_sub(left.nan_count);
        out
    }

    /// Find best Gini split by prefix-scanning bins.
    /// Returns `(best_bin, proxy_value, nan_goes_left)` where proxy = sq_l/wl + sq_r/wr
    /// (higher = better). `best_bin` means left = bins 0..=best_bin.
    /// NaN samples are tried in both directions; the direction with better proxy wins.
    pub(crate) fn best_gini_split(
        &self,
        min_samples_leaf: usize,
    ) -> Option<(u8, f64, bool)> {
        let k = self.n_classes;
        let nb = self.n_bins;

        // Total counts per class (excluding NaN)
        let mut total_counts = vec![0.0_f64; k];
        for bin in 0..nb {
            let base = bin * k;
            for c in 0..k {
                total_counts[c] += self.class_w[base + c];
            }
        }
        let total_w: f64 = total_counts.iter().sum();

        // NaN statistics
        let nan_w: f64 = self.nan_class_w.iter().sum();
        let has_nan = self.nan_count > 0;

        // Prefix scan (over non-NaN bins only)
        let mut left_counts = vec![0.0_f64; k];
        let mut n_left_w = 0.0_f64;
        let mut count_left = 0_usize;
        let total_count: usize = self.bin_counts[..nb].iter().sum();
        let mut count_right = total_count;

        let mut best_proxy = f64::NEG_INFINITY;
        let mut best_bin = 0_u8;
        let mut best_nan_left = true;
        let mut found = false;

        for bin in 0..(nb - 1) {
            let base = bin * k;
            for c in 0..k {
                left_counts[c] += self.class_w[base + c];
            }
            let bin_w: f64 = (0..k).map(|c| self.class_w[base + c]).sum();
            n_left_w += bin_w;
            count_left += self.bin_counts[bin];
            count_right -= self.bin_counts[bin];

            if self.bin_counts[bin] == 0 {
                continue;
            }
            if count_left < min_samples_leaf || count_right < min_samples_leaf {
                // NaN can potentially rescue a failing min_samples_leaf check
                // (handled below), but if both sides fail even with NaN, skip.
                if !has_nan {
                    continue;
                }
            }
            let n_right_w = total_w - n_left_w;
            if n_left_w < 1e-15 && n_right_w < 1e-15 {
                continue;
            }

            if !has_nan {
                // No NaN: original fast path
                if count_left < min_samples_leaf || count_right < min_samples_leaf {
                    continue;
                }
                if n_left_w < 1e-15 || n_right_w < 1e-15 {
                    continue;
                }
                let sq_l: f64 = left_counts.iter().map(|&c| c * c).sum();
                let sq_r: f64 = left_counts
                    .iter()
                    .enumerate()
                    .map(|(c, &lc)| {
                        let rc = total_counts[c] - lc;
                        rc * rc
                    })
                    .sum();
                let proxy = sq_l / n_left_w + sq_r / n_right_w;
                if proxy > best_proxy {
                    best_proxy = proxy;
                    best_bin = bin as u8;
                    best_nan_left = true;
                    found = true;
                }
            } else {
                // Try NaN → left
                let nl_w = n_left_w + nan_w;
                let nr_w = n_right_w;
                let cl = count_left + self.nan_count;
                let cr = count_right;
                if cl >= min_samples_leaf && cr >= min_samples_leaf && nl_w > 1e-15 && nr_w > 1e-15 {
                    let sq_l: f64 = (0..k).map(|c| {
                        let v = left_counts[c] + self.nan_class_w[c];
                        v * v
                    }).sum();
                    let sq_r: f64 = (0..k).map(|c| {
                        let rc = total_counts[c] - left_counts[c];
                        rc * rc
                    }).sum();
                    let proxy = sq_l / nl_w + sq_r / nr_w;
                    if proxy > best_proxy {
                        best_proxy = proxy;
                        best_bin = bin as u8;
                        best_nan_left = true;
                        found = true;
                    }
                }

                // Try NaN → right
                let nl_w2 = n_left_w;
                let nr_w2 = n_right_w + nan_w;
                let cl2 = count_left;
                let cr2 = count_right + self.nan_count;
                if cl2 >= min_samples_leaf && cr2 >= min_samples_leaf && nl_w2 > 1e-15 && nr_w2 > 1e-15 {
                    let sq_l: f64 = left_counts.iter().map(|&c| c * c).sum();
                    let sq_r: f64 = (0..k).map(|c| {
                        let rc = total_counts[c] - left_counts[c] + self.nan_class_w[c];
                        rc * rc
                    }).sum();
                    let proxy = sq_l / nl_w2 + sq_r / nr_w2;
                    if proxy > best_proxy {
                        best_proxy = proxy;
                        best_bin = bin as u8;
                        best_nan_left = false;
                        found = true;
                    }
                }
            }
        }

        if !found {
            return None;
        }

        // Check for actual improvement: compare against no-split proxy.
        // Include NaN in total for the comparison.
        let mut full_counts = total_counts.clone();
        let full_w = total_w + nan_w;
        if has_nan {
            for c in 0..k {
                full_counts[c] += self.nan_class_w[c];
            }
        }
        let sq_total: f64 = full_counts.iter().map(|&c| c * c).sum();
        let no_split_proxy = if full_w > 1e-15 { sq_total / full_w } else { 0.0 };
        if best_proxy <= no_split_proxy + 1e-10 {
            return None;
        }

        Some((best_bin, best_proxy, best_nan_left))
    }

    /// Find best entropy (information gain) split by prefix-scanning bins.
    /// Returns `(best_bin, child_weighted_entropy, nan_goes_left)` where **lower = better**.
    /// NaN samples are tried in both directions; the direction with lower cost wins.
    pub(crate) fn best_entropy_split(
        &self,
        min_samples_leaf: usize,
    ) -> Option<(u8, f64, bool)> {
        let k = self.n_classes;
        let nb = self.n_bins;

        let mut total_counts = vec![0.0_f64; k];
        for bin in 0..nb {
            let base = bin * k;
            for c in 0..k {
                total_counts[c] += self.class_w[base + c];
            }
        }
        let total_w: f64 = total_counts.iter().sum();

        // NaN statistics
        let nan_w: f64 = self.nan_class_w.iter().sum();
        let has_nan = self.nan_count > 0;

        // Parent entropy cost (including NaN for fair comparison)
        let full_w = total_w + nan_w;
        let parent_cost = if has_nan {
            let full_counts: Vec<f64> = (0..k).map(|c| total_counts[c] + self.nan_class_w[c]).collect();
            full_w * entropy_from_counts(&full_counts, full_w)
        } else {
            total_w * entropy_from_counts(&total_counts, total_w)
        };

        let mut left_counts = vec![0.0_f64; k];
        let mut n_left_w = 0.0_f64;
        let mut count_left = 0_usize;
        let total_count: usize = self.bin_counts[..nb].iter().sum();
        let mut count_right = total_count;

        let mut best_cost = f64::INFINITY;
        let mut best_bin = 0_u8;
        let mut best_nan_left = true;
        let mut found = false;

        for bin in 0..(nb - 1) {
            let base = bin * k;
            for c in 0..k {
                left_counts[c] += self.class_w[base + c];
            }
            let bin_w: f64 = (0..k).map(|c| self.class_w[base + c]).sum();
            n_left_w += bin_w;
            count_left += self.bin_counts[bin];
            count_right -= self.bin_counts[bin];

            if self.bin_counts[bin] == 0 {
                continue;
            }
            let n_right_w = total_w - n_left_w;

            if !has_nan {
                if count_left < min_samples_leaf || count_right < min_samples_leaf {
                    continue;
                }
                if n_left_w < 1e-15 || n_right_w < 1e-15 {
                    continue;
                }
                let right_counts: Vec<f64> = (0..k).map(|c| total_counts[c] - left_counts[c]).collect();
                let cost = n_left_w * entropy_from_counts(&left_counts, n_left_w)
                    + n_right_w * entropy_from_counts(&right_counts, n_right_w);
                if cost < best_cost {
                    best_cost = cost;
                    best_bin = bin as u8;
                    best_nan_left = true;
                    found = true;
                }
            } else {
                // Try NaN → left
                let nl_w = n_left_w + nan_w;
                let nr_w = n_right_w;
                let cl = count_left + self.nan_count;
                let cr = count_right;
                if cl >= min_samples_leaf && cr >= min_samples_leaf && nl_w > 1e-15 && nr_w > 1e-15 {
                    let lc: Vec<f64> = (0..k).map(|c| left_counts[c] + self.nan_class_w[c]).collect();
                    let rc: Vec<f64> = (0..k).map(|c| total_counts[c] - left_counts[c]).collect();
                    let cost = nl_w * entropy_from_counts(&lc, nl_w)
                        + nr_w * entropy_from_counts(&rc, nr_w);
                    if cost < best_cost {
                        best_cost = cost;
                        best_bin = bin as u8;
                        best_nan_left = true;
                        found = true;
                    }
                }

                // Try NaN → right
                let nl_w2 = n_left_w;
                let nr_w2 = n_right_w + nan_w;
                let cl2 = count_left;
                let cr2 = count_right + self.nan_count;
                if cl2 >= min_samples_leaf && cr2 >= min_samples_leaf && nl_w2 > 1e-15 && nr_w2 > 1e-15 {
                    let rc: Vec<f64> = (0..k).map(|c| total_counts[c] - left_counts[c] + self.nan_class_w[c]).collect();
                    let cost = nl_w2 * entropy_from_counts(&left_counts, nl_w2)
                        + nr_w2 * entropy_from_counts(&rc, nr_w2);
                    if cost < best_cost {
                        best_cost = cost;
                        best_bin = bin as u8;
                        best_nan_left = false;
                        found = true;
                    }
                }
            }
        }

        if !found {
            return None;
        }

        // No-improvement guard: no entropy reduction → no useful split
        if best_cost >= parent_cost - 1e-10 {
            return None;
        }

        Some((best_bin, best_cost, best_nan_left))
    }

}

// ---------------------------------------------------------------------------
// Regression histogram
// ---------------------------------------------------------------------------

/// Per-feature histogram for MSE-based regression splitting.
///
/// Stores (sum_w, sum_wy, sum_wy2) and sample count per bin.
/// NaN samples are tracked separately (not part of the bin array).
pub(crate) struct RegHistogram {
    sum_w: Vec<f64>,
    sum_wy: Vec<f64>,
    sum_wy2: Vec<f64>,
    bin_counts: Vec<usize>,
    n_bins: usize,
    /// NaN accumulator: sum of weights for NaN samples.
    nan_sw: f64,
    /// NaN accumulator: sum of weight*y for NaN samples.
    nan_swy: f64,
    /// NaN accumulator: sum of weight*y*y for NaN samples.
    nan_swy2: f64,
    /// Number of NaN samples.
    nan_count: usize,
}

impl RegHistogram {
    pub(crate) fn new(n_bins: usize) -> Self {
        Self {
            sum_w: vec![0.0; n_bins],
            sum_wy: vec![0.0; n_bins],
            sum_wy2: vec![0.0; n_bins],
            bin_counts: vec![0; n_bins],
            n_bins,
            nan_sw: 0.0,
            nan_swy: 0.0,
            nan_swy2: 0.0,
            nan_count: 0,
        }
    }

    /// Reset all bins to zero for reuse across features.
    pub(crate) fn clear(&mut self, n_bins: usize) {
        for v in [&mut self.sum_w, &mut self.sum_wy, &mut self.sum_wy2] {
            if v.len() < n_bins {
                v.resize(n_bins, 0.0);
            }
            v[..n_bins].fill(0.0);
        }
        if self.bin_counts.len() < n_bins {
            self.bin_counts.resize(n_bins, 0);
        }
        self.bin_counts[..n_bins].fill(0);
        self.n_bins = n_bins;
        self.nan_sw = 0.0;
        self.nan_swy = 0.0;
        self.nan_swy2 = 0.0;
        self.nan_count = 0;
    }

    /// Compute right child histogram = parent - left child (histogram subtraction trick).
    /// O(bins) instead of O(n_right × p) for building the larger child's histograms.
    pub(crate) fn subtract(parent: &Self, left: &Self) -> Self {
        debug_assert_eq!(parent.n_bins, left.n_bins);
        let nb = parent.n_bins;
        let mut out = Self::new(nb);
        for b in 0..nb {
            out.sum_w[b] = (parent.sum_w[b] - left.sum_w[b]).max(0.0);
            out.sum_wy[b] = parent.sum_wy[b] - left.sum_wy[b];
            out.sum_wy2[b] = (parent.sum_wy2[b] - left.sum_wy2[b]).max(0.0);
            out.bin_counts[b] = parent.bin_counts[b].saturating_sub(left.bin_counts[b]);
        }
        // NaN accumulators
        out.nan_sw = (parent.nan_sw - left.nan_sw).max(0.0);
        out.nan_swy = parent.nan_swy - left.nan_swy;
        out.nan_swy2 = (parent.nan_swy2 - left.nan_swy2).max(0.0);
        out.nan_count = parent.nan_count.saturating_sub(left.nan_count);
        out
    }

    /// Accumulate one sample. NAN_BIN samples go to separate NaN accumulators.
    #[inline]
    pub(crate) fn add(&mut self, bin: u8, y: f64, weight: f64) {
        if bin == NAN_BIN {
            self.nan_sw += weight;
            self.nan_swy += weight * y;
            self.nan_swy2 += weight * y * y;
            self.nan_count += 1;
            return;
        }
        let b = bin as usize;
        self.sum_w[b] += weight;
        self.sum_wy[b] += weight * y;
        self.sum_wy2[b] += weight * y * y;
        self.bin_counts[b] += 1;
    }

    /// Find best MSE split by prefix-scanning bins.
    /// Returns `(best_bin, child_iw, nan_goes_left)` where lower child_iw = better.
    /// NaN samples are tried in both directions; the direction with lower cost wins.
    ///
    /// `split_lambda`: L2 regularization on node weight sums for GBT Newton-gain splits.
    /// When > 0, the gain formula becomes G²/(H+λ) instead of G²/H (XGBoost-style).
    /// Set to 0.0 for standard regression trees (no regularization in split selection).
    pub(crate) fn best_mse_split(
        &self,
        min_samples_leaf: usize,
        split_lambda: f64,
    ) -> Option<(u8, f64, bool)> {
        let nb = self.n_bins;

        let total_sw: f64 = self.sum_w[..nb].iter().sum();
        let total_swy: f64 = self.sum_wy[..nb].iter().sum();
        let total_swy2: f64 = self.sum_wy2[..nb].iter().sum();
        let total_count: usize = self.bin_counts[..nb].iter().sum();

        let has_nan = self.nan_count > 0;

        let mut l_sw = 0.0_f64;
        let mut l_swy = 0.0_f64;
        let mut l_swy2 = 0.0_f64;
        let mut count_left = 0_usize;
        let mut count_right = total_count;

        let mut best_child_iw = f64::INFINITY;
        let mut best_bin = 0_u8;
        let mut best_nan_left = true;
        let mut found = false;

        for bin in 0..(nb - 1) {
            l_sw += self.sum_w[bin];
            l_swy += self.sum_wy[bin];
            l_swy2 += self.sum_wy2[bin];
            count_left += self.bin_counts[bin];
            count_right -= self.bin_counts[bin];

            if self.bin_counts[bin] == 0 {
                continue;
            }

            let r_sw = total_sw - l_sw;
            let r_swy = total_swy - l_swy;
            let r_swy2 = total_swy2 - l_swy2;

            // When split_lambda > 0 (GBT Newton-gain), the gain formula becomes
            // G²/(H+λ) instead of G²/H. This adds per-node L2 regularization
            // to split selection, preventing deep trees from overfitting.
            let lam = split_lambda;

            if !has_nan {
                if count_left < min_samples_leaf || count_right < min_samples_leaf {
                    continue;
                }
                if l_sw < 1e-15 || r_sw < 1e-15 {
                    continue;
                }
                let ld = l_sw + lam;
                let rd = r_sw + lam;
                let mse_l = (l_swy2 / ld - (l_swy / ld).powi(2)).max(0.0);
                let mse_r = (r_swy2 / rd - (r_swy / rd).powi(2)).max(0.0);
                let child_iw = ld * mse_l + rd * mse_r;
                if child_iw < best_child_iw {
                    best_child_iw = child_iw;
                    best_bin = bin as u8;
                    best_nan_left = true;
                    found = true;
                }
            } else {
                // Try NaN → left
                let nl_sw = l_sw + self.nan_sw;
                let nl_swy = l_swy + self.nan_swy;
                let nl_swy2 = l_swy2 + self.nan_swy2;
                let cl = count_left + self.nan_count;
                let cr = count_right;
                if cl >= min_samples_leaf && cr >= min_samples_leaf && nl_sw > 1e-15 && r_sw > 1e-15 {
                    let nld = nl_sw + lam;
                    let rd = r_sw + lam;
                    let mse_l = (nl_swy2 / nld - (nl_swy / nld).powi(2)).max(0.0);
                    let mse_r = (r_swy2 / rd - (r_swy / rd).powi(2)).max(0.0);
                    let child_iw = nld * mse_l + rd * mse_r;
                    if child_iw < best_child_iw {
                        best_child_iw = child_iw;
                        best_bin = bin as u8;
                        best_nan_left = true;
                        found = true;
                    }
                }

                // Try NaN → right
                let nr_sw = r_sw + self.nan_sw;
                let nr_swy = r_swy + self.nan_swy;
                let nr_swy2 = r_swy2 + self.nan_swy2;
                let cl2 = count_left;
                let cr2 = count_right + self.nan_count;
                if cl2 >= min_samples_leaf && cr2 >= min_samples_leaf && l_sw > 1e-15 && nr_sw > 1e-15 {
                    let ld = l_sw + lam;
                    let nrd = nr_sw + lam;
                    let mse_l = (l_swy2 / ld - (l_swy / ld).powi(2)).max(0.0);
                    let mse_r = (nr_swy2 / nrd - (nr_swy / nrd).powi(2)).max(0.0);
                    let child_iw = ld * mse_l + nrd * mse_r;
                    if child_iw < best_child_iw {
                        best_child_iw = child_iw;
                        best_bin = bin as u8;
                        best_nan_left = false;
                        found = true;
                    }
                }
            }
        }

        if found {
            Some((best_bin, best_child_iw, best_nan_left))
        } else {
            None
        }
    }

    /// Find best Poisson deviance split by prefix-scanning bins.
    /// Returns `(best_bin, child_weighted_deviance, nan_goes_left)` where **lower = better**.
    /// Assumes target y >= 0 (caller must validate before fitting).
    /// NaN samples are tried in both directions; the direction with lower cost wins.
    pub(crate) fn best_poisson_split(
        &self,
        min_samples_leaf: usize,
    ) -> Option<(u8, f64, bool)> {
        let nb = self.n_bins;
        let total_sw: f64 = self.sum_w[..nb].iter().sum();
        let total_swy: f64 = self.sum_wy[..nb].iter().sum();
        let total_count: usize = self.bin_counts[..nb].iter().sum();

        let has_nan = self.nan_count > 0;
        let full_swy = total_swy + self.nan_swy;
        let full_sw = total_sw + self.nan_sw;

        // All y == 0: Poisson log is undefined — no meaningful split
        if full_swy < 1e-15 {
            return None;
        }

        // Parent deviance proxy (including NaN for fair comparison)
        let parent_cost = -full_swy * (full_swy / full_sw).ln();

        let mut l_sw = 0.0_f64;
        let mut l_swy = 0.0_f64;
        let mut count_left = 0_usize;
        let mut count_right = total_count;

        let mut best_cost = f64::INFINITY;
        let mut best_bin = 0_u8;
        let mut best_nan_left = true;
        let mut found = false;

        for bin in 0..(nb - 1) {
            l_sw += self.sum_w[bin];
            l_swy += self.sum_wy[bin];
            count_left += self.bin_counts[bin];
            count_right -= self.bin_counts[bin];

            if self.bin_counts[bin] == 0 {
                continue;
            }
            let r_sw = total_sw - l_sw;
            let r_swy = total_swy - l_swy;

            if !has_nan {
                if count_left < min_samples_leaf || count_right < min_samples_leaf {
                    continue;
                }
                if l_sw < 1e-15 || r_sw < 1e-15 {
                    continue;
                }
                if l_swy < 1e-15 || r_swy < 1e-15 {
                    continue;
                }
                let cost = -l_swy * (l_swy / l_sw).ln() - r_swy * (r_swy / r_sw).ln();
                if cost < best_cost {
                    best_cost = cost;
                    best_bin = bin as u8;
                    best_nan_left = true;
                    found = true;
                }
            } else {
                // Try NaN → left
                let nl_sw = l_sw + self.nan_sw;
                let nl_swy = l_swy + self.nan_swy;
                let cl = count_left + self.nan_count;
                let cr = count_right;
                if cl >= min_samples_leaf && cr >= min_samples_leaf
                    && nl_sw > 1e-15 && r_sw > 1e-15
                    && nl_swy > 1e-15 && r_swy > 1e-15
                {
                    let cost = -nl_swy * (nl_swy / nl_sw).ln() - r_swy * (r_swy / r_sw).ln();
                    if cost < best_cost {
                        best_cost = cost;
                        best_bin = bin as u8;
                        best_nan_left = true;
                        found = true;
                    }
                }

                // Try NaN → right
                let nr_sw = r_sw + self.nan_sw;
                let nr_swy = r_swy + self.nan_swy;
                let cl2 = count_left;
                let cr2 = count_right + self.nan_count;
                if cl2 >= min_samples_leaf && cr2 >= min_samples_leaf
                    && l_sw > 1e-15 && nr_sw > 1e-15
                    && l_swy > 1e-15 && nr_swy > 1e-15
                {
                    let cost = -l_swy * (l_swy / l_sw).ln() - nr_swy * (nr_swy / nr_sw).ln();
                    if cost < best_cost {
                        best_cost = cost;
                        best_bin = bin as u8;
                        best_nan_left = false;
                        found = true;
                    }
                }
            }
        }

        if !found {
            return None;
        }

        // No-improvement guard
        if best_cost >= parent_cost - 1e-10 {
            return None;
        }

        Some((best_bin, best_cost, best_nan_left))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantizer_basic() {
        // 10 values → 4 bins
        let values: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let q = Quantizer::from_column(&values, 4);
        assert!(q.n_bins <= 4);
        assert!(q.n_bins >= 2);
        // First value should be in bin 0
        assert_eq!(q.bin_of(0.0), 0);
        // Last value should be in last bin
        assert_eq!(q.bin_of(9.0), (q.n_bins - 1) as u8);
    }

    #[test]
    fn test_quantizer_constant_feature() {
        let values = vec![5.0; 100];
        let q = Quantizer::from_column(&values, 256);
        assert_eq!(q.n_bins, 1);
        assert_eq!(q.bin_of(5.0), 0);
    }

    #[test]
    fn test_quantizer_two_values() {
        let mut values = vec![0.0; 50];
        values.extend(vec![1.0; 50]);
        let q = Quantizer::from_column(&values, 256);
        assert_eq!(q.n_bins, 2);
        assert_eq!(q.bin_of(0.0), 0);
        assert_eq!(q.bin_of(1.0), 1);
    }

    #[test]
    fn test_quantizer_monotonic() {
        // Bins must be monotonically non-decreasing with value
        let values: Vec<f64> = (0..1000).map(|i| i as f64 * 0.1).collect();
        let q = Quantizer::from_column(&values, 256);
        let mut prev_bin = 0_u8;
        for v in &values {
            let bin = q.bin_of(*v);
            assert!(bin >= prev_bin, "non-monotonic: {v} → bin {bin} < {prev_bin}");
            prev_bin = bin;
        }
    }

    #[test]
    fn test_quantizer_threshold_roundtrip() {
        let values: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let q = Quantizer::from_column(&values, 10);
        // For each edge, values <= threshold should get bin <= b
        for b in 0..q.n_bins.saturating_sub(1) {
            let thresh = q.threshold_of(b as u8);
            let bin = q.bin_of(thresh);
            assert!(
                bin <= b as u8,
                "threshold {thresh} for bin {b} mapped to bin {bin}"
            );
        }
    }

    #[test]
    fn test_quantized_matrix() {
        // 4 rows, 2 columns, column-major
        let col_data = vec![
            1.0, 2.0, 3.0, 4.0, // col 0
            10.0, 20.0, 30.0, 40.0, // col 1
        ];
        let qm = QuantizedMatrix::from_col_data(&col_data, 4, 2, 256);
        assert_eq!(qm.nrows, 4);
        assert_eq!(qm.ncols, 2);
        // Bins should be monotonically ordered within each column
        for col in 0..2 {
            let mut prev = 0_u8;
            for row in 0..4 {
                let bin = unsafe { qm.get_unchecked(row, col) };
                assert!(bin >= prev);
                prev = bin;
            }
        }
    }

    #[test]
    fn test_clf_histogram_gini_split() {
        // Binary classification: 100 samples, feature perfectly separates classes
        // Bins 0-4: all class 0, Bins 5-9: all class 1
        let n_bins = 10;
        let n_classes = 2;
        let mut h = ClfHistogram::new(n_bins, n_classes);
        for bin in 0..5_u8 {
            for _ in 0..10 {
                h.add(bin, 0, 1.0);
            }
        }
        for bin in 5..10_u8 {
            for _ in 0..10 {
                h.add(bin, 1, 1.0);
            }
        }

        let result = h.best_gini_split(1);
        assert!(result.is_some());
        let (best_bin, _proxy, _nan_left) = result.unwrap();
        // Best split should be at bin 4 (left=0..=4, right=5..=9)
        assert_eq!(best_bin, 4, "expected split at bin 4, got {best_bin}");
    }

    #[test]
    fn test_clf_histogram_no_split() {
        // All same class → no valid split
        let n_bins = 10;
        let n_classes = 2;
        let mut h = ClfHistogram::new(n_bins, n_classes);
        for bin in 0..10_u8 {
            h.add(bin, 0, 1.0);
        }
        assert!(h.best_gini_split(1).is_none());
    }

    #[test]
    fn test_reg_histogram_mse_split() {
        // Regression: bins 0-4 have y≈0, bins 5-9 have y≈10
        let n_bins = 10;
        let mut h = RegHistogram::new(n_bins);
        for bin in 0..5_u8 {
            for _ in 0..10 {
                h.add(bin, 0.0, 1.0);
            }
        }
        for bin in 5..10_u8 {
            for _ in 0..10 {
                h.add(bin, 10.0, 1.0);
            }
        }

        let result = h.best_mse_split(1, 0.0);
        assert!(result.is_some());
        let (best_bin, child_iw, _nan_left) = result.unwrap();
        assert_eq!(best_bin, 4, "expected split at bin 4, got {best_bin}");
        // Perfect split → child_iw should be 0 (each side is constant)
        assert!(
            child_iw < 1e-10,
            "expected near-zero child_iw, got {child_iw}"
        );
    }

    #[test]
    fn test_reg_histogram_no_split() {
        // All in one bin → no split possible
        let mut h = RegHistogram::new(1);
        h.add(0, 5.0, 1.0);
        h.add(0, 6.0, 1.0);
        assert!(h.best_mse_split(1, 0.0).is_none());
    }

    #[test]
    fn test_histogram_clear_and_reuse() {
        let mut h = ClfHistogram::new(10, 2);
        h.add(0, 0, 1.0);
        h.add(5, 1, 1.0);
        h.clear(10);
        // After clear, should find no valid split (all zeros)
        let mut has_data = false;
        for bin in 0..10 {
            for c in 0..2 {
                if h.class_w[bin * 2 + c] != 0.0 {
                    has_data = true;
                }
            }
        }
        assert!(!has_data, "histogram not cleared properly");
    }

    #[test]
    fn test_histogram_min_samples_leaf() {
        // 2 samples total, min_samples_leaf=2 → no valid split
        let mut h = ClfHistogram::new(10, 2);
        h.add(0, 0, 1.0);
        h.add(9, 1, 1.0);
        assert!(h.best_gini_split(2).is_none());
    }

    #[test]
    fn test_quantizer_empty() {
        let q = Quantizer::from_column(&[], 256);
        assert_eq!(q.n_bins, 1);
    }

    #[test]
    fn test_clf_histogram_subtract() {
        // Build a parent histogram with 3 bins, 2 classes.
        // Partition: bins 0-1 → left, bin 2 → right.
        let n_bins = 3;
        let n_classes = 2;
        let mut parent = ClfHistogram::new(n_bins, n_classes);
        // bins 0-1: class 0
        parent.add(0, 0, 1.0);
        parent.add(1, 0, 1.0);
        // bin 2: class 1
        parent.add(2, 1, 1.0);

        let mut left = ClfHistogram::new(n_bins, n_classes);
        left.add(0, 0, 1.0);
        left.add(1, 0, 1.0);

        let right = ClfHistogram::subtract(&parent, &left);

        // right should have only bin 2 with class 1
        assert_eq!(right.bin_counts[0], 0);
        assert_eq!(right.bin_counts[1], 0);
        assert_eq!(right.bin_counts[2], 1);
        // class_w: bin 2, class 1 = 1.0; all others ≈ 0
        let w = right.class_w[2 * n_classes + 1]; // bin 2, class 1
        assert!((w - 1.0).abs() < 1e-10, "expected 1.0, got {w}");
        let w0 = right.class_w[0]; // bin 0, class 0
        assert!(w0.abs() < 1e-10, "expected 0.0, got {w0}");
    }

    #[test]
    fn test_reg_histogram_subtract() {
        // Parent: bins 0-1 have regression values, bin 2 also.
        // Partition: bins 0-1 → left, bin 2 → right.
        let n_bins = 3;
        let mut parent = RegHistogram::new(n_bins);
        parent.add(0, 2.0, 1.0);
        parent.add(1, 4.0, 1.0);
        parent.add(2, 10.0, 1.0);

        let mut left = RegHistogram::new(n_bins);
        left.add(0, 2.0, 1.0);
        left.add(1, 4.0, 1.0);

        let right = RegHistogram::subtract(&parent, &left);

        // right should only have bin 2: sum_w=1, sum_wy=10, sum_wy2=100
        assert!((right.sum_w[2] - 1.0).abs() < 1e-10);
        assert!((right.sum_wy[2] - 10.0).abs() < 1e-10);
        assert!((right.sum_wy2[2] - 100.0).abs() < 1e-10);
        assert_eq!(right.bin_counts[2], 1);
        assert!(right.sum_w[0].abs() < 1e-10);
        assert!(right.sum_w[1].abs() < 1e-10);
    }
}
