// shuffle.rs — Deterministic, platform-independent shuffle.
//
// PCG-XSH-RR 64/32 (O'Neill 2014). Chosen for:
//   - Identical output on all platforms (no OS/stdlib dependency)
//   - Fast (one multiply + add per step)
//   - Well-studied statistical quality
//   - Trivially auditable (< 30 lines)
//
// This is the SINGLE source of truth for random permutations across
// Python, R, and Julia. Same seed → same permutation, always.

/// PCG-XSH-RR 64→32 state.
struct Pcg32 {
    state: u64,
    inc: u64,
}

impl Pcg32 {
    /// Create a new PCG from a seed. The increment is fixed (no stream selection
    /// needed — we only need reproducibility, not multiple independent streams).
    fn new(seed: u64) -> Self {
        // Increment must be odd. Using a large prime-derived constant.
        let inc: u64 = 0x14057B7EF767814F;
        let mut rng = Pcg32 { state: 0, inc };
        // Warm up: advance state twice (PCG seeding protocol)
        rng.state = rng.state.wrapping_mul(6364136223846793005).wrapping_add(inc);
        rng.state = rng.state.wrapping_add(seed);
        rng.state = rng.state.wrapping_mul(6364136223846793005).wrapping_add(inc);
        rng
    }

    /// Generate one u32 output using XSH-RR output function.
    fn next_u32(&mut self) -> u32 {
        let old_state = self.state;
        // Advance internal state
        self.state = old_state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(self.inc);
        // XSH-RR output permutation
        let xorshifted = (((old_state >> 18) ^ old_state) >> 27) as u32;
        let rot = (old_state >> 59) as u32;
        xorshifted.rotate_right(rot)
    }

    /// Uniform u32 in [0, bound) using rejection sampling (no modulo bias).
    fn bounded(&mut self, bound: u32) -> u32 {
        let threshold = bound.wrapping_neg() % bound; // = (2^32 - bound) % bound
        loop {
            let r = self.next_u32();
            if r >= threshold {
                return r % bound;
            }
        }
    }
}

/// Fisher-Yates shuffle of [0, 1, ..., n-1] using PCG-XSH-RR seeded with `seed`.
///
/// Returns a `Vec<usize>` permutation. Identical output on all platforms
/// for the same (n, seed) pair. This is the function all language bindings call.
///
/// # Panics
/// Panics if `n > u32::MAX as usize` (4 billion elements — not a real concern).
pub fn shuffle(n: usize, seed: u64) -> Vec<usize> {
    assert!(n <= u32::MAX as usize, "shuffle: n too large (max 4 billion)");
    let mut rng = Pcg32::new(seed);
    let mut perm: Vec<usize> = (0..n).collect();
    // Fisher-Yates (Knuth) shuffle — O(n), uniform, in-place
    for i in (1..n).rev() {
        let j = rng.bounded((i + 1) as u32) as usize;
        perm.swap(i, j);
    }
    perm
}

/// Round to nearest, half to even (banker's rounding).
/// Matches Python's `round()`, R's `round()`, and Julia's `round(Int, x)`.
/// Uses `f64::round_ties_even()` (Rust 1.77+, IEEE 754 roundTiesToEven) —
/// single hardware instruction, correct by construction.
fn round_half_to_even(x: f64) -> usize {
    x.round_ties_even() as usize
}

/// Partition sizes using the canonical formula: round(n * ratio).
/// Uses banker's rounding (half-to-even) for cross-language parity with
/// Python, R, and Julia. Remainder absorbed by the last partition.
///
/// Returns (n_train, n_valid, n_test).
pub fn partition_sizes(n: usize, ratio: [f64; 3]) -> (usize, usize, usize) {
    let n_train = round_half_to_even(n as f64 * ratio[0]);
    let n_valid = round_half_to_even(n as f64 * ratio[1]);
    let n_test = n - n_train - n_valid;
    (n_train, n_valid, n_test)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shuffle_deterministic() {
        let a = shuffle(100, 42);
        let b = shuffle(100, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn test_shuffle_different_seeds() {
        let a = shuffle(100, 1);
        let b = shuffle(100, 2);
        assert_ne!(a, b);
    }

    #[test]
    fn test_shuffle_is_permutation() {
        let perm = shuffle(100, 42);
        assert_eq!(perm.len(), 100);
        let mut sorted = perm.clone();
        sorted.sort();
        assert_eq!(sorted, (0..100).collect::<Vec<_>>());
    }

    #[test]
    fn test_shuffle_empty() {
        let perm = shuffle(0, 42);
        assert!(perm.is_empty());
    }

    #[test]
    fn test_shuffle_single() {
        let perm = shuffle(1, 42);
        assert_eq!(perm, vec![0]);
    }

    #[test]
    fn test_partition_sizes_clean() {
        let (t, v, te) = partition_sizes(100, [0.6, 0.2, 0.2]);
        assert_eq!((t, v, te), (60, 20, 20));
    }

    #[test]
    fn test_partition_sizes_remainder() {
        let (t, v, te) = partition_sizes(101, [0.6, 0.2, 0.2]);
        assert_eq!(t + v + te, 101);
        assert_eq!(t, 61); // round(101 * 0.6) = round(60.6) = 61
        assert_eq!(v, 20); // round(101 * 0.2) = round(20.2) = 20
        assert_eq!(te, 20);
    }

    #[test]
    fn test_partition_sizes_503() {
        let (t, v, te) = partition_sizes(503, [0.6, 0.2, 0.2]);
        assert_eq!(t + v + te, 503);
        assert_eq!(t, 302); // round(503 * 0.6) = round(301.8) = 302
        assert_eq!(v, 101); // round(503 * 0.2) = round(100.6) = 101
        assert_eq!(te, 100);
    }

    /// Golden test — this vector must NEVER change across versions.
    /// If it does, cross-language reproducibility breaks.
    #[test]
    fn test_shuffle_golden_10() {
        let perm = shuffle(10, 42);
        // Pin this exact output. Any change = regression.
        assert_eq!(perm, vec![4, 5, 9, 0, 1, 7, 8, 3, 2, 6]);
    }

    /// Banker's rounding: half-to-even must match Python/R/Julia.
    /// Rust's f64::round() would give 3 for 2.5, but banker's gives 2.
    #[test]
    fn test_partition_sizes_bankers_rounding() {
        // n=5, ratio=(0.5, 0.3, 0.2): 5*0.5=2.5 → banker's rounds to 2 (even)
        let (t, v, te) = partition_sizes(5, [0.5, 0.3, 0.2]);
        assert_eq!(t, 2, "2.5 should round to 2 (banker's), not 3");
        assert_eq!(v, 2, "1.5 should round to 2 (banker's)");
        assert_eq!(te, 1);

        // n=10, ratio=(0.25, 0.25, 0.5): 10*0.25=2.5 → 2
        let (t, v, te) = partition_sizes(10, [0.25, 0.25, 0.5]);
        assert_eq!(t, 2);
        assert_eq!(v, 2);
        assert_eq!(te, 6);

        // n=15, ratio=(0.5, 0.3, 0.2): 15*0.5=7.5 → 8 (round to even)
        let (t, v, te) = partition_sizes(15, [0.5, 0.3, 0.2]);
        assert_eq!(t, 8, "7.5 should round to 8 (banker's)");
        assert_eq!(t + v + te, 15);
    }
}
