//! Rung 12: the arithmetic outside the matmuls in f32, as PyTorch does it.
//!
//! Up to rung 11, three things are still computed in f64: the error function
//! in GELU, the LayerNorm mean and variance, and the softmax total. f64 was a
//! safe first choice, but PyTorch does all three in f32 vectors
//! (`notes/pytorch_cpu_research.md`, sections 2 and 3), and f64 halves the
//! lanes of every vector. Rung 12 moves all three to f32. That is one
//! variable, the precision of the non-matmul arithmetic.
//!
//! **The trap is the sum.** The residual stream holds values near 28,000, and
//! a plain f32 sum of 1,024 of them, one add after another, loses the low
//! bits of every small term once the running total is large. So the sums here
//! keep 16 running totals, one per lane, each over every 16th value, and add
//! the 16 together pairwise at the end. Each total sees a 16th of the terms,
//! and the vector loop is what the compiler wants anyway. PyTorch's
//! `RowwiseMoments` does the same kind of cascade, with chunks of 16.

use crate::expf::exp_f32;

/// How many running totals [`lane_sum`] keeps.
const LANES: usize = 16;

/// The sum of `f(v)` over `values`, with 16 running totals, added pairwise at
/// the end. See the module comment for why not one total.
#[inline(always)]
pub fn lane_sum(values: &[f32], f: impl Fn(f32) -> f32) -> f32 {
    let mut acc = [0.0f32; LANES];
    let chunks = values.chunks_exact(LANES);
    let rest = chunks.remainder();
    for chunk in chunks {
        for i in 0..LANES {
            acc[i] += f(chunk[i]);
        }
    }
    for (i, &v) in rest.iter().enumerate() {
        acc[i] += f(v);
    }
    // Pairwise: 16 -> 8 -> 4 -> 2 -> 1.
    let mut width = LANES / 2;
    while width > 0 {
        for i in 0..width {
            acc[i] += acc[i + width];
        }
        width /= 2;
    }
    acc[0]
}

/// Rung 25: the largest `f(v)` over `values`, or `start` if that is larger,
/// with 16 running maxima, one per lane.
///
/// A single running maximum, `values.iter().fold(start, f32::max)`, is a
/// chain: each step waits for the one before, so the loop cannot vectorise.
/// Sixteen maxima are sixteen independent chains, one vector wide. Unlike a
/// sum, a maximum is exact in any order, so the answer is the fold's, bit for
/// bit (no value here is NaN).
#[inline(always)]
pub fn lane_max(values: &[f32], start: f32, f: impl Fn(f32) -> f32) -> f32 {
    let mut acc = [start; LANES];
    let chunks = values.chunks_exact(LANES);
    let rest = chunks.remainder();
    for chunk in chunks {
        for i in 0..LANES {
            acc[i] = acc[i].max(f(chunk[i]));
        }
    }
    for (i, &v) in rest.iter().enumerate() {
        acc[i] = acc[i].max(f(v));
    }
    acc.iter().copied().fold(start, f32::max)
}

/// The error function in f32: Abramowitz and Stegun 7.1.26, the same formula
/// and coefficients as the f64 `ops::erf`, and as PyTorch's NEON
/// `Vectorized<float>::erf`.
///
/// Branch-free, so a loop over it vectorises: the sign is copied back on at
/// the end instead of chosen by an `if`.
#[inline(always)]
pub fn erf_f32(x: f32) -> f32 {
    const A1: f32 = 0.254_829_6;
    const A2: f32 = -0.284_496_74;
    const A3: f32 = 1.421_413_8;
    const A4: f32 = -1.453_152;
    const A5: f32 = 1.061_405_4;
    const P: f32 = 0.327_591_1;
    let a = x.abs();
    let t = 1.0 / (1.0 + P * a);
    let y = 1.0 - (((((A5 * t + A4) * t) + A3) * t + A2) * t + A1) * t * exp_f32(-a * a);
    y.copysign(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_lane_sum_adds_everything_once() {
        for n in [0usize, 1, 15, 16, 17, 1024, 1031] {
            let values: Vec<f32> = (0..n).map(|i| i as f32).collect();
            let want = (n * n.saturating_sub(1) / 2) as f32;
            assert_eq!(lane_sum(&values, |v| v), want, "{n} values");
        }
        assert_eq!(lane_sum(&[1.0, 2.0, 3.0], |v| v * v), 14.0);
    }

    #[test]
    fn the_lane_max_equals_the_fold() {
        for n in [0usize, 1, 15, 16, 17, 160, 1031] {
            // Mixed signs, the largest magnitude negative and late.
            let values: Vec<f32> = (0..n)
                .map(|i| ((i * 7919) % 101) as f32 * if i % 3 == 0 { -1.5 } else { 1.0 })
                .collect();
            let plain = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            assert_eq!(lane_max(&values, f32::NEG_INFINITY, |v| v), plain, "{n}");
            let abs = values.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            assert_eq!(lane_max(&values, 0.0, f32::abs), abs, "{n} abs");
        }
    }

    /// The case that sinks a plain f32 sum: one large value, then many small
    /// ones. The lane sum must stay near the f64 answer.
    #[test]
    fn the_lane_sum_keeps_small_terms_next_to_a_large_one() {
        let mut values = vec![0.3f32; 1024];
        values[0] = 28_000.0;
        let exact: f64 = values.iter().map(|&v| v as f64).sum();
        let lanes = lane_sum(&values, |v| v) as f64;
        let mut plain = 0.0f32;
        for &v in &values {
            plain += v;
        }
        assert!(
            (lanes - exact).abs() < (plain as f64 - exact).abs(),
            "lanes {lanes}, plain {plain}, exact {exact}"
        );
        // Not exact: one f32 step at 28,000 is 0.002, and the lane that holds
        // the large value still adds 63 small terms onto it. The tier-2 tests
        // judge whether that matters.
        assert!(
            (lanes - exact).abs() / exact < 1e-5,
            "lanes {lanes}, exact {exact}"
        );
    }

    #[test]
    fn erf_f32_is_close_to_the_f64_formula() {
        let mut worst = 0.0f64;
        for i in -6000..=6000 {
            let x = i as f32 / 1000.0;
            let a = x.abs() as f64;
            let t = 1.0 / (1.0 + 0.3275911 * a);
            let y = 1.0
                - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t
                    + 0.254829592)
                    * t
                    * (-a * a).exp();
            let want = y.copysign(x as f64);
            worst = worst.max((erf_f32(x) as f64 - want).abs());
        }
        assert!(worst < 1e-6, "worst gap to the f64 formula {worst:e}");
        assert_eq!(erf_f32(0.0), 0.0);
        assert!(erf_f32(-0.5) < 0.0);
    }
}
