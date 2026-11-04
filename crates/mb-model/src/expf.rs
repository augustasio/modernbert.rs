//! `exp` in f32, written so the compiler can vectorise it. Rung 9.
//!
//! Rungs 7 and 8 call f64 `exp` from libm once per element, in the attention
//! softmax and in `erf`. That is a function call the compiler cannot see
//! into, so those loops run one element at a time, and the rung 8 profile
//! puts 17% of the time there. PyTorch uses an f32 polynomial on NEON
//! instead (`notes/pytorch_cpu_research.md`). This is the same idea, written
//! as a plain scalar function with no branches and no libm calls, so LLVM
//! turns a loop over it into NEON code. No intrinsics and no `unsafe`.
//!
//! The method is the classic one (Cephes `expf`):
//!
//! 1. Split `x = n ln2 + r` with integer `n` and `|r| <= ln2 / 2`.
//! 2. `exp(r)` by a degree-5 polynomial, which is accurate to about 1 ulp on
//!    that short interval.
//! 3. `exp(x) = 2^n exp(r)`, where `2^n` is built directly in the exponent
//!    bits.

/// Below this, `exp(x)` is under the smallest normal f32, so the answer is 0.
///
/// **Trap:** the softmax relies on `exp(f32::MIN - max)` being exactly 0, so
/// a masked key adds nothing to the total (see `fused.rs`). Returning 0 here,
/// rather than a subnormal or garbage from an out-of-range exponent, keeps
/// that true.
const MIN_X: f32 = -87.336_54;
/// Above this, `2^n` would need the exponent 255, which is infinity.
const MAX_X: f32 = 88.376_26;

const LOG2_E: f32 = std::f32::consts::LOG2_E;
// ln 2 in two parts (Cody and Waite). LN2_HI has few enough bits that
// `n * LN2_HI` is exact for every `n` here, so `x - n * LN2_HI` loses nothing.
const LN2_HI: f32 = 0.693_359_4;
const LN2_LO: f32 = -2.121_944_4e-4;

// The Cephes `expf` coefficients, for exp(r) = 1 + r + r^2 P(r).
const P0: f32 = 1.987_569_1e-4;
const P1: f32 = 1.398_199_9e-3;
const P2: f32 = 8.333_452e-3;
const P3: f32 = 4.166_579_6e-2;
const P4: f32 = 1.666_666_5e-1;
// Cephes writes 5.0000001e-1, which rounds to exactly 0.5 in f32.
const P5: f32 = 0.5;

/// `e^x` in f32, within 2 ulp of the true value, and exactly 0 below
/// [`MIN_X`].
#[inline(always)]
pub fn exp_f32(x: f32) -> f32 {
    let clamped = x.clamp(MIN_X, MAX_X);
    let n = (clamped * LOG2_E).round();
    let r = clamped - n * LN2_HI - n * LN2_LO;

    let mut p = P0;
    p = p.mul_add(r, P1);
    p = p.mul_add(r, P2);
    p = p.mul_add(r, P3);
    p = p.mul_add(r, P4);
    p = p.mul_add(r, P5);
    let exp_r = (p * r).mul_add(r, r) + 1.0;

    // 2^n: n + 127 in the exponent field, zero mantissa. The clamp keeps n
    // in [-126, 127], so the field stays in [1, 254].
    let two_n = f32::from_bits(((n as i32 + 127) as u32) << 23);
    let y = exp_r * two_n;
    if x < MIN_X {
        0.0
    } else {
        y
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Distance in units in the last place between two positive f32 values.
    fn ulps(a: f32, b: f32) -> u32 {
        (a.to_bits() as i64 - b.to_bits() as i64).unsigned_abs() as u32
    }

    #[test]
    fn within_two_ulp_of_f64_exp_across_the_range() {
        // Every 61st f32 bit pattern between MIN_X and MAX_X, both signs:
        // about 35 million values.
        let mut worst = (0u32, 0.0f32);
        for bits in (0..=u32::MAX).step_by(61) {
            let x = f32::from_bits(bits);
            if !(MIN_X..=MAX_X).contains(&x) {
                continue;
            }
            let want = (x as f64).exp() as f32;
            let err = ulps(exp_f32(x), want);
            if err > worst.0 {
                worst = (err, x);
            }
        }
        assert!(worst.0 <= 2, "{} ulp at x = {}", worst.0, worst.1);
    }

    #[test]
    fn exact_at_zero_and_zero_below_the_range() {
        assert_eq!(exp_f32(0.0), 1.0);
        assert_eq!(exp_f32(-0.0), 1.0);
        assert_eq!(exp_f32(MIN_X - 0.01), 0.0);
        assert_eq!(exp_f32(f32::MIN), 0.0);
        assert_eq!(exp_f32(-1.0e30), 0.0);
    }

    #[test]
    fn a_masked_key_adds_exactly_nothing() {
        // What the softmax computes for a masked score: f32::MIN minus a
        // typical row maximum.
        assert_eq!(exp_f32(f32::MIN - 12.5), 0.0);
    }

    #[test]
    fn finite_at_the_top_of_the_range() {
        assert!(exp_f32(MAX_X).is_finite());
    }
}
