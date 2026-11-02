//! The check made on every reply during a benchmarked run (hard rule 5).
//!
//! A variant that got fast by computing something else must fail loudly, so
//! the logits are checked in the run that is timed, not only in a separate
//! correctness run.

/// Looser than the 1e-3 the tier-2 tests use on short fixtures, on purpose.
/// Over all 7,600 prompts, correct fp32 rungs reach 4.7e-4 (ONNX Runtime) and,
/// on one example, 2.55e-3 (candle's `gemm` summation order). A gate that
/// fails a correct rung is useless, so the bound is 4x the worst seen. A rung
/// computing something else (bf16 matmul, a skipped layer) misses by far more,
/// and the argmax check stands beside it. See the PLAN.md changelog.
pub const MAX_LOGIT_ABS: f32 = 1e-2;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Check {
    pub argmax_agrees: bool,
    pub max_abs: f32,
}

impl Check {
    pub fn passes(&self) -> bool {
        self.argmax_agrees && self.max_abs <= MAX_LOGIT_ABS
    }
}

pub fn argmax(v: &[f32]) -> usize {
    let mut best = 0;
    for (i, &x) in v.iter().enumerate() {
        if x > v[best] {
            best = i;
        }
    }
    best
}

pub fn check(reference: &[f32], got: &[f32]) -> Check {
    assert_eq!(reference.len(), got.len(), "logit counts differ");
    let max_abs = reference
        .iter()
        .zip(got)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    Check {
        argmax_agrees: argmax(reference) == argmax(got),
        max_abs,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matching_logits_pass() {
        let c = check(&[1.0, 3.0, 2.0, 0.0], &[1.0, 3.0002, 2.0, 0.0]);
        assert!(c.passes());
    }

    #[test]
    fn a_different_prediction_fails_even_when_close() {
        let c = check(&[2.0, 2.0005, 0.0, 0.0], &[2.0005, 2.0, 0.0, 0.0]);
        assert!(c.max_abs <= MAX_LOGIT_ABS);
        assert!(!c.passes(), "the argmax changed, so this must fail");
    }

    #[test]
    fn a_large_difference_fails_even_with_the_same_prediction() {
        let c = check(&[1.0, 5.0, 0.0, 0.0], &[1.0, 5.1, 0.0, 0.0]);
        assert!(c.argmax_agrees);
        assert!(!c.passes());
    }
}
