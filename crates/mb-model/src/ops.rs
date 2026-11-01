//! The operations every component is built from.
//!
//! Three of them, and each has one detail that a reimplementation gets wrong
//! by default. The comments say which.

use crate::tensor::Matrix;

/// LayerNorm over the last dimension, one row at a time.
///
/// **It is LayerNorm, not RMSNorm.** The mean is subtracted. The config's own
/// docstring in the reference implementation mentions "rms normalization",
/// and because `norm_bias` is false there is exactly one tensor per norm, so
/// the state dict shape does not tell the two apart either. Getting this
/// wrong gives plausible-looking activations with the wrong values.
///
/// `weight` is the per-feature scale. There is no bias: `norm_bias` is false
/// for this model, so no norm in the checkpoint has one.
pub fn layer_norm(input: &Matrix, weight: &[f32], eps: f32) -> Matrix {
    assert_eq!(
        weight.len(),
        input.cols,
        "the norm weight has {} values for {} features",
        weight.len(),
        input.cols
    );

    let mut out = Matrix::zeros(input.rows, input.cols);
    let n = input.cols as f32;

    for row in 0..input.rows {
        let x = input.row(row);

        // Accumulate in f64. The sums run over 1024 values, and f32
        // accumulation there costs real precision against PyTorch, which
        // computes the statistics in f32 but from a different summation
        // order. f64 here keeps us inside the 1e-4 tolerance.
        let mut sum = 0.0f64;
        for &v in x {
            sum += v as f64;
        }
        let mean = (sum / n as f64) as f32;

        let mut variance = 0.0f64;
        for &v in x {
            let d = (v - mean) as f64;
            variance += d * d;
        }
        // The biased variance, dividing by n rather than n - 1, which is what
        // LayerNorm uses.
        let variance = (variance / n as f64) as f32;

        let scale = 1.0 / (variance + eps).sqrt();
        let y = out.row_mut(row);
        for i in 0..x.len() {
            y[i] = (x[i] - mean) * scale * weight[i];
        }
    }
    out
}

/// `input @ weight.T`, which is what a `torch.nn.Linear` with no bias does.
///
/// safetensors stores a linear layer's weight as `[out_features,
/// in_features]`, so the weight's rows are the output features and the
/// multiply is a dot product of an input row against a weight row. That makes
/// both operands row-major in their inner loop, which is why there is no
/// transpose anywhere.
///
/// This is the naive version, and it is rung 6 of the ladder on purpose.
/// Rung 7 replaces it with `cblas_sgemm` through Accelerate, and the gap
/// between the two is the measurement the whole project is built around.
pub fn linear(input: &Matrix, weight: &Matrix) -> Matrix {
    assert_eq!(
        input.cols, weight.cols,
        "cannot multiply a {}x{} input by a {}x{} weight",
        input.rows, input.cols, weight.rows, weight.cols
    );

    let mut out = Matrix::zeros(input.rows, weight.rows);
    for row in 0..input.rows {
        let x = input.row(row);
        let y = out.row_mut(row);
        for (feature, value) in y.iter_mut().enumerate() {
            let w = weight.row(feature);
            let mut sum = 0.0f32;
            for i in 0..x.len() {
                sum += x[i] * w[i];
            }
            *value = sum;
        }
    }
    out
}

/// Add `bias` to every row, in place.
pub fn add_bias(matrix: &mut Matrix, bias: &[f32]) {
    assert_eq!(
        bias.len(),
        matrix.cols,
        "the bias has {} values for {} features",
        bias.len(),
        matrix.cols
    );
    for row in 0..matrix.rows {
        let y = matrix.row_mut(row);
        for i in 0..bias.len() {
            y[i] += bias[i];
        }
    }
}

/// GELU, the **exact** form, using the error function.
///
/// `config.json` says `hidden_activation: "gelu"`, which the reference
/// implementation maps to `torch.nn.functional.gelu` with no `approximate`
/// argument, so it is `x * 0.5 * (1 + erf(x / sqrt(2)))` and not the tanh
/// approximation. The two differ by up to about 0.0003, which is three times
/// the whole fp32 tolerance, and it is applied in the feed-forward of all 28
/// layers and again in the MLM head.
///
/// candle's `gelu()` is the tanh approximation and its `gelu_erf()` is this
/// one, so rungs 4 and 5 have the same trap waiting.
pub fn gelu_erf(x: f32) -> f32 {
    x * 0.5 * (1.0 + erf(x as f64 / std::f64::consts::SQRT_2) as f32)
}

/// Apply [`gelu_erf`] to every value, in place.
pub fn gelu_erf_in_place(values: &mut [f32]) {
    for v in values.iter_mut() {
        *v = gelu_erf(*v);
    }
}

/// The error function.
///
/// Rust's standard library has no `erf`, so here is one. This is the rational
/// approximation from Abramowitz and Stegun 7.1.26, evaluated in f64, whose
/// error is below 1.5e-7. That is comfortably under the 1e-4 activation
/// tolerance, and computing it in f64 keeps the approximation error well
/// below the f32 rounding that follows.
fn erf(x: f64) -> f64 {
    const A1: f64 = 0.254829592;
    const A2: f64 = -0.284496736;
    const A3: f64 = 1.421413741;
    const A4: f64 = -1.453152027;
    const A5: f64 = 1.061405429;
    const P: f64 = 0.3275911;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + P * x);
    let y = 1.0 - (((((A5 * t + A4) * t) + A3) * t + A2) * t + A1) * t * (-x * x).exp();
    sign * y
}

/// Softmax over `values`, in place.
///
/// The maximum is subtracted first, which is the usual guard against
/// `exp` overflowing.
///
/// **The accumulation is f64.** The reference implementation computes the
/// attention softmax in fp32 and casts back, and a global layer sums over the
/// whole sequence, so f32 accumulation drifts enough to matter at the 1e-4
/// tolerance.
pub fn softmax_in_place(values: &mut [f32]) {
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if !max.is_finite() {
        // Every entry is masked. Leave zeros rather than produce NaN.
        values.fill(0.0);
        return;
    }
    let mut total = 0.0f64;
    for v in values.iter_mut() {
        let e = ((*v - max) as f64).exp();
        *v = e as f32;
        total += e;
    }
    if total > 0.0 {
        let inv = (1.0 / total) as f32;
        for v in values.iter_mut() {
            *v *= inv;
        }
    }
}
