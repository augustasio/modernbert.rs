//! The feed-forward block: GeGLU, with the exact erf GELU.
//!
//! Two tensors per layer and no bias, because `mlp_bias` is false:
//! `model.layers.{N}.mlp.Wi.weight` with shape `[5248, 1024]` and
//! `model.layers.{N}.mlp.Wo.weight` with shape `[1024, 2624]`. The 5248 is
//! `2 * intermediate_size`, which is the GLU doubling: one projection
//! produces both halves at once.
//!
//! The reference implementation, from `modeling_modernbert.py`:
//!
//! ```python
//! def forward(self, hidden_states):
//!     input, gate = self.Wi(hidden_states).chunk(2, dim=-1)
//!     return self.Wo(self.drop(self.act(input) * gate))
//! ```
//!
//! Three traps live in those two lines, and each gets a comment below at the
//! line where it would otherwise be silently violated:
//!
//! 1. **The activation goes on the FIRST half.** The two halves are named
//!    `input, gate` in that order and the code computes `act(input) * gate`,
//!    so the half called `gate` is the one *without* the activation. This is
//!    backwards from the LLaMA and SwiGLU habit, where the gate is the
//!    activated branch. Swapping the two gives plausible numbers that do not
//!    match the reference.
//! 2. **The halves are contiguous.** `chunk(2, dim=-1)` splits columns
//!    `0..2624` from `2624..5248`. It does not take every other column.
//! 3. **The activation is the exact erf GELU**, not the tanh approximation.
//!    See [`crate::ops::gelu_erf`].
//!
//! `mlp_dropout` is 0.0, so the `drop` in the reference is the identity at
//! inference and has no counterpart here.

use crate::ops::{gelu_erf_in_place, linear};
use crate::safetensors::{Checkpoint, TensorError};
use crate::tensor::Matrix;

/// One layer's feed-forward weights.
pub struct Mlp {
    /// `Wi`, shaped `[2 * intermediate_size, hidden_size]`. Rows
    /// `0..intermediate_size` project the activated branch, the rest project
    /// the multiplicative branch.
    pub(crate) wi: Matrix,
    /// `Wo`, shaped `[hidden_size, intermediate_size]`.
    pub(crate) wo: Matrix,
    /// `intermediate_size`, which is where `Wi`'s output splits. Kept because
    /// it is half of `wi.rows` and the halving is the thing a reader checks.
    pub(crate) intermediate_size: usize,
}

impl Mlp {
    /// Load layer `layer`'s `Wi` and `Wo`.
    ///
    /// Both shapes are required exactly, so a checkpoint that disagrees names
    /// the tensor and both shapes rather than producing a wrongly shaped
    /// matrix that fails much later.
    pub fn load(
        checkpoint: &mut Checkpoint,
        layer: usize,
        hidden_size: usize,
        intermediate_size: usize,
    ) -> Result<Self, TensorError> {
        let wi_name = format!("model.layers.{layer}.mlp.Wi.weight");
        let wo_name = format!("model.layers.{layer}.mlp.Wo.weight");

        // Trap 2, stated as a shape: `Wi` has *two* rows per intermediate
        // feature, one for each half of the GLU.
        let wi = checkpoint.take(&wi_name, &[2 * intermediate_size, hidden_size])?;
        let wo = checkpoint.take(&wo_name, &[hidden_size, intermediate_size])?;

        // No bias is loaded, and none exists: `mlp_bias` is false, so the
        // checkpoint holds no `mlp.Wi.bias` or `mlp.Wo.bias`. The only bias
        // in the whole file is `decoder.bias`.
        Ok(Mlp {
            wi: Matrix::from_vec(2 * intermediate_size, hidden_size, wi),
            wo: Matrix::from_vec(hidden_size, intermediate_size, wo),
            intermediate_size,
        })
    }

    /// `Wo(gelu_erf(first half of Wi(x)) * second half of Wi(x))`.
    ///
    /// `x` is `[seq, hidden_size]` and so is the result.
    pub fn forward(&self, x: &Matrix) -> Matrix {
        // One projection gives both halves: `[seq, 2 * intermediate_size]`.
        let mut projected = linear(x, &self.wi);

        let intermediate = self.intermediate_size;
        let mut gated = Matrix::zeros(projected.rows, intermediate);

        for row in 0..projected.rows {
            // Trap 2: `chunk(2, dim=-1)` is a split down the middle, so the
            // first `intermediate` columns are one branch and the last
            // `intermediate` columns are the other. Nothing is interleaved.
            let (input_half, gate_half) = projected.row_mut(row).split_at_mut(intermediate);

            // Trap 1: the activation goes on the FIRST half, the one the
            // reference calls `input`. The second half, confusingly called
            // `gate`, is multiplied in raw.
            //
            // Trap 3: this is the erf GELU. `gelu_erf_in_place` is the exact
            // form; the tanh approximation differs by up to 3e-4, which is
            // three times the whole fp32 tolerance, and this line runs in all
            // 28 layers.
            gelu_erf_in_place(input_half);

            let out = gated.row_mut(row);
            for i in 0..intermediate {
                out[i] = input_half[i] * gate_half[i];
            }
        }

        linear(&gated, &self.wo)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build an in-memory safetensors file and read it back, so the tests
    /// exercise the real loader without touching `weights/model.safetensors`.
    /// That file is 791 MB on disk and 1.58 GB as f32; see hard rule 8 in
    /// `CLAUDE.md`. Every weight here is written by hand instead.
    ///
    /// The values are stored as `F32` rather than the checkpoint's `BF16`,
    /// which the loader also accepts, so the test arithmetic is not rounded
    /// to 8 mantissa bits before it starts.
    fn checkpoint_of(tensors: &[(String, Vec<usize>, Vec<f32>)]) -> Checkpoint {
        let mut entries: Vec<String> = Vec::new();
        let mut payload: Vec<u8> = Vec::new();

        for (name, shape, data) in tensors {
            let start = payload.len();
            for value in data {
                payload.extend_from_slice(&value.to_le_bytes());
            }
            let dims: Vec<String> = shape.iter().map(usize::to_string).collect();
            entries.push(format!(
                "{:?}:{{\"dtype\":\"F32\",\"shape\":[{}],\"data_offsets\":[{},{}]}}",
                name,
                dims.join(","),
                start,
                payload.len()
            ));
        }

        let header = format!("{{{}}}", entries.join(","));
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(header.as_bytes());
        bytes.extend_from_slice(&payload);

        Checkpoint::from_bytes(&bytes).expect("the synthetic checkpoint must parse")
    }

    /// A one-layer checkpoint holding just `Wi` and `Wo` for layer 0, with
    /// the shapes given rather than inferred, so a test can deliberately
    /// store a wrong one.
    fn layer_zero(wi: (Vec<usize>, Vec<f32>), wo: (Vec<usize>, Vec<f32>)) -> Checkpoint {
        checkpoint_of(&[
            ("model.layers.0.mlp.Wi.weight".to_string(), wi.0, wi.1),
            ("model.layers.0.mlp.Wo.weight".to_string(), wo.0, wo.1),
        ])
    }

    /// Deterministic values in roughly `[-0.5, 0.5]`, for the tests that care
    /// about shapes rather than arithmetic. A linear congruential generator,
    /// so the sequence is the same on every run and every machine.
    fn spread(count: usize, seed: u32) -> Vec<f32> {
        let mut state = seed;
        (0..count)
            .map(|_| {
                state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                let unit = (state >> 8) as f32 / 16_777_216.0;
                unit - 0.5
            })
            .collect()
    }

    fn assert_close(actual: f32, expected: f32, what: &str) {
        assert!(
            (actual - expected).abs() < 1e-5,
            "{what}: got {actual}, expected {expected}"
        );
    }

    /// The real shapes: `[seq, 1024]` in, `[seq, 1024]` out, through a
    /// `[5248, 1024]` and a `[1024, 2624]`. This is the one test with
    /// production-sized weights, and they are synthetic: 32 MB of f32, not
    /// the 1.58 GB checkpoint.
    #[test]
    fn the_hidden_size_survives_the_round_trip() {
        let hidden = 1024;
        let intermediate = 2624;

        let mut checkpoint = layer_zero(
            (
                vec![2 * intermediate, hidden],
                spread(2 * intermediate * hidden, 1),
            ),
            (vec![hidden, intermediate], spread(hidden * intermediate, 2)),
        );
        let mlp =
            Mlp::load(&mut checkpoint, 0, hidden, intermediate).expect("the shapes are right");

        let x = Matrix::from_vec(3, hidden, spread(3 * hidden, 3));
        let y = mlp.forward(&x);

        assert_eq!(y.shape(), (3, hidden));
        assert!(
            y.data.iter().all(|v| v.is_finite()),
            "every output value must be finite"
        );
    }

    /// `gelu_erf(0) == 0`, so a zero input gives a zero output whatever the
    /// weights are. The products are exactly zero, not nearly, so this
    /// asserts equality: it catches a stray bias or a constant added
    /// somewhere in the chain.
    #[test]
    fn a_zero_input_gives_a_zero_output() {
        let hidden = 3;
        let intermediate = 2;

        let mut checkpoint = layer_zero(
            (
                vec![2 * intermediate, hidden],
                spread(2 * intermediate * hidden, 7),
            ),
            (vec![hidden, intermediate], spread(hidden * intermediate, 8)),
        );
        let mlp =
            Mlp::load(&mut checkpoint, 0, hidden, intermediate).expect("the shapes are right");

        let y = mlp.forward(&Matrix::zeros(4, hidden));

        assert_eq!(y.shape(), (4, hidden));
        assert!(
            y.data.iter().all(|&v| v == 0.0),
            "a zero input must give exact zeros, got {y:?}"
        );
    }

    /// The whole block worked out by hand, with `hidden_size = 2` and
    /// `intermediate_size = 2`.
    ///
    /// ```text
    /// x  = [1, 2]
    ///
    /// Wi = [[1, 0],      row 0, input branch unit 0 -> 1*1 + 0*2 = 1
    ///       [0, 1],      row 1, input branch unit 1 -> 0*1 + 1*2 = 2
    ///       [1, 1],      row 2, gate  branch unit 0 -> 1*1 + 1*2 = 3
    ///       [0, 0.5]]    row 3, gate  branch unit 1 -> 0*1 + 0.5*2 = 1
    ///
    /// input = [1, 2]     the FIRST half, so this is the activated one
    /// gate  = [3, 1]     the second half, multiplied in raw
    ///
    /// gelu_erf(1) = 1 * 0.5 * (1 + erf(1/sqrt(2))) = 0.5 * 1.6826894921
    ///             = 0.8413447461
    /// gelu_erf(2) = 2 * 0.5 * (1 + erf(2/sqrt(2))) = 1 + 0.9544997361
    ///             = 1.9544997361
    ///
    /// h = gelu_erf(input) * gate
    ///   = [0.8413447461 * 3, 1.9544997361 * 1]
    ///   = [2.5240342383, 1.9544997361]
    ///
    /// Wo = [[1, 0],      out 0 = 1*h0 + 0*h1 = 2.5240342383
    ///       [1, 1]]      out 1 = 1*h0 + 1*h1 = 4.4785339744
    /// ```
    ///
    /// The erf values are the familiar normal-distribution ones: `erf(1/√2)`
    /// is the one-sigma mass 0.6826894921 and `erf(2/√2)` is the two-sigma
    /// mass 0.9544997361.
    #[test]
    fn the_hand_computed_case() {
        let mut checkpoint = layer_zero(
            (vec![4, 2], vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.5]),
            (vec![2, 2], vec![1.0, 0.0, 1.0, 1.0]),
        );
        let mlp = Mlp::load(&mut checkpoint, 0, 2, 2).expect("the shapes are right");

        let y = mlp.forward(&Matrix::from_vec(1, 2, vec![1.0, 2.0]));

        assert_eq!(y.shape(), (1, 2));
        assert_close(y.at(0, 0), 2.524_034_2, "out 0");
        assert_close(y.at(0, 1), 4.478_534, "out 1");

        // Had the activation gone on the second half instead, the same
        // weights would give gelu_erf(gate) * input
        //   = [2.9959503 * 1, 0.8413447 * 2] = [2.9959503, 1.6826895]
        // and so out 0 = 2.9959503, out 1 = 4.6786398. Both are the right
        // order of magnitude, which is exactly why this trap survives.
        assert!(
            (y.at(0, 0) - 2.9959503).abs() > 0.1,
            "out 0 must not match the swapped convention"
        );
    }

    /// Trap 1 on its own: the activation goes on the first half.
    ///
    /// `hidden_size = 1`, `intermediate_size = 1`, so `Wi` is `[2, 1]` and
    /// `Wo` is `[1, 1]`, and the two conventions reduce to two numbers.
    ///
    /// ```text
    /// x  = [2]
    /// Wi = [[0.5],   input branch -> 0.5 * 2 = 1
    ///       [1.5]]   gate  branch -> 1.5 * 2 = 3
    /// Wo = [[1]]
    ///
    /// first half activated (correct):
    ///     gelu_erf(1) * 3 = 0.8413447461 * 3 = 2.5240342383
    /// second half activated (the LLaMA/SwiGLU habit, wrong here):
    ///     gelu_erf(3) * 1 = 2.9959502985 * 1 = 2.9959502985
    /// ```
    ///
    /// The two differ by 0.47 here, where a real layer averages over 2624
    /// units and the difference hides in the noise.
    #[test]
    fn the_activation_goes_on_the_first_half() {
        let mut checkpoint = layer_zero((vec![2, 1], vec![0.5, 1.5]), (vec![1, 1], vec![1.0]));
        let mlp = Mlp::load(&mut checkpoint, 0, 1, 1).expect("the shapes are right");

        let y = mlp.forward(&Matrix::from_vec(1, 1, vec![2.0]));

        assert_close(y.at(0, 0), 2.524_034_2, "first half activated");
        assert!(
            (y.at(0, 0) - 2.995_950_3).abs() > 0.4,
            "activating the second half would have given 2.9959503, got {}",
            y.at(0, 0)
        );
    }

    /// A wrong `Wi` shape is an error that names the tensor, not a panic
    /// inside `Matrix::from_vec` and not a silently reshaped weight.
    #[test]
    fn a_wrong_shaped_weight_names_the_tensor() {
        // `Wi` must have 2 * intermediate_size rows. Three rows is the
        // mistake someone makes by forgetting the GLU doubling.
        let mut checkpoint = layer_zero(
            (vec![3, 2], vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0]),
            (vec![2, 2], vec![1.0, 0.0, 0.0, 1.0]),
        );

        let message = match Mlp::load(&mut checkpoint, 0, 2, 2) {
            Err(error) => error.to_string(),
            Ok(_) => panic!("a Wi with 3 rows must not load"),
        };

        assert!(
            message.contains("model.layers.0.mlp.Wi.weight"),
            "the message must name the tensor, got {message:?}"
        );
        assert!(
            message.contains("[3, 2]") && message.contains("[4, 2]"),
            "the message must give both shapes, got {message:?}"
        );
    }

    /// Trap 2: the halves are contiguous, not interleaved.
    ///
    /// `hidden_size = 1`, `intermediate_size = 2`, so `Wi` is `[4, 1]` and
    /// with `x = [1]` the projection is just `Wi`'s column.
    ///
    /// ```text
    /// Wi(x) = [1, 0, 2, 3]
    ///
    /// contiguous halves (correct):
    ///     input = [1, 0], gate = [2, 3]
    ///     h = [gelu_erf(1) * 2, gelu_erf(0) * 3]
    ///       = [1.6826894921, 0]
    ///     out = h0 + h1 = 1.6826894921
    ///
    /// every other column (wrong):
    ///     input = [1, 2], gate = [0, 3]
    ///     h = [gelu_erf(1) * 0, gelu_erf(2) * 3]
    ///       = [0, 5.8634992083]
    ///     out = h0 + h1 = 5.8634992083
    /// ```
    #[test]
    fn the_halves_are_contiguous() {
        let mut checkpoint = layer_zero(
            (vec![4, 1], vec![1.0, 0.0, 2.0, 3.0]),
            (vec![1, 2], vec![1.0, 1.0]),
        );
        let mlp = Mlp::load(&mut checkpoint, 0, 1, 2).expect("the shapes are right");

        let y = mlp.forward(&Matrix::from_vec(1, 1, vec![1.0]));

        assert_close(y.at(0, 0), 1.682_689_5, "contiguous halves");
        assert!(
            (y.at(0, 0) - 5.863_499).abs() > 1.0,
            "an interleaved split would have given 5.8634992, got {}",
            y.at(0, 0)
        );
    }

    /// Trap 3: the exact erf GELU, not the tanh approximation.
    ///
    /// The two forms differ most around `|x| = 2`, where the gap is about
    /// 8e-5. So drive the activated branch to exactly 2 with a gate of 1 and
    /// an identity `Wo`, which makes the output the activation itself:
    ///
    /// ```text
    /// erf form:  2 * 0.5 * (1 + erf(2/sqrt(2))) = 1.9544997361
    /// tanh form: 0.5 * 2 * (1 + tanh(sqrt(2/pi) * (2 + 0.044715 * 8)))
    ///                                           = 1.9545827
    /// ```
    ///
    /// The tanh value is recomputed here rather than trusted as a constant,
    /// and the test asserts both that the answer is the erf one and that the
    /// two forms really are far enough apart for this input to tell them
    /// apart.
    #[test]
    fn the_activation_is_erf_and_not_tanh() {
        let mut checkpoint = layer_zero((vec![2, 1], vec![2.0, 1.0]), (vec![1, 1], vec![1.0]));
        let mlp = Mlp::load(&mut checkpoint, 0, 1, 1).expect("the shapes are right");

        let y = mlp.forward(&Matrix::from_vec(1, 1, vec![1.0]));

        let tanh_form = {
            let x = 2.0f64;
            let inner = (2.0 / std::f64::consts::PI).sqrt() * (x + 0.044715 * x * x * x);
            (0.5 * x * (1.0 + inner.tanh())) as f32
        };

        assert!(
            (1.954_499_7 - tanh_form).abs() > 1e-5,
            "this input must separate the two GELU forms, tanh gives {tanh_form}"
        );
        assert_close(y.at(0, 0), 1.954_499_7, "erf GELU");
        assert!(
            (y.at(0, 0) - tanh_form).abs() > 1e-5,
            "the output matches the tanh approximation {tanh_form}, so the wrong \
             GELU is in use"
        );
    }
}
