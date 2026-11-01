//! Rotary position embeddings, the `rotate_half` convention.
//!
//! RoPE is the only place position enters this model. There are no absolute
//! position embeddings, despite `position_embedding_type: "absolute"` in
//! `config.json`; that field is vestigial and no such tensor exists.
//!
//! Four things here are easy to get wrong, and three of them fail quietly:
//!
//! 1. **The full head is rotated.** `dim = head_dim = 64`, so all 64 values of
//!    every head turn. There is no partial-rotary fraction in this model, so
//!    a 25% or 50% rotary habit carried over from GPT-NeoX or Phi is wrong
//!    here.
//! 2. **The convention is `rotate_half`, not interleaved.** Value `i` pairs
//!    with value `i + 32`, not with `i + 1`. An interleaved implementation
//!    rotates by the same angles in a different basis: the norms come out
//!    right, nothing overflows, nothing crashes, and every downstream number
//!    is wrong. `tests::halves_not_interleaved` is the test that catches it.
//! 3. **Two thetas, paired counterintuitively.** Local layers get 10000 and
//!    global layers get 160000, so the layers with the *small* window get the
//!    *small* base. [`crate::config::Config::rope_theta`] picks the right one
//!    per layer; do not swap them.
//! 4. **Positions are true absolute positions in local layers too.** The
//!    reference sizes the local layers' cached table to `local_attention`, but
//!    that only bounds the cache. Position indices stay `0..seq_len-1` and are
//!    never wrapped into `0..127`.
//!
//! The reference, transformers 4.48 `modeling_modernbert.py`:
//!
//! ```python
//! def rotate_half(x):
//!     x1 = x[..., : x.shape[-1] // 2]
//!     x2 = x[..., x.shape[-1] // 2 :]
//!     return torch.cat((-x2, x1), dim=-1)
//!
//! q_embed = (q * cos) + (rotate_half(q) * sin)
//! ```
//!
//! with `cos`/`sin` built from a half-length frequency vector concatenated
//! with itself: `emb = torch.cat((freqs, freqs), dim=-1)`.
//!
//! RoPE applies to Q and K only, never to V.

use crate::tensor::Matrix;

/// Precomputed `cos` and `sin` for one theta.
///
/// One row per position, `head_dim` values per row, each row being the
/// half-length frequency vector concatenated with itself. Storing the
/// duplicated form rather than the 32 distinct values costs 128 bytes per
/// position and lets [`apply_rope`] index `cos[i]` directly for every `i`,
/// which is what the reference expression reads like.
///
/// A layer needs exactly one of these, and the whole model needs exactly two:
/// one at 10000 for the local layers and one at 160000 for the global ones.
pub struct RopeTable {
    /// `[max_positions, head_dim]`.
    cos: Matrix,
    /// `[max_positions, head_dim]`.
    sin: Matrix,
    head_dim: usize,
}

impl RopeTable {
    /// cos/sin for positions `0..max_positions` at base `theta`.
    ///
    /// Built in **f64** and stored as f32. The reference builds these in fp32
    /// under `torch.autocast(enabled=False)`, which is already a deliberate
    /// step up from the bf16 activations, and at theta 160000 the
    /// `position * inv_freq` products reach into the thousands by position
    /// 8192, where f32 trigonometry has drifted. Doing the trigonometry in
    /// f64 and rounding once at the end costs nothing, since the tables are
    /// built once per model.
    ///
    /// `attention_scaling` is 1.0 for `rope_type: "default"`, which is what
    /// these configs use, so there is no factor folded into the tables.
    pub fn new(head_dim: usize, max_positions: usize, theta: f32) -> Self {
        assert!(head_dim > 0, "head_dim must be positive");
        assert!(
            head_dim % 2 == 0,
            "head_dim {head_dim} must be even to split into halves"
        );
        assert!(max_positions > 0, "max_positions must be positive");

        let half = head_dim / 2;

        // inv_freq[j] = 1 / theta^(2j / head_dim), for j in 0..head_dim/2.
        // `compute_default_rope_parameters` with no NTK or YaRN scaling.
        let theta = theta as f64;
        let inv_freq: Vec<f64> = (0..half)
            .map(|j| theta.powf(-(2.0 * j as f64) / head_dim as f64))
            .collect();

        let mut cos = Matrix::zeros(max_positions, head_dim);
        let mut sin = Matrix::zeros(max_positions, head_dim);
        for position in 0..max_positions {
            let cos_row = cos.row_mut(position);
            let sin_row = sin.row_mut(position);
            for j in 0..half {
                let angle = position as f64 * inv_freq[j];
                let (s, c) = angle.sin_cos();
                // The duplicated-half form: `cat((freqs, freqs), -1)`. Index
                // j and index j + half carry the same angle, because they are
                // the two components of one rotation.
                cos_row[j] = c as f32;
                cos_row[j + half] = c as f32;
                sin_row[j] = s as f32;
                sin_row[j + half] = s as f32;
            }
        }

        RopeTable { cos, sin, head_dim }
    }

    /// How many positions this table covers.
    pub fn max_positions(&self) -> usize {
        self.cos.rows
    }

    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    /// `cos` for one position, `head_dim` values in the duplicated-half form.
    pub fn cos_row(&self, position: usize) -> &[f32] {
        self.cos.row(position)
    }

    /// `sin` for one position, `head_dim` values in the duplicated-half form.
    pub fn sin_row(&self, position: usize) -> &[f32] {
        self.sin.row(position)
    }
}

/// Rotate `x` in place.
///
/// `x` is `[seq_len, num_heads * head_dim]`, head-major: row `t` holds head
/// 0's `head_dim` values, then head 1's, and so on. This is the layout
/// `Wqkv` already produces, since each of its three 1024-row blocks is
/// head-major.
///
/// The position of row `t` is `t`. Positions are plain `0..seq_len-1` for
/// every layer, global and local alike, and are not offset by padding.
///
/// Each head rotates independently, with the same angles, because `cos` and
/// `sin` depend on the position and the dimension index inside the head, not
/// on which head it is.
pub fn apply_rope(x: &mut Matrix, table: &RopeTable, num_heads: usize, head_dim: usize) {
    assert_eq!(
        head_dim, table.head_dim,
        "the table was built for head_dim {} but was asked for {head_dim}",
        table.head_dim
    );
    assert_eq!(
        x.cols,
        num_heads * head_dim,
        "a {num_heads}-head row of head_dim {head_dim} is {} wide, but x has {} columns",
        num_heads * head_dim,
        x.cols
    );
    assert!(
        x.rows <= table.max_positions(),
        "x has {} positions but the table covers only {}",
        x.rows,
        table.max_positions()
    );

    let half = head_dim / 2;
    for position in 0..x.rows {
        // Copy the two slices out of the table before borrowing `x` mutably.
        let cos = table.cos.row(position);
        let sin = table.sin.row(position);
        let row = x.row_mut(position);

        for head in 0..num_heads {
            let base = head * head_dim;
            for i in 0..half {
                // Both values of the pair are read before either is written.
                // Writing `row[base + i]` first and then reading it back for
                // the second output is the obvious-looking bug here.
                let lo = row[base + i];
                let hi = row[base + i + half];

                // out[i]        = x[i]        * cos[i] - x[i + half] * sin[i]
                // out[i + half] = x[i + half] * cos[i] + x[i]        * sin[i]
                //
                // which is `(x * cos) + (rotate_half(x) * sin)` written out,
                // with `rotate_half` contributing -x2 to the first half and
                // +x1 to the second. cos[i] == cos[i + half] by construction,
                // so one lookup serves both lines.
                row[base + i] = lo * cos[i] - hi * sin[i];
                row[base + i + half] = hi * cos[i] + lo * sin[i];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const HEAD_DIM: usize = 64;
    const LOCAL_THETA: f32 = 10000.0;
    const GLOBAL_THETA: f32 = 160000.0;

    /// Deterministic spread-out values, so a failure is reproducible and the
    /// tests load nothing from disk. Not random; just not constant, since a
    /// constant row hides sign errors.
    fn filled(rows: usize, cols: usize) -> Matrix {
        let data = (0..rows * cols)
            .map(|k| {
                let k = k as f32;
                (0.7 * k + 0.3).sin() * (1.0 + 0.1 * (k % 7.0))
            })
            .collect();
        Matrix::from_vec(rows, cols, data)
    }

    fn head_norm(row: &[f32], head: usize, head_dim: usize) -> f64 {
        row[head * head_dim..(head + 1) * head_dim]
            .iter()
            .map(|&v| v as f64 * v as f64)
            .sum::<f64>()
            .sqrt()
    }

    fn head_dot(a: &[f32], b: &[f32], head: usize, head_dim: usize) -> f64 {
        let span = head * head_dim..(head + 1) * head_dim;
        a[span.clone()]
            .iter()
            .zip(&b[span])
            .map(|(&p, &q)| p as f64 * q as f64)
            .sum()
    }

    /// A rotation preserves length, so each head's L2 norm over its 64 values
    /// must survive. This is the cheap sanity check, and it is also the one
    /// that an interleaved implementation passes, which is why it is not the
    /// only test here.
    #[test]
    fn rotation_preserves_each_head_norm() {
        let heads = 3;
        let table = RopeTable::new(HEAD_DIM, 300, GLOBAL_THETA);
        let before = filled(17, heads * HEAD_DIM);
        let mut after = before.clone();
        apply_rope(&mut after, &table, heads, HEAD_DIM);

        for position in 0..before.rows {
            for head in 0..heads {
                let want = head_norm(before.row(position), head, HEAD_DIM);
                let got = head_norm(after.row(position), head, HEAD_DIM);
                assert!(
                    (want - got).abs() <= 1e-5 * want.max(1.0),
                    "head {head} at position {position}: norm {want} became {got}"
                );
            }
        }
    }

    /// Position 0 is the identity: every angle is `0 * inv_freq = 0`, so
    /// `cos = 1` and `sin = 0` exactly, and the arithmetic is
    /// `x * 1.0 - y * 0.0`. Exact equality is the right assertion.
    #[test]
    fn position_zero_is_the_identity() {
        let heads = 2;
        let table = RopeTable::new(HEAD_DIM, 8, LOCAL_THETA);
        let before = filled(1, heads * HEAD_DIM);
        let mut after = before.clone();
        apply_rope(&mut after, &table, heads, HEAD_DIM);
        assert_eq!(
            before.data, after.data,
            "row 0 is position 0 and must not move"
        );

        // And the table itself, so the reason is visible rather than inferred.
        assert!(table.cos_row(0).iter().all(|&c| c == 1.0));
        assert!(table.sin_row(0).iter().all(|&s| s == 0.0));
    }

    /// The hand-computed case, worked out in full.
    ///
    /// `head_dim = 4`, one head, `theta = 10000`, position 1, `x = [1,2,3,4]`.
    ///
    /// `inv_freq[j] = 1 / theta^(2j / head_dim) = 1 / 10000^(j / 2)`:
    ///
    /// ```text
    /// inv_freq[0] = 1 / 10000^0   = 1
    /// inv_freq[1] = 1 / 10000^0.5 = 1 / 100 = 0.01
    /// ```
    ///
    /// Position 1, so the angles are `1 * inv_freq = [1, 0.01]`, and the
    /// duplicated-half form makes that four values:
    ///
    /// ```text
    /// angles = [1, 0.01, 1, 0.01]
    /// cos    = [cos 1, cos 0.01, cos 1, cos 0.01]
    ///        = [0.5403023059, 0.9999500004, 0.5403023059, 0.9999500004]
    /// sin    = [sin 1, sin 0.01, sin 1, sin 0.01]
    ///        = [0.8414709848, 0.0099998333, 0.8414709848, 0.0099998333]
    /// ```
    ///
    /// With `half = 2`, value 0 pairs with value 2 and value 1 with value 3:
    ///
    /// ```text
    /// out[0] = x[0]*cos[0] - x[2]*sin[0] = 1*0.5403023059 - 3*0.8414709848
    ///        = 0.5403023059 - 2.5244129544 = -1.9841106486
    /// out[1] = x[1]*cos[1] - x[3]*sin[1] = 2*0.9999500004 - 4*0.0099998333
    ///        = 1.9999000008 - 0.0399993333 =  1.9599006675
    /// out[2] = x[2]*cos[2] + x[0]*sin[2] = 3*0.5403023059 + 1*0.8414709848
    ///        = 1.6209069176 + 0.8414709848 =  2.4623779024
    /// out[3] = x[3]*cos[3] + x[1]*sin[3] = 4*0.9999500004 + 2*0.0099998333
    ///        = 3.9998000017 + 0.0199996667 =  4.0197996683
    /// ```
    ///
    /// Interleaved would have paired 0 with 1 and 2 with 3, both at frequency
    /// index `i / 2`, giving a first value of
    /// `1*cos 1 - 2*sin 1 = 0.5403023059 - 1.6829419696 = -1.1426396637`
    /// instead of `-1.9841106486`. Same norms, wrong numbers. See also
    /// `halves_not_interleaved`.
    #[test]
    fn hand_computed_position_one() {
        let head_dim = 4;
        let table = RopeTable::new(head_dim, 2, LOCAL_THETA);

        // Row 0 is position 0, row 1 is position 1. Only row 1 is checked;
        // row 0 is a copy of the same input and must come back untouched.
        let mut x = Matrix::from_vec(2, head_dim, vec![1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0]);
        apply_rope(&mut x, &table, 1, head_dim);

        assert_eq!(x.row(0), &[1.0, 2.0, 3.0, 4.0]);

        // f64, so the worked-out digits above can be written down as they are
        // and compared against the f32 result widened.
        let want: [f64; 4] = [-1.9841106486, 1.9599006675, 2.4623779024, 4.0197996683];
        for (i, &w) in want.iter().enumerate() {
            let got = x.at(1, i) as f64;
            assert!(
                (got - w).abs() < 1e-6,
                "out[{i}]: wanted {w}, got {got} (an interleaved pairing gives \
                 out[0] = -1.1426396637 here)"
            );
        }
    }

    /// The halves convention on its own, with an input chosen so that the two
    /// conventions cannot agree.
    ///
    /// `head_dim = 4`, one head, theta 10000, position 1, `x = [1, 0, 0, 0]`.
    /// Only value 0 is non-zero, so the rotation moves it to whichever index
    /// the convention says it pairs with.
    ///
    /// `rotate_half` pairs 0 with 2, so the mass lands in value 2:
    ///
    /// ```text
    /// out = [cos 1, 0, sin 1, 0] = [0.5403023059, 0, 0.8414709848, 0]
    /// ```
    ///
    /// Interleaved pairs 0 with 1, so it would have landed in value 1:
    ///
    /// ```text
    /// out = [cos 1, sin 1, 0, 0] = [0.5403023059, 0.8414709848, 0, 0]
    /// ```
    ///
    /// Both have L2 norm 1 and neither looks wrong on its own. The difference
    /// is which index holds the 0.8414709848, and that is what is asserted.
    #[test]
    fn halves_not_interleaved() {
        let head_dim = 4;
        let table = RopeTable::new(head_dim, 2, LOCAL_THETA);
        let mut x = Matrix::from_vec(2, head_dim, vec![0.0; 8]);
        x.set(1, 0, 1.0);
        apply_rope(&mut x, &table, 1, head_dim);

        let cos1 = 0.540_302_3_f32;
        let sin1 = 0.841_471_f32;
        assert!(
            (x.at(1, 0) - cos1).abs() < 1e-6,
            "value 0 keeps cos, in either convention"
        );
        assert!(
            x.at(1, 1).abs() < 1e-9,
            "value 1 must stay zero; it is non-zero exactly when the \
             implementation is interleaved, where it would hold {sin1}"
        );
        assert!(
            (x.at(1, 2) - sin1).abs() < 1e-6,
            "value 2 must hold sin, because 0 pairs with 0 + half = 2"
        );
        assert!(x.at(1, 3).abs() < 1e-9, "value 3 must stay zero");
    }

    /// The two thetas must produce different rotations at the same position,
    /// so a layer that reaches for the wrong base cannot pass silently.
    ///
    /// Local layers use 10000 and global layers 160000, which is the
    /// counterintuitive pairing: the small window gets the small base.
    #[test]
    fn the_two_thetas_differ() {
        let local = RopeTable::new(HEAD_DIM, 16, LOCAL_THETA);
        let global = RopeTable::new(HEAD_DIM, 16, GLOBAL_THETA);

        // inv_freq[0] is theta^0 = 1 for any base, so the first frequency is
        // shared. Everything after it separates.
        assert_eq!(local.cos_row(1)[0], global.cos_row(1)[0]);
        assert!(
            (local.cos_row(1)[1] - global.cos_row(1)[1]).abs() > 1e-4,
            "inv_freq[1] must differ between the bases"
        );

        let before = filled(16, HEAD_DIM);
        let mut with_local = before.clone();
        let mut with_global = before.clone();
        apply_rope(&mut with_local, &local, 1, HEAD_DIM);
        apply_rope(&mut with_global, &global, 1, HEAD_DIM);

        let spread = with_local.max_abs_diff(&with_global);
        assert!(
            spread > 1e-2,
            "the two bases rotated the same input to within {spread}, so a \
             wrong-theta layer could hide"
        );
    }

    /// A rotation is orthogonal, so it preserves inner products: two vectors
    /// rotated at the *same* position keep their dot product. A sign error in
    /// one of the two output lines leaves every norm intact and breaks this.
    #[test]
    fn rotation_preserves_dot_products() {
        let table = RopeTable::new(HEAD_DIM, 32, GLOBAL_THETA);
        let rows = 11;
        let a_before = filled(rows, HEAD_DIM);
        let mut b_before = filled(rows, HEAD_DIM);
        // Make b genuinely different from a, and not a scalar multiple.
        for v in b_before.data.iter_mut().rev().step_by(3) {
            *v = -0.5 * *v + 0.25;
        }

        let mut a_after = a_before.clone();
        let mut b_after = b_before.clone();
        apply_rope(&mut a_after, &table, 1, HEAD_DIM);
        apply_rope(&mut b_after, &table, 1, HEAD_DIM);

        for position in 0..rows {
            let want = head_dot(a_before.row(position), b_before.row(position), 0, HEAD_DIM);
            let got = head_dot(a_after.row(position), b_after.row(position), 0, HEAD_DIM);
            assert!(
                (want - got).abs() <= 1e-5 * want.abs().max(1.0),
                "position {position}: dot product {want} became {got}"
            );
        }
    }

    /// Heads in one row rotate independently and identically. Two heads
    /// holding the same 64 values must come out holding the same 64 values,
    /// because the angles depend on the position and on the index inside the
    /// head, never on which head it is.
    ///
    /// This catches a stride mistake: an implementation that walked the whole
    /// row as one 128-value vector, or that reused head 0's slice for head 1,
    /// fails here.
    #[test]
    fn heads_rotate_independently_and_identically() {
        let heads = 2;
        let one_head = filled(9, HEAD_DIM);

        let mut two_heads = Matrix::zeros(one_head.rows, heads * HEAD_DIM);
        for position in 0..one_head.rows {
            let source = one_head.row(position).to_vec();
            let row = two_heads.row_mut(position);
            row[..HEAD_DIM].copy_from_slice(&source);
            row[HEAD_DIM..].copy_from_slice(&source);
        }

        let table = RopeTable::new(HEAD_DIM, 9, LOCAL_THETA);
        let mut rotated_two = two_heads.clone();
        apply_rope(&mut rotated_two, &table, heads, HEAD_DIM);

        let mut rotated_one = one_head.clone();
        apply_rope(&mut rotated_one, &table, 1, HEAD_DIM);

        for position in 0..one_head.rows {
            let row = rotated_two.row(position);
            assert_eq!(
                &row[..HEAD_DIM],
                &row[HEAD_DIM..],
                "the two heads held the same values and must still agree at \
                 position {position}"
            );
            // And both must equal the same input rotated on its own, so
            // "identical to each other" cannot be satisfied by both being
            // wrong in the same way.
            let alone = rotated_one.row(position);
            for (i, (&got, &want)) in row.iter().zip(alone).enumerate() {
                assert!(
                    (got - want).abs() < 1e-6,
                    "position {position} value {i}: one-head gave {want}, \
                     two-head gave {got}"
                );
            }
        }
    }
}
