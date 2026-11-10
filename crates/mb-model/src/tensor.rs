//! The array types the forward pass uses.
//!
//! Two shapes cover everything: a 2-D row-major [`Matrix`] for weights and
//! activations, and a plain `Vec<f32>` for the 1-D LayerNorm weights and the
//! decoder bias. Attention wants per-head 3-D views, and those are done by
//! indexing a `Matrix` with the stride written out at the call site, which
//! reads better than a general n-dimensional type would.
//!
//! Everything is `f32`. The checkpoint is bf16 and gets upcast on load; see
//! `safetensors.rs`.

/// A row-major 2-D array of `f32`.
///
/// Activations keep one row per token, so a sequence of 152 tokens with a
/// hidden size of 1024 is a 152 by 1024 matrix. Weights keep the shape
/// safetensors gives them, which for a linear layer is
/// `[out_features, in_features]`.
#[derive(Clone, PartialEq)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    /// `rows * cols` values, row by row.
    pub data: Vec<f32>,
    /// Rung 17: the same values in bf16, for a weight that goes through the
    /// bf16 matmul (`onednn.rs`). Empty for every other matrix, which then
    /// takes the f32 path.
    pub bf16: Vec<u16>,
    /// Rung 20: this weight in s8, for the int8 matmul. Empty for every other
    /// matrix. Each row (an output column of the product) has its own scale.
    pub int8: Vec<i8>,
    /// Rung 20: for each row, one over the scale its s8 values were made with.
    pub int8_unscale: Vec<f32>,
    /// Rung 23: one factor per input channel (column), 1 / s_j, that the
    /// input is multiplied by before it is rounded to s8. Empty when the
    /// weight is not smoothed. See [`Matrix::keep_int8_smoothed`].
    pub int8_smooth: Vec<f32>,
}

impl Matrix {
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Matrix {
            rows,
            cols,
            data: vec![0.0; rows * cols],
            bf16: Vec::new(),
            int8: Vec::new(),
            int8_unscale: Vec::new(),
            int8_smooth: Vec::new(),
        }
    }

    pub fn from_vec(rows: usize, cols: usize, data: Vec<f32>) -> Self {
        assert_eq!(
            data.len(),
            rows * cols,
            "a {rows} by {cols} matrix needs {} values, got {}",
            rows * cols,
            data.len()
        );
        Matrix {
            rows,
            cols,
            data,
            bf16: Vec::new(),
            int8: Vec::new(),
            int8_unscale: Vec::new(),
            int8_smooth: Vec::new(),
        }
    }

    /// Rung 17: keep a bf16 copy of this weight beside the f32 one.
    ///
    /// The checkpoint is bf16 and the f32 values are its exact upcast, so
    /// the copy is exact too: this asserts that no value had low bits to
    /// lose. The f32 copy stays, for the paths that still read it.
    #[cfg(feature = "bf16")]
    pub fn keep_bf16(&mut self) {
        self.bf16 = self
            .data
            .iter()
            .map(|&x| {
                let bits = x.to_bits();
                assert_eq!(
                    bits & 0xFFFF,
                    0,
                    "{x} is not a bf16 value, so the copy would round it"
                );
                (bits >> 16) as u16
            })
            .collect();
    }

    /// Rung 20: keep an s8 copy of this weight, one scale per row. The row is
    /// `[K]` long, one output column of the product. The f32 copy stays.
    #[cfg(feature = "int8")]
    pub fn keep_int8(&mut self) {
        self.int8.clear();
        self.int8_unscale.clear();
        for row in self.data.chunks_exact(self.cols) {
            let scale = crate::onednn::int8_scale(row);
            self.int8_unscale.push(1.0 / scale);
            self.int8
                .extend(row.iter().map(|&w| (w * scale).round_ties_even() as i8));
        }
    }

    /// Rung 23: keep an s8 copy of this weight with each input channel j
    /// multiplied by `smooth[j]` (SmoothQuant's s_j) before rounding, and
    /// remember 1 / s_j for the input. The product is unchanged in exact
    /// arithmetic, `x W^T = (x / s)(W s)^T`, but `x / s` has flatter rows, so
    /// one scale per row loses less to its few large channels.
    #[cfg(feature = "int8")]
    pub fn keep_int8_smoothed(&mut self, smooth: &[f32]) {
        assert_eq!(
            smooth.len(),
            self.cols,
            "{} smoothing scales for a weight with {} input channels",
            smooth.len(),
            self.cols
        );
        self.int8.clear();
        self.int8_unscale.clear();
        let mut row_scaled = vec![0.0f32; self.cols];
        for row in self.data.chunks_exact(self.cols) {
            for ((y, &w), &s) in row_scaled.iter_mut().zip(row).zip(smooth) {
                *y = w * s;
            }
            let scale = crate::onednn::int8_scale(&row_scaled);
            self.int8_unscale.push(1.0 / scale);
            self.int8.extend(
                row_scaled
                    .iter()
                    .map(|&w| (w * scale).round_ties_even() as i8),
            );
        }
        self.int8_smooth = smooth.iter().map(|&s| 1.0 / s).collect();
    }

    /// Change the number of rows, keeping the buffer.
    ///
    /// Rung 8's workspace holds one buffer per activation and reuses it for
    /// every forward pass. A shorter sequence than the buffer has held
    /// before never allocates. A longer one grows the buffer once. The
    /// values are left as they were, so the caller overwrites every row.
    pub fn resize_rows(&mut self, rows: usize) {
        self.data.resize(rows * self.cols, 0.0);
        self.rows = rows;
    }

    #[inline]
    pub fn row(&self, i: usize) -> &[f32] {
        &self.data[i * self.cols..(i + 1) * self.cols]
    }

    #[inline]
    pub fn row_mut(&mut self, i: usize) -> &mut [f32] {
        &mut self.data[i * self.cols..(i + 1) * self.cols]
    }

    #[inline]
    pub fn at(&self, row: usize, col: usize) -> f32 {
        self.data[row * self.cols + col]
    }

    #[inline]
    pub fn set(&mut self, row: usize, col: usize, value: f32) {
        self.data[row * self.cols + col] = value;
    }

    pub fn shape(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    /// The largest absolute difference against `other`, for the tier-2
    /// comparisons against PyTorch.
    pub fn max_abs_diff(&self, other: &Matrix) -> f32 {
        assert_eq!(self.shape(), other.shape(), "shapes must match to compare");
        self.data
            .iter()
            .zip(&other.data)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max)
    }

    /// Cosine similarity against `other`, flattened.
    ///
    /// This catches what the largest absolute difference misses: a uniform
    /// scale error on small activations gives a tiny absolute difference and
    /// a cosine well below 1.
    pub fn cosine_similarity(&self, other: &Matrix) -> f32 {
        assert_eq!(self.shape(), other.shape(), "shapes must match to compare");
        let mut dot = 0.0f64;
        let mut left = 0.0f64;
        let mut right = 0.0f64;
        for (&a, &b) in self.data.iter().zip(&other.data) {
            dot += a as f64 * b as f64;
            left += a as f64 * a as f64;
            right += b as f64 * b as f64;
        }
        if left == 0.0 || right == 0.0 {
            return if left == right { 1.0 } else { 0.0 };
        }
        (dot / (left.sqrt() * right.sqrt())) as f32
    }
}

impl std::fmt::Debug for Matrix {
    /// Print the shape and a few values. A full 152 by 1024 matrix in a test
    /// failure is unreadable.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Matrix {}x{} [", self.rows, self.cols)?;
        for (i, v) in self.data.iter().take(6).enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{v:.5}")?;
        }
        if self.data.len() > 6 {
            write!(f, ", ... {} more", self.data.len() - 6)?;
        }
        write!(f, "]")
    }
}
