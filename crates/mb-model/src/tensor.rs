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
}

impl Matrix {
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Matrix {
            rows,
            cols,
            data: vec![0.0; rows * cols],
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
        Matrix { rows, cols, data }
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
