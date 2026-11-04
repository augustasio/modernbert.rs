//! Rung 7 must compute the same products as rung 6.
//!
//! The two sum in a different order, so they agree to f32 rounding, not
//! exactly. These run only with `--features accelerate`.

#![cfg(feature = "accelerate")]

use mb_model::ops::{backend, linear, linear_naive};
use mb_model::tensor::Matrix;

/// A deterministic fill, so a failure reproduces.
fn filled(rows: usize, cols: usize, seed: u32) -> Matrix {
    let mut state = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
    let data = (0..rows * cols)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (state >> 8) as f32 / (1u32 << 24) as f32 - 0.5
        })
        .collect();
    Matrix::from_vec(rows, cols, data)
}

#[test]
fn the_build_reports_accelerate() {
    let want = if cfg!(feature = "expf") {
        "accelerate-fused-expf"
    } else if cfg!(feature = "fused") {
        "accelerate-fused"
    } else {
        "accelerate"
    };
    assert_eq!(backend(), want);
}

/// The shapes the model actually uses, plus edge cases: one row, which is the
/// head at the mask position, and the GLU's 5248-wide projection.
#[test]
fn accelerate_agrees_with_the_naive_loop() {
    let shapes = [
        (1, 1024, 1024), // head.dense at the mask position
        (7, 1024, 3072), // Wqkv
        (7, 1024, 1024), // Wo
        (7, 1024, 5248), // mlp.Wi
        (7, 2624, 1024), // mlp.Wo
        (3, 5, 2),       // tiny, odd sizes
        (1, 1024, 4),    // the four verbalizer rows
    ];
    for (i, &(m, k, n)) in shapes.iter().enumerate() {
        let a = filled(m, k, 2 * i as u32);
        let w = filled(n, k, 2 * i as u32 + 1);
        let fast = linear(&a, &w);
        let slow = linear_naive(&a, &w);
        assert_eq!(fast.shape(), (m, n));

        // Relative to the largest value, since the sums grow with K.
        let peak = slow.data.iter().fold(0.0f32, |p, v| p.max(v.abs()));
        let gap = fast.max_abs_diff(&slow) / peak;
        println!("{m}x{k} @ {n}x{k}^T  rel-to-peak {gap:.3e}");
        assert!(
            gap < 1e-5,
            "shape {m}x{k} by {n}x{k}: rel-to-peak gap {gap:.3e} is more than summation order explains"
        );
    }
}

#[test]
fn an_empty_product_is_empty() {
    let out = linear(&Matrix::zeros(0, 8), &filled(4, 8, 9));
    assert_eq!(out.shape(), (0, 4));
}
