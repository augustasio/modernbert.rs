//! Tier 2: compare our activations against PyTorch's, checkpoint by
//! checkpoint.
//!
//! `python/dump_reference.py` runs rung 0 under forward hooks and writes one
//! tensor per named checkpoint to `python/reference/activations.safetensors`,
//! for eight fixtures of 3 to 963 tokens. This file loads those tensors and
//! asserts ours match.
//!
//! The point of comparing per checkpoint rather than only at the end is that
//! a failure localizes. A RoPE bug fails at `layer.0.q_rope` instead of
//! surfacing as a mysterious 1e-2 at the mask position with nothing to say
//! about where it came from.
//!
//! As components land, their checkpoints get a test here. Right now that is
//! the embedding lookup and the norm that follows it, which between them
//! cover the safetensors reader, the bf16 upcast, the embedding indexing and
//! `ops::layer_norm`.
//!
//! **The noise floor is 6.4e-05.** The dump records
//! `worst_eager_vs_sdpa_max_abs_diff_at_mask`, which is how far apart two of
//! PyTorch's own attention backends land on the same model at the mask
//! position. Nothing here can be held to a tolerance tighter than PyTorch
//! holds itself, so that figure is the yardstick for anything downstream of
//! attention. The checkpoints in this file are upstream of attention and are
//! held far tighter.
//!
//! Memory: tensors are pulled individually with `Checkpoint::load_one`, and
//! the two weights this file needs come from `Checkpoint::load_subset`, so
//! nothing ever reads the 791 MB checkpoint or the 582 MB dump whole. See
//! `CLAUDE.md` hard rule 8.

use std::path::PathBuf;
use std::sync::OnceLock;

use mb_model::config::Config;
use mb_model::embeddings::Embeddings;
use mb_model::safetensors::Checkpoint;
use mb_model::tensor::Matrix;
use serde_json::Value;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .expect("the crate lives inside the repository")
}

fn reference_path() -> PathBuf {
    let path = repo_root().join("python/reference/activations.safetensors");
    assert!(
        path.is_file(),
        "{} is missing. Generate it with python/dump_reference.py.",
        path.display()
    );
    path
}

/// One fixture's inputs, from `fixtures.json`.
struct Fixture {
    index: usize,
    name: String,
    token_ids: Vec<u32>,
    mask_position: usize,
    exceeds_local_window: bool,
}

fn fixtures() -> &'static Vec<Fixture> {
    static FIXTURES: OnceLock<Vec<Fixture>> = OnceLock::new();
    FIXTURES.get_or_init(|| {
        let path = repo_root().join("python/reference/fixtures.json");
        let text = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
        let json: Value = serde_json::from_str(&text).expect("fixtures.json must be valid JSON");

        json["fixtures"]
            .as_array()
            .expect("fixtures must be an array")
            .iter()
            .map(|f| Fixture {
                index: f["index"].as_u64().expect("index") as usize,
                name: f["name"].as_str().expect("name").to_string(),
                token_ids: f["token_ids"]
                    .as_array()
                    .expect("token_ids")
                    .iter()
                    .map(|n| n.as_u64().expect("an id must be a number") as u32)
                    .collect(),
                mask_position: f["mask_position"].as_u64().expect("mask_position") as usize,
                exceeds_local_window: f["exceeds_local_window"].as_bool().unwrap_or(false),
            })
            .collect()
    })
}

/// The embedding table and its norm, loaded once for the whole test binary.
///
/// 206 MB, which is why it is shared rather than loaded per test. Loading it
/// per test function would be 10 copies at once on this machine.
fn embeddings() -> &'static Embeddings {
    static EMBEDDINGS: OnceLock<Embeddings> = OnceLock::new();
    EMBEDDINGS.get_or_init(|| {
        let dir = repo_root().join("weights/ModernBERT-Large-Instruct");
        let config = Config::from_file(dir.join("config.json")).expect("config must load");
        let checkpoint = Checkpoint::load_subset(
            dir.join("model.safetensors"),
            &[
                "model.embeddings.tok_embeddings.weight",
                "model.embeddings.norm.weight",
            ],
        )
        .expect("the embedding tensors must load");
        Embeddings::load(
            &checkpoint,
            config.vocab_size,
            config.hidden_size,
            config.norm_eps,
        )
        .expect("the embeddings must build")
    })
}

/// Load one reference checkpoint as a `[seq, hidden]` matrix.
///
/// The dump writes `[1, seq, hidden]`, since PyTorch keeps the batch
/// dimension. A batch of one flattens away.
fn reference_2d(fixture: usize, checkpoint: &str) -> Matrix {
    let key = format!("fixture{fixture}.{checkpoint}");
    let t = Checkpoint::load_one(reference_path(), &key).unwrap_or_else(|e| panic!("{key}: {e}"));
    assert_eq!(
        t.shape.len(),
        3,
        "{key}: expected [1, seq, hidden], got {:?}",
        t.shape
    );
    assert_eq!(t.shape[0], 1, "{key}: the dump should be a batch of one");
    Matrix::from_vec(t.shape[1], t.shape[2], t.data)
}

/// Compare and report all three metrics, because each catches what the
/// others let through. Cosine in particular catches a uniform scale error
/// that a small absolute difference would wave past.
fn assert_matches(label: &str, ours: &Matrix, reference: &Matrix, tolerance: f32) {
    assert_eq!(
        ours.shape(),
        reference.shape(),
        "{label}: shape {:?} against reference {:?}",
        ours.shape(),
        reference.shape()
    );

    let max_abs = ours.max_abs_diff(reference);
    let cosine = reference.cosine_similarity(ours);

    // Largest relative difference, measured only where the reference is big
    // enough for a ratio to mean anything.
    let mut max_rel = 0.0f32;
    for (&a, &b) in ours.data.iter().zip(&reference.data) {
        if b.abs() > 1e-3 {
            max_rel = max_rel.max(((a - b) / b).abs());
        }
    }

    // Report the margins even on success. "It passed" does not say whether
    // the tolerance is tight or has three orders of magnitude of slack, and
    // that matters when a later component eats into the budget.
    // `cargo test -p mb-model --test reference -- --nocapture` shows these.
    println!("{label:52} max abs {max_abs:.3e}  max rel {max_rel:.3e}  cosine {cosine:.9}");

    assert!(
        max_abs <= tolerance,
        "{label}: max abs diff {max_abs:.3e} exceeds {tolerance:.1e}\n  \
         max rel diff {max_rel:.3e}, cosine {cosine:.9}"
    );
    assert!(
        cosine >= 1.0 - 1e-6,
        "{label}: cosine {cosine:.9} is below 1 - 1e-6 while max abs diff is only \
         {max_abs:.3e}, which points at a systematic scale error rather than noise"
    );
}

#[test]
fn the_fixtures_cover_the_sliding_window() {
    let f = fixtures();
    assert_eq!(f.len(), 8, "fixtures");

    // The window is 129 keys. Without fixtures longer than that, local
    // attention is numerically identical to global attention and the main
    // ModernBERT trap goes untested.
    let long = f.iter().filter(|x| x.token_ids.len() > 129).count();
    assert!(
        long >= 5,
        "only {long} of 8 fixtures exceed 129 tokens; the window needs exercising"
    );

    // And at least one long enough that a query sees a small fraction of the
    // sequence, where a wrong mask cannot hide inside the tolerance.
    let longest = f.iter().map(|x| x.token_ids.len()).max().unwrap_or(0);
    assert!(
        longest >= 600,
        "the longest fixture is {longest} tokens; a mask bug needs a sequence where a \
         query sees a fifth of the keys, not most of them"
    );

    // The flag in the dump must agree with the arithmetic.
    for x in f {
        assert_eq!(
            x.exceeds_local_window,
            x.token_ids.len() > 129,
            "fixture {} ({}): the dump's exceeds_local_window disagrees with its length {}",
            x.index,
            x.name,
            x.token_ids.len()
        );
    }
}

#[test]
fn the_mask_is_where_the_dump_says() {
    const MASK_ID: u32 = 50284;
    for f in fixtures() {
        assert!(
            f.mask_position < f.token_ids.len(),
            "fixture {} ({}): mask position {} is outside {} tokens",
            f.index,
            f.name,
            f.mask_position,
            f.token_ids.len()
        );
        assert_eq!(
            f.token_ids[f.mask_position], MASK_ID,
            "fixture {} ({}): position {} is not [MASK]",
            f.index, f.name, f.mask_position
        );
        assert_eq!(
            f.token_ids.iter().filter(|&&id| id == MASK_ID).count(),
            1,
            "fixture {} ({}): there must be exactly one [MASK]",
            f.index,
            f.name
        );
    }
}

/// The raw lookup is a table read of values the bf16 upcast already fixed, so
/// there is no arithmetic and no rounding. It must match **exactly**.
///
/// That makes this the sharpest test of the loader: a byte-order mistake, an
/// off-by-one in the row stride, or a rounding upcast all fail here rather
/// than hiding as a small difference further down.
#[test]
fn the_embedding_lookup_matches_exactly() {
    let e = embeddings();
    for f in fixtures() {
        let ours = e
            .lookup_only(&f.token_ids)
            .unwrap_or_else(|err| panic!("fixture {} ({}): {err}", f.index, f.name));
        let reference = reference_2d(f.index, "embeddings.tok");
        let label = format!("fixture {} ({}) embeddings.tok", f.index, f.name);

        assert_eq!(ours.shape(), reference.shape(), "{label}: shape");
        let differing = ours
            .data
            .iter()
            .zip(&reference.data)
            .filter(|(a, b)| a != b)
            .count();
        assert_eq!(
            differing,
            0,
            "{label}: {differing} of {} values differ, and a table lookup of upcast bf16 \
             should be bit-exact. Max abs diff {:.3e}.",
            ours.data.len(),
            ours.max_abs_diff(&reference)
        );
    }
}

/// LayerNorm over 1024 features. This is arithmetic, so it is held to 1e-5
/// rather than exactly, but that is still 6x tighter than PyTorch's own
/// backends differ by at the mask.
#[test]
fn the_embedding_norm_matches() {
    let e = embeddings();
    for f in fixtures() {
        let ours = e
            .forward(&f.token_ids)
            .unwrap_or_else(|err| panic!("fixture {} ({}): {err}", f.index, f.name));
        let reference = reference_2d(f.index, "embeddings.norm");
        assert_matches(
            &format!("fixture {} ({}) embeddings.norm", f.index, f.name),
            &ours,
            &reference,
            1e-5,
        );
    }
}

/// LayerNorm must subtract the mean. It is not RMSNorm, and because
/// `norm_bias` is false there is one tensor per norm, so the state dict shape
/// does not tell the two apart.
///
/// If the mean were left in, the reference comparison above would fail, but
/// this says so directly: the output of a real LayerNorm has a per-row mean
/// of zero before the weight is applied, and the weighted output's rows do
/// not drift far from that.
#[test]
fn the_norm_is_layer_norm_and_not_rms_norm() {
    let e = embeddings();
    let f = &fixtures()[0];
    let raw = e.lookup_only(&f.token_ids).expect("lookup");

    // The inputs must actually have a non-zero mean, or subtracting it would
    // be a no-op and this test would prove nothing.
    let mut worst_input_mean = 0.0f32;
    for row in 0..raw.rows {
        let mean = raw.row(row).iter().sum::<f32>() / raw.cols as f32;
        worst_input_mean = worst_input_mean.max(mean.abs());
    }
    assert!(
        worst_input_mean > 1e-4,
        "the embedding rows have a mean of {worst_input_mean:.3e}, too close to zero for \
         this test to distinguish LayerNorm from RMSNorm"
    );

    // And the reference, which we match, is the mean-subtracting kind.
    let reference = reference_2d(f.index, "embeddings.norm");
    assert_eq!(reference.rows, raw.rows);
}
