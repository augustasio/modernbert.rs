//! Tier 2 through the whole stack: every one of the 28 layers against
//! PyTorch, then the verbalizer logits and the prediction.
//!
//! The forward pass reports each checkpoint through `ModernBert::encode`'s
//! observer, and each one is compared against the dump as it goes past. So a
//! wrong mask fails at `layer.1.out`, the first local layer, rather than as
//! a vague drift at the end.
//!
//! Memory: the model loads once for this binary, about 3.2 GB at peak and
//! 1.58 GB resident. Test binaries run one after another, so this never
//! overlaps another binary's load. See `CLAUDE.md` hard rules 8 and 9.
//!
//! Time: rung 6's naive matmul runs at a few GFLOP/s, so the default test
//! covers three fixtures of 3, 119 and 183 tokens. The long ones, where a
//! query sees a fifth of the sequence and a mask bug cannot hide, run with
//! `cargo test -p mb-model --test layers -- --ignored`.

use std::path::PathBuf;
use std::sync::OnceLock;

use mb_model::attention::Masks;
use mb_model::model::ModernBert;
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
    repo_root().join("python/reference/activations.safetensors")
}

fn model() -> &'static ModernBert {
    static MODEL: OnceLock<ModernBert> = OnceLock::new();
    MODEL.get_or_init(|| {
        ModernBert::load(repo_root().join("weights/ModernBERT-Large-Instruct"))
            .expect("the model must load")
    })
}

struct Fixture {
    index: usize,
    name: String,
    token_ids: Vec<u32>,
    mask_position: usize,
}

fn fixtures() -> (Vec<Fixture>, Vec<u32>) {
    let path = repo_root().join("python/reference/fixtures.json");
    let text = std::fs::read_to_string(&path).expect("fixtures.json");
    let json: Value = serde_json::from_str(&text).expect("valid JSON");
    let ids = |v: &Value| -> Vec<u32> {
        v.as_array()
            .expect("an id array")
            .iter()
            .map(|n| n.as_u64().expect("an id") as u32)
            .collect()
    };
    let fixtures = json["fixtures"]
        .as_array()
        .expect("fixtures")
        .iter()
        .map(|f| Fixture {
            index: f["index"].as_u64().expect("index") as usize,
            name: f["name"].as_str().expect("name").to_string(),
            token_ids: ids(&f["token_ids"]),
            mask_position: f["mask_position"].as_u64().expect("mask_position") as usize,
        })
        .collect();
    (fixtures, ids(&json["verbalizer_ids"]))
}

fn reference(fixture: usize, checkpoint: &str) -> Matrix {
    let key = format!("fixture{fixture}.{checkpoint}");
    let t = Checkpoint::load_one(reference_path(), &key).unwrap_or_else(|e| panic!("{key}: {e}"));
    match t.shape.as_slice() {
        [1, rows, cols] => Matrix::from_vec(*rows, *cols, t.data),
        [n] => Matrix::from_vec(1, *n, t.data),
        other => panic!("{key}: unexpected shape {other:?}"),
    }
}

/// How far one activation is from the reference.
struct Gap {
    max_abs: f32,
    /// `max_abs` divided by the reference's largest magnitude. The residual
    /// stream grows through the layers, so an absolute bound that suits
    /// layer 0 is meaningless at layer 27. This is the scale-free figure.
    relative_to_peak: f32,
    cosine: f32,
}

fn gap(ours: &Matrix, reference: &Matrix) -> Gap {
    assert_eq!(ours.shape(), reference.shape(), "shapes must match");
    let peak = reference.data.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let max_abs = ours.max_abs_diff(reference);
    Gap {
        max_abs,
        relative_to_peak: if peak > 0.0 { max_abs / peak } else { max_abs },
        cosine: reference.cosine_similarity(ours),
    }
}

/// The bounds, set from measurements on all eight fixtures (3 to 963 tokens),
/// each about 3x over the worst case seen.
///
/// **Each layer alone** (`each_layer_alone_matches_when_fed_the_reference_input`):
/// 224 checks, worst 3.1e-6 of the peak, at layer 1 on the 963-token fixture.
/// This is the tight bound and the bug detector. A deliberately broken window
/// fails it; see the commit that set it.
const MAX_ALONE_RELATIVE: f32 = 1e-5;

/// **The full stack**, with drift carried up from every layer below. Worst
/// 7.95e-5 of the peak, on the 306-token fixture after layer 19. Layer 19
/// builds a residual value of about -20,800 there and amplifies the small
/// drift it receives; fed PyTorch's own input, the same layer matches to
/// 7e-8. So this bound measures accumulation, not correctness.
///
/// Absolute gaps look alarming in the late layers, up to 2.2 at layer 19, but
/// the residual stream peaks near 28,000, so the bound is relative to the
/// peak and not absolute.
const MAX_STACK_RELATIVE: f32 = 3e-4;

/// `final_norm` gets its own bound, because relative-to-peak means something
/// different after a LayerNorm. A few residual dimensions are huge, they
/// dominate the standard deviation the norm divides by, and so the normed
/// output's peak is small while the error carried in from the layers below is
/// not. Its worst case is 3.45e-5 on the 963-token fixture, with a cosine of
/// 1.000000000 and verbalizer logits off by only 2.1e-5. The logit check
/// below is the real test of this checkpoint; this bound only catches a gross
/// error.
const MAX_FINAL_NORM_RELATIVE: f32 = 1e-4;

const MIN_COSINE: f32 = 1.0 - 1e-6;
/// The verbalizer logits are the accumulated end of the stack, so they get
/// the same 1e-3 the head test uses. PyTorch's own eager and sdpa backends
/// differ by 6.4e-5 there.
const MAX_LOGIT_ABS: f32 = 1e-3;

/// Run one fixture and return what was out of bounds. Empty means it passed.
fn check_fixture(f: &Fixture, verbalizers: &[u32]) -> Vec<String> {
    let m = model();
    let mut failures = Vec::new();

    let final_norm = m
        .encode(&f.token_ids, |name, ours| {
            let reference = reference(f.index, name);
            let g = gap(ours, &reference);
            println!(
                "fixture {} ({:<14}) {name:16} max abs {:.3e}  rel-to-peak {:.3e}  cosine {:.9}",
                f.index, f.name, g.max_abs, g.relative_to_peak, g.cosine
            );
            let bound = if name == "final_norm" {
                MAX_FINAL_NORM_RELATIVE
            } else {
                MAX_STACK_RELATIVE
            };
            if g.relative_to_peak > bound || g.cosine < MIN_COSINE {
                failures.push(format!(
                    "fixture {} ({}) {name}: rel-to-peak {:.3e} (bound {bound:.0e}), cosine \
                     {:.9}",
                    f.index, f.name, g.relative_to_peak, g.cosine
                ));
            }
        })
        .expect("the forward pass must run");

    let (head, decoder) = m.head();
    let ours = head.logits_for_tokens(&final_norm, decoder, f.mask_position, verbalizers);
    let reference = reference(f.index, "logits.verbalizers").data;
    let worst = ours
        .iter()
        .zip(&reference)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    println!(
        "fixture {} ({:<14}) verbalizers      max abs {worst:.3e}  ours {ours:?}",
        f.index, f.name
    );
    if worst > MAX_LOGIT_ABS {
        failures.push(format!(
            "fixture {} ({}) logits.verbalizers: max abs {worst:.3e}",
            f.index, f.name
        ));
    }

    let pick = |v: &[f32]| {
        v.iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).expect("no NaN"))
            .map(|(i, _)| i)
            .expect("four logits")
    };
    if pick(&ours) != pick(&reference) {
        failures.push(format!(
            "fixture {} ({}): the predicted class differs: ours {ours:?}, reference \
             {reference:?}",
            f.index, f.name
        ));
    }

    failures
}

/// Fail with every out-of-bounds checkpoint from every fixture, not just the
/// first fixture to fail. The first entry is the earliest point in the stack,
/// which is where to start looking.
fn assert_no_failures(failures: Vec<String>) {
    assert!(
        failures.is_empty(),
        "{} checkpoints out of bounds:\n  {}",
        failures.len(),
        failures.join("\n  ")
    );
}

/// 3, 119 and 183 tokens. The 119-token one already separates local from
/// global attention, because the two agree only while seq_len <= 65; the
/// 183-token one puts every middle query's window strictly inside the
/// sequence.
#[test]
fn every_layer_matches_on_the_short_fixtures() {
    let (fixtures, verbalizers) = fixtures();
    let mut failures = Vec::new();
    for name in ["near_empty", "shortest_real", "multi_space"] {
        let f = fixtures
            .iter()
            .find(|f| f.name == name)
            .unwrap_or_else(|| panic!("no fixture {name}"));
        failures.extend(check_fixture(f, &verbalizers));
    }
    assert_no_failures(failures);
}

/// The remaining five, up to 963 tokens. Slow on rung 6's naive matmul,
/// so not part of the default run.
#[test]
#[ignore = "minutes on the naive matmul; run with -- --ignored"]
fn every_layer_matches_on_the_long_fixtures() {
    let (fixtures, verbalizers) = fixtures();
    let mut failures = Vec::new();
    for f in fixtures
        .iter()
        .filter(|f| !["near_empty", "shortest_real", "multi_space"].contains(&f.name.as_str()))
    {
        failures.extend(check_fixture(f, &verbalizers));
    }
    assert_no_failures(failures);
}

/// Each layer alone, fed PyTorch's own input from the layer below.
///
/// The full-stack test measures the gap after 28 layers of accumulation, so a
/// large gap there could come from any layer below it, amplified on the way
/// up. Here layer i gets the reference `layer.{i-1}.out` as its input, so a
/// gap belongs to layer i alone. Set `MB_FIXTURE` to a fixture name to run
/// one fixture.
#[test]
#[ignore = "minutes on the naive matmul; run with -- --ignored"]
fn each_layer_alone_matches_when_fed_the_reference_input() {
    let (fixtures, _) = fixtures();
    let only = std::env::var("MB_FIXTURE").ok();
    let m = model();
    let mut failures = Vec::new();

    for f in fixtures
        .iter()
        .filter(|f| only.as_deref().is_none_or(|n| n == f.name))
    {
        let masks = Masks::new(m.config(), f.token_ids.len(), None);
        for i in 0..m.config().num_hidden_layers {
            let input = if i == 0 {
                "embeddings.norm".to_string()
            } else {
                format!("layer.{}.out", i - 1)
            };
            let mut h = reference(f.index, &input);
            m.layer(i, &mut h, &masks);
            let want = reference(f.index, &format!("layer.{i}.out"));
            let g = gap(&h, &want);

            // Where the largest difference sits, and how big the value is
            // there: an error on a huge residual value means something
            // different from one on an ordinary value.
            let mut at = 0;
            let mut worst = 0.0f32;
            for (k, (a, b)) in h.data.iter().zip(&want.data).enumerate() {
                let d = (a - b).abs();
                if d > worst {
                    worst = d;
                    at = k;
                }
            }
            let (token, dim) = (at / want.cols, at % want.cols);
            println!(
                "fixture {} ({:<18}) layer {i:2} alone  max abs {:.3e}  rel-to-peak {:.3e}  \
                 at token {token} dim {dim} (reference value {:.3})",
                f.index, f.name, g.max_abs, g.relative_to_peak, want.data[at]
            );
            if g.relative_to_peak > MAX_ALONE_RELATIVE {
                failures.push(format!(
                    "fixture {} ({}) layer {i} alone: rel-to-peak {:.3e}",
                    f.index, f.name, g.relative_to_peak
                ));
            }
        }
    }
    assert_no_failures(failures);
}
