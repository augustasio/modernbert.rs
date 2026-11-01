//! The checkpoint must hold exactly the tensors the forward pass expects.
//!
//! This is a structural gate, not a numerical one. It catches a renamed
//! tensor, a transposed shape, or a missing weight before any of that can
//! turn into a confusing activation mismatch.
//!
//! **Nothing here loads the whole checkpoint.** `Header` reads names, shapes
//! and dtypes without any values, and `Checkpoint::load_one` seeks to a
//! single tensor. That matters more than it looks: `cargo test` runs its test
//! functions in parallel across every core, so a helper that loaded the whole
//! file would ask for 1.58 GB per test and ten of them at once. An earlier
//! version of this file did exactly that and took the machine down with it.

use std::path::PathBuf;

use mb_model::config::Config;
use mb_model::safetensors::{Checkpoint, Header, TensorInfo};

fn weights_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../weights/ModernBERT-Large-Instruct")
}

fn safetensors_path() -> PathBuf {
    let path = weights_dir().join("model.safetensors");
    assert!(
        path.is_file(),
        "{} is missing. Fetch the checkpoint first.",
        path.display()
    );
    path
}

fn config() -> Config {
    Config::from_file(weights_dir().join("config.json")).expect("config must load")
}

/// Names, shapes and dtypes only. Costs the first 18 KB of the file.
fn header() -> Header {
    Header::from_file(safetensors_path()).expect("the header must parse")
}

#[test]
fn the_config_matches_the_published_model() {
    let c = config();
    assert_eq!(c.num_hidden_layers, 28);
    assert_eq!(c.hidden_size, 1024);
    assert_eq!(c.intermediate_size, 2624);
    assert_eq!(c.num_attention_heads, 16);
    assert_eq!(c.head_dim(), 64);
    assert_eq!(c.vocab_size, 50368);
    assert_eq!(c.max_position_embeddings, 8192);
    assert_eq!(c.norm_eps, 1e-5);

    // `intermediate_size` is not 4 * hidden_size and not a multiple of 1024,
    // which is worth pinning so nobody "corrects" it.
    assert_ne!(c.intermediate_size, 4 * c.hidden_size);

    // No biases except the decoder's.
    assert!(!c.norm_bias);
    assert!(!c.attention_bias);
    assert!(!c.mlp_bias);
    assert!(c.decoder_bias);
}

#[test]
fn the_global_layer_schedule_starts_at_zero() {
    let c = config();
    let global: Vec<usize> = (0..c.num_hidden_layers)
        .filter(|&i| c.is_global_layer(i))
        .collect();
    assert_eq!(global, vec![0, 3, 6, 9, 12, 15, 18, 21, 24, 27]);
    assert_eq!(global.len(), 10, "global layers");
    assert_eq!(c.num_hidden_layers - global.len(), 18, "local layers");

    // Both the first and the last layer are global. Reading "every third
    // layer" as 2, 5, 8 is the mistake this guards.
    assert!(c.is_global_layer(0));
    assert!(c.is_global_layer(c.num_hidden_layers - 1));
    assert!(!c.is_global_layer(1));
}

#[test]
fn the_local_window_is_sixty_four_each_side_inclusive() {
    let c = config();
    assert_eq!(c.local_attention, 128, "the total window");
    assert_eq!(c.local_window_each_side(), 64);

    // 129, not 128: the distance test is inclusive. Current transformers
    // stores local_attention / 2 + 1 and its flash-attention path subtracts
    // the 1 back out, so the +1 must not be copied.
    assert_eq!(c.local_window_keys(), 129);

    assert!(c.in_local_window(100, 100 + 64));
    assert!(c.in_local_window(100, 100 - 64));
    assert!(!c.in_local_window(100, 100 + 65));
    assert_eq!(
        (0..300).filter(|&k| c.in_local_window(150, k)).count(),
        129,
        "a query in the middle of a long sequence"
    );
}

#[test]
fn the_rope_bases_are_paired_counterintuitively() {
    let c = config();
    assert_eq!(c.global_rope_theta, 160000.0);
    assert_eq!(c.local_rope_theta, 10000.0);

    // The small-window local layers get the SMALL base and the global layers
    // the large one.
    assert_eq!(c.rope_theta(0), 160000.0, "layer 0 is global");
    assert_eq!(c.rope_theta(1), 10000.0, "layer 1 is local");
    assert!(c.rope_theta(0) > c.rope_theta(1));
}

/// Every tensor the forward pass will ask for, with its shape.
fn expected_inventory(c: &Config) -> Vec<(String, Vec<usize>)> {
    let mut expected: Vec<(String, Vec<usize>)> = vec![
        (
            "model.embeddings.tok_embeddings.weight".into(),
            vec![c.vocab_size, c.hidden_size],
        ),
        ("model.embeddings.norm.weight".into(), vec![c.hidden_size]),
        ("model.final_norm.weight".into(), vec![c.hidden_size]),
        // The head tensors carry no `model.` prefix, unlike the encoder's.
        (
            "head.dense.weight".into(),
            vec![c.hidden_size, c.hidden_size],
        ),
        ("head.norm.weight".into(), vec![c.hidden_size]),
        ("decoder.bias".into(), vec![c.vocab_size]),
    ];

    for n in 0..c.num_hidden_layers {
        // Layer 0 has NO attn_norm: the reference sets it to nn.Identity, and
        // the tensor is absent. A loader that defaults a missing norm weight
        // to ones would normalize twice.
        if n != 0 {
            expected.push((
                format!("model.layers.{n}.attn_norm.weight"),
                vec![c.hidden_size],
            ));
        }
        expected.push((
            format!("model.layers.{n}.attn.Wqkv.weight"),
            vec![3 * c.hidden_size, c.hidden_size],
        ));
        expected.push((
            format!("model.layers.{n}.attn.Wo.weight"),
            vec![c.hidden_size, c.hidden_size],
        ));
        expected.push((
            format!("model.layers.{n}.mlp_norm.weight"),
            vec![c.hidden_size],
        ));
        // 2 * intermediate_size, the GLU doubling.
        expected.push((
            format!("model.layers.{n}.mlp.Wi.weight"),
            vec![2 * c.intermediate_size, c.hidden_size],
        ));
        expected.push((
            format!("model.layers.{n}.mlp.Wo.weight"),
            vec![c.hidden_size, c.intermediate_size],
        ));
    }
    expected
}

#[test]
fn the_checkpoint_holds_exactly_the_expected_tensors() {
    let c = config();
    let h = header();
    let expected = expected_inventory(&c);

    // 6 + 28 * 6 - 1 = 173.
    assert_eq!(expected.len(), 173, "the expected inventory");
    assert_eq!(h.len(), 173, "tensors in the file");

    for (name, shape) in &expected {
        h.require(name, shape)
            .unwrap_or_else(|e| panic!("{name}: {e}"));
    }

    // Nothing unexpected either, so a renamed tensor cannot slip past.
    let mut unexpected: Vec<&str> = h
        .names()
        .filter(|n| !expected.iter().any(|(e, _)| e == n))
        .collect();
    unexpected.sort_unstable();
    assert!(unexpected.is_empty(), "unexpected tensors: {unexpected:?}");
}

#[test]
fn every_tensor_is_bf16() {
    let h = header();
    let wrong: Vec<&str> = h
        .names()
        .filter(|n| h.get(n).map(|i| i.dtype.as_str()) != Some("BF16"))
        .collect();
    assert!(
        wrong.is_empty(),
        "the checkpoint should be entirely BF16; these are not: {wrong:?}"
    );
}

#[test]
fn layer_zero_has_no_attention_norm() {
    let h = header();
    assert!(
        h.get("model.layers.0.attn_norm.weight").is_none(),
        "layer 0's attn_norm must be absent, so the forward pass skips it"
    );
    assert!(
        h.get("model.layers.1.attn_norm.weight").is_some(),
        "every other layer has one"
    );
}

#[test]
fn the_decoder_weight_is_tied_and_only_its_bias_is_stored() {
    let c = config();
    let h = header();

    // Tied to the embeddings, so it is not in the file.
    assert!(
        h.get("decoder.weight").is_none(),
        "decoder.weight is tied to tok_embeddings and must not be stored"
    );

    // And the bias is the ONLY bias in the whole file. Dropping it shifts
    // every MLM logit and changes the argmax, which is the entire inference
    // protocol.
    h.require("decoder.bias", &[c.vocab_size])
        .expect("decoder.bias must exist");
    let biases: Vec<&str> = h.names().filter(|n| n.ends_with(".bias")).collect();
    assert_eq!(biases, vec!["decoder.bias"], "the only bias in the file");

    // head.dense has no bias, unlike BERT's equivalent layer.
    assert!(h.get("head.dense.bias").is_none());
}

#[test]
fn the_parameter_count_is_what_the_hub_reports() {
    let h = header();

    // The figure the Hub reports. It matching confirms both the tied decoder,
    // since the embeddings are counted once, and the single missing
    // attn_norm.
    assert_eq!(h.parameter_count(), 395_881_664, "parameters");
}

#[test]
fn the_byte_ranges_tile_the_payload_without_gaps() {
    let h = header();
    let mut ranges: Vec<&TensorInfo> = h.names().filter_map(|n| h.get(n)).collect();
    ranges.sort_by_key(|i| i.start);

    let mut cursor = 0usize;
    for info in &ranges {
        assert_eq!(
            info.start, cursor,
            "a gap or overlap before the tensor at {}..{}",
            info.start, info.end
        );
        // bf16 is two bytes per value.
        assert_eq!(
            info.end - info.start,
            info.count() * 2,
            "the byte range does not match the shape {:?}",
            info.shape
        );
        cursor = info.end;
    }
    assert_eq!(cursor, 395_881_664 * 2, "the payload is bf16 throughout");
}

#[test]
fn the_bf16_upcast_is_exact() {
    // One tensor, read by seeking to its range: 100 KB rather than 791 MB.
    let t =
        Checkpoint::load_one(safetensors_path(), "decoder.bias").expect("decoder.bias must load");
    assert_eq!(t.shape, vec![50368]);

    // bf16 is the top 16 bits of an f32, so every upcast value must have a
    // zero low mantissa. A rounding or truncating conversion would not.
    for (i, &v) in t.data.iter().enumerate() {
        assert_eq!(
            v.to_bits() & 0xFFFF,
            0,
            "decoder.bias[{i}] = {v} has low mantissa bits set, so the upcast was not a \
             plain 16-bit shift"
        );
    }

    // And it must not be all zeros, or the "only bias in the file" would not
    // matter and this test would prove nothing.
    assert!(
        t.data.iter().any(|&v| v != 0.0),
        "decoder.bias is entirely zero, which would make it harmless to drop"
    );
}

#[test]
fn loading_one_tensor_agrees_with_the_header() {
    let path = safetensors_path();
    let h = header();
    let info = h
        .require("model.embeddings.norm.weight", &[1024])
        .expect("the norm weight");

    let t = Checkpoint::load_one(&path, "model.embeddings.norm.weight").expect("must load");
    assert_eq!(t.shape, info.shape);
    assert_eq!(t.len(), info.count());
}

#[test]
fn a_missing_tensor_names_itself() {
    let h = header();
    let message = h
        .require("model.layers.0.attn_norm.weight", &[1024])
        .expect_err("layer 0 has no attn_norm")
        .to_string();
    assert!(
        message.contains("attn_norm"),
        "the error must name the tensor, got {message:?}"
    );
}

#[test]
fn a_wrong_shape_names_both_shapes() {
    let h = header();
    let message = h
        .require("decoder.bias", &[1024])
        .expect_err("decoder.bias is 50368 long")
        .to_string();
    assert!(
        message.contains("50368") && message.contains("1024"),
        "the error must name both shapes, got {message:?}"
    );
}
