//! Rungs 4 and 5: ModernBERT-Large-Instruct run through candle.
//!
//! The encoder is candle-transformers' own `models::modernbert::ModernBert`,
//! unchanged, because these rungs measure what a framework gives you. A hand
//! port onto candle tensors would measure us again.
//!
//! The head is ours, written with candle ops. candle-transformers has a
//! `ModernBertForMaskedLM`, but it projects every position onto all 50,368
//! vocabulary rows, and its head and decoder fields and loaders are private,
//! so it cannot be asked for less. Rungs 1 and 7 compute the head at the
//! `[MASK]` position only and for the four verbalizer rows only. Doing the same
//! here keeps rungs 4 and 5 different from rung 7 in the framework, not in the
//! work done.
//!
//! Rung 4 is the default build: candle's CPU backend, whose matmul is the
//! `gemm` crate. Rung 5 is the same code built with `--features accelerate`,
//! which routes candle's matmul through Apple Accelerate. That feature is the
//! one variable between the two rungs.
//!
//! The candle-transformers 0.11.0 implementation was checked against the trap
//! list in `PLAN.md` before use. Where it is right, a comment below says why it
//! can be trusted; see `check_attn_norms` for the one place it is fragile.

use std::path::Path;

use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::{layer_norm_no_bias, linear_no_bias, LayerNorm, Linear, Module, VarBuilder};
use candle_transformers::models::modernbert;

/// Which build this is, as the sidecar `.json` and the worker's ready line
/// record it. `rust_rung.py` refuses logits whose backend does not match the
/// rung, so a build without the feature cannot pass as rung 5.
pub fn backend() -> &'static str {
    if cfg!(feature = "accelerate") {
        "candle-accelerate"
    } else {
        "candle-cpu"
    }
}

/// The ladder rung this build is.
pub fn variant() -> &'static str {
    if cfg!(feature = "accelerate") {
        "rung5"
    } else {
        "rung4"
    }
}

/// Cap candle's CPU threads at `threads`.
///
/// candle sizes its parallelism from environment variables, read when first
/// needed: `RAYON_NUM_THREADS` sizes rayon's global pool and candle's own
/// pool, and is also the thread count candle passes to `gemm` for every matmul
/// (`candle_core::utils::get_num_threads`). `CANDLE_NUM_THREADS` sizes the
/// barrier pool used by quantized and flash-attention kernels, which this
/// model does not call, but it is set too so no pool escapes the cap. The
/// Accelerate build's sgemm is capped separately by `VECLIB_MAXIMUM_THREADS`,
/// which the harness sets.
///
/// Call this first in `main`, before any tensor exists: once a pool is built
/// it keeps its size.
pub fn cap_threads(threads: usize) {
    std::env::set_var("RAYON_NUM_THREADS", threads.to_string());
    std::env::set_var("CANDLE_NUM_THREADS", threads.to_string());
}

/// The threads candle will use for a matmul, as candle itself computes it.
pub fn candle_threads() -> usize {
    candle_core::utils::get_num_threads()
}

/// The MLM head, at one position and for a few vocabulary rows.
///
/// ```text
/// logits = decoder(head.norm(gelu_erf(head.dense(h))))
/// ```
struct MaskedLmHead {
    dense: Linear,
    norm: LayerNorm,
    /// `tok_embeddings.weight` rows for the verbalizers, shape `[k, 1024]`.
    decoder_rows: Tensor,
    /// `decoder.bias` entries for the verbalizers, shape `[k]`.
    decoder_bias: Tensor,
}

impl MaskedLmHead {
    fn load(vb: &VarBuilder, config: &modernbert::Config, ids: &[u32]) -> Result<Self, String> {
        let h = config.hidden_size;
        // Head tensors are not prefixed with `model.`, unlike the encoder's.
        // `head.dense` has no bias (classifier_bias: false).
        let dense = linear_no_bias(h, h, vb.pp("head.dense"))
            .map_err(|e| format!("head.dense.weight, expected [{h}, {h}]: {e}"))?;
        // LayerNorm, not RMSNorm: candle's `layer_norm_no_bias` sets
        // remove_mean, so the mean is subtracted. `layer_norm_eps` equals
        // norm_eps here; `check_config` asserts it.
        let norm = layer_norm_no_bias(h, config.layer_norm_eps, vb.pp("head.norm"))
            .map_err(|e| format!("head.norm.weight, expected [{h}]: {e}"))?;

        // The decoder is tied: `decoder.weight` is absent from the checkpoint
        // and is `tok_embeddings.weight`. Load it, keep only the verbalizer
        // rows, and let the full matrix drop.
        let v = config.vocab_size;
        let index = Tensor::new(ids, &Device::Cpu).map_err(|e| e.to_string())?;
        let embeddings = vb
            .get((v, h), "model.embeddings.tok_embeddings.weight")
            .map_err(|e| {
                format!("model.embeddings.tok_embeddings.weight, expected [{v}, {h}]: {e}")
            })?;
        let decoder_rows = embeddings
            .index_select(&index, 0)
            .map_err(|e| format!("verbalizer rows of tok_embeddings: {e}"))?;
        // `decoder.bias` is the only bias in the file, and it decides the
        // argmax. Dropping it shifts every logit.
        let decoder_bias = vb
            .get(v, "decoder.bias")
            .and_then(|b| b.index_select(&index, 0))
            .map_err(|e| format!("decoder.bias, expected [{v}]: {e}"))?;

        Ok(Self {
            dense,
            norm,
            decoder_rows,
            decoder_bias,
        })
    }

    /// `hidden` is one position, shape `[1, 1024]`. Returns `[k]` logits.
    fn forward(&self, hidden: &Tensor) -> candle_core::Result<Tensor> {
        // Exact erf GELU. candle's bare `gelu()` is the tanh approximation,
        // which is wrong for ModernBERT.
        let x = self.dense.forward(hidden)?.gelu_erf()?;
        let x = self.norm.forward(&x)?;
        let logits = x.matmul(&self.decoder_rows.t()?)?.squeeze(0)?;
        logits + &self.decoder_bias
    }
}

/// candle-transformers' encoder plus our mask-only head.
pub struct CandleModernBert {
    encoder: modernbert::ModernBert,
    head: MaskedLmHead,
}

impl CandleModernBert {
    /// Load from a checkpoint directory holding `config.json` and
    /// `model.safetensors`, for the given verbalizer ids.
    ///
    /// Hard rule 7: the checkpoint is bf16 and every rung upcasts it to f32
    /// on load. The VarBuilder's dtype makes candle convert each tensor as it
    /// is read; bf16 to f32 is exact, so this is the same upcast as every
    /// other rung.
    pub fn load(dir: &str, verbalizer_ids: &[u32]) -> Result<Self, String> {
        let dir = Path::new(dir);
        let config_path = dir.join("config.json");
        let config_text = std::fs::read_to_string(&config_path)
            .map_err(|e| format!("cannot read {}: {e}", config_path.display()))?;
        let config: modernbert::Config = serde_json::from_str(&config_text)
            .map_err(|e| format!("{}: {e}", config_path.display()))?;
        check_config(&config_text, &config)?;

        // Read the file into memory rather than mmap it, because candle's
        // mmap constructor is `unsafe` and CLAUDE.md allows `unsafe` only at
        // the Accelerate FFI boundary. The bf16 buffer drops with `vb`.
        let weights_path = dir.join("model.safetensors");
        let bytes = std::fs::read(&weights_path)
            .map_err(|e| format!("cannot read {}: {e}", weights_path.display()))?;
        let vb = VarBuilder::from_buffered_safetensors(bytes, DType::F32, &Device::Cpu)
            .map_err(|e| format!("{}: {e}", weights_path.display()))?;
        check_attn_norms(&vb, config.num_hidden_layers)?;

        let encoder = modernbert::ModernBert::load(vb.clone(), &config)
            .map_err(|e| format!("encoder: {e}"))?;
        let head = MaskedLmHead::load(&vb, &config, verbalizer_ids)?;
        Ok(Self { encoder, head })
    }

    /// The verbalizer logits at position `mask`, in the order the ids were
    /// given at load time.
    pub fn logits_for_tokens(&self, ids: &[u32], mask: usize) -> Result<Vec<f32>, String> {
        if mask >= ids.len() {
            return Err(format!("mask position {mask} is past {} tokens", ids.len()));
        }
        let n = ids.len();
        let run = || -> candle_core::Result<Vec<f32>> {
            let input = Tensor::new(ids, &Device::Cpu)?.unsqueeze(0)?;
            // One unpadded example: every position is a real token, so the
            // padding mask is all ones and only the local window masks.
            let attention_mask = Tensor::ones((1, n), DType::U32, &Device::Cpu)?;
            let hidden = self.encoder.forward(&input, &attention_mask)?;
            let at_mask = hidden.i((.., mask, ..))?;
            self.head.forward(&at_mask)?.to_vec1()
        };
        run().map_err(|e| format!("forward pass over {n} tokens: {e}"))
    }
}

/// Refuse a config that candle-transformers would read differently from the
/// reference.
///
/// candle's `Config` reads `layer_norm_eps` for every LayerNorm, but the
/// reference ignores that field in favour of `norm_eps`. They are equal in this
/// checkpoint (1e-5). If they ever differ, candle computes a different model,
/// so stop rather than run it.
fn check_config(config_text: &str, config: &modernbert::Config) -> Result<(), String> {
    let raw: serde_json::Value =
        serde_json::from_str(config_text).map_err(|e| format!("config.json: {e}"))?;
    let norm_eps = raw["norm_eps"]
        .as_f64()
        .ok_or("config.json has no numeric norm_eps")?;
    if norm_eps != config.layer_norm_eps {
        return Err(format!(
            "norm_eps is {norm_eps} but layer_norm_eps is {}; candle uses the latter",
            config.layer_norm_eps
        ));
    }
    Ok(())
}

/// Layer 0 has no `attn_norm`; layers 1 to 27 do.
///
/// candle-transformers loads `attn_norm` with `.ok()`
/// (`models/modernbert.rs`, `ModernBertLayer::load`), so any load error, not
/// only an absent tensor, silently becomes "no norm". That is right for this
/// checkpoint only if the tensors are where we expect, so check that here.
fn check_attn_norms(vb: &VarBuilder, layers: usize) -> Result<(), String> {
    for layer in 0..layers {
        let name = format!("model.layers.{layer}.attn_norm.weight");
        let present = vb.contains_tensor(&name);
        if present != (layer != 0) {
            return Err(format!(
                "{name} is {}, expected it only for layers 1 to {}",
                if present { "present" } else { "absent" },
                layers - 1
            ));
        }
    }
    Ok(())
}
