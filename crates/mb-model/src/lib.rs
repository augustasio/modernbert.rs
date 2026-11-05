//! ModernBERT-Large-Instruct forward pass.
//!
//! 28 layers, hidden 1024, intermediate 2624, 16 heads of 64. Attention
//! alternates global (`layer_id % 3 == 0`, so layer 0 is global) and local
//! (128 total window, 64 each side, inclusive, giving 129 keys per query).
//!
//! ```text
//! h = embeddings.norm(tok_embeddings[ids])
//! for each layer:
//!     h = h + Attn(attn_norm(h))      // attn_norm absent for layer 0
//!     h = h + MLP(mlp_norm(h))
//! h = final_norm(h)
//! logits = decoder(head.norm(gelu_erf(head.dense(h))))   // mask position only
//! ```
//!
//! See the trap list in `PLAN.md` and `docs/research/architecture.md`. Every
//! trap there gets a comment at the line where it would otherwise be
//! silently violated.

#[cfg(feature = "accelerate")]
pub mod accelerate;
pub mod attention;
pub mod config;
pub mod embeddings;
pub mod expf;
pub mod f32math;
pub mod fused;
pub mod head;
pub mod mlp;
pub mod model;
pub mod ops;
pub mod rope;
pub mod safetensors;
pub mod tensor;
