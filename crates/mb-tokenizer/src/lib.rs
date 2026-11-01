//! Byte-level BPE tokenizer for ModernBERT, producing token IDs identical to HF
//! `tokenizers` for `answerdotai/ModernBERT-Large-Instruct`.
//!
//! Encoding runs in five stages, in this order. The order matters: stage 2 changes
//! what stage 4 sees, which is why a correct GPT-2 regex alone does not reproduce
//! HF's output.
//!
//! 1. NFC normalization.
//! 2. Added-token extraction: 116 entries, including 23 whitespace-run tokens and
//!    `[MASK]`, whose `lstrip` flag swallows preceding whitespace.
//! 3. `add_prefix_space`, applied per surviving span and only when the span does not
//!    already begin with a space.
//! 4. GPT-2 regex pre-tokenization, implemented as a forward-scanning lexer.
//! 5. BPE merges by pair rank, then the `[CLS] $A [SEP]` post-processor.
//!
//! See `docs/research/tokenizer-json.md` for the verified configuration this is
//! written against.
