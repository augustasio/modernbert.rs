//! The loader must read `tokenizer.json` the way HF reads it.
//!
//! Every number here was measured against live HF `tokenizers` 0.23.3 during
//! the research pass; `docs/research/tokenizer-json.md` records how. They are
//! asserted rather than merely written down, so a parsing mistake fails here
//! instead of turning into a wrong token id later.

use std::path::PathBuf;

use mb_tokenizer::load::Vocabulary;

/// The checkpoint directory, relative to the repository root.
fn tokenizer_json() -> PathBuf {
    let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .expect("the crate lives inside the repository");
    let path = repo_root.join("weights/ModernBERT-Large-Instruct/tokenizer.json");
    assert!(
        path.is_file(),
        "{} is missing.\nFetch the checkpoint first:\n  \
         python/.venv/bin/python -c \"from huggingface_hub import snapshot_download; \
         snapshot_download('answerdotai/ModernBERT-Large-Instruct', \
         local_dir='weights/ModernBERT-Large-Instruct')\"",
        path.display()
    );
    path
}

fn load() -> Vocabulary {
    Vocabulary::from_file(tokenizer_json()).expect("the file must load")
}

#[test]
fn sizes_match_the_file() {
    let v = load();

    // `model.vocab` has 50,280 keys, but 23 of them are the whitespace-run
    // added tokens written under their literal content instead of in the
    // byte-level alphabet. Those belong to the added-token namespace, so the
    // byte-level vocabulary holds 50,257, which is exactly GPT-2's size.
    assert_eq!(v.len(), 50257, "byte-level tokens");
    assert_eq!(
        v.literal_key_count, 23,
        "model.vocab keys written literally"
    );
    assert_eq!(v.len() + v.literal_key_count, 50280, "keys in model.vocab");

    assert_eq!(v.merge_count(), 50009, "merges");
    assert_eq!(v.added.len(), 116, "added tokens");

    // The id space is padded to a multiple of 64, so it is larger than the
    // number of real tokens: 50,280 vocabulary entries plus 83 `[unused*]`
    // plus the five BERT specials gives 50,368.
    assert_eq!(v.id_space, 50368, "the whole id space");
    assert_eq!(
        v.id_space % 64,
        0,
        "the id space is padded to a multiple of 64"
    );
}

#[test]
fn stage_configuration_matches_the_file() {
    let v = load();

    // GPT-2's own tokenizer.json has this false. ModernBERT has it true, and
    // that one flag is why "A" and " A" encode identically.
    assert!(v.add_prefix_space, "pre_tokenizer.add_prefix_space");

    assert_eq!(
        v.normalizer.as_deref(),
        Some("NFC"),
        "the normalizer is not null"
    );

    // True would let the encoder look a whole piece up and skip merging.
    assert!(!v.ignore_merges, "model.ignore_merges");
}

#[test]
fn special_token_ids_match_the_file() {
    let v = load();
    assert_eq!(v.unk, 50280, "[UNK]");
    assert_eq!(v.cls, 50281, "[CLS]");
    assert_eq!(v.sep, 50282, "[SEP]");
    assert_eq!(v.pad, 50283, "[PAD]");
    assert_eq!(v.mask, 50284, "[MASK]");
}

#[test]
fn mask_strips_the_space_before_it() {
    let v = load();
    let mask = v
        .added
        .iter()
        .find(|t| t.content == "[MASK]")
        .expect("[MASK] is an added token");

    // This flag is why "x  [MASK]" holds no whitespace token: the two spaces
    // are eaten, not tokenized.
    assert!(mask.lstrip, "[MASK].lstrip");
    assert!(!mask.rstrip, "[MASK].rstrip");
    assert!(mask.special, "[MASK].special");
    assert!(!mask.normalized, "[MASK].normalized");
}

#[test]
fn the_whitespace_run_tokens_are_present() {
    let v = load();

    // 23 added tokens cover runs of 2 to 24 spaces, at ids 50276 down to
    // 50254. They are matched before the regex, which is why "  A" gives
    // [50276, 329] where the plain GPT-2 regex would give [209, 329].
    for (spaces, want_id) in [(2u32, 50276u32), (3, 50275), (24, 50254)] {
        let run = " ".repeat(spaces as usize);
        let token = v
            .added
            .iter()
            .find(|t| t.content == run)
            .unwrap_or_else(|| panic!("no added token for a run of {spaces} spaces"));
        assert_eq!(token.id, want_id, "the run of {spaces} spaces");
    }

    let runs = v
        .added
        .iter()
        .filter(|t| !t.content.is_empty() && t.content.bytes().all(|b| b == b' '))
        .count();
    assert_eq!(runs, 23, "whitespace run tokens");
}

#[test]
fn the_space_prefixed_letters_have_the_expected_ids() {
    let v = load();

    // The answer-option logits are read at these ids. Resolving them by
    // encoding is what the encoder must do; this test pins the vocabulary
    // entries the encoder has to reach.
    for (bytes, want) in [
        (&b" A"[..], 329u32),
        (&b" B"[..], 378),
        (&b" C"[..], 330),
        (&b" D"[..], 399),
    ] {
        assert_eq!(
            v.id_of(bytes),
            Some(want),
            "{:?}",
            Vocabulary::show_bytes(bytes)
        );
    }

    // The bare letters exist, but encoding a letter on its own can never
    // reach them, because add_prefix_space puts a space in front first. Code
    // that looks up the raw vocabulary string instead of encoding reads a
    // logit the model was never trained to raise.
    for (bytes, want) in [
        (&b"A"[..], 34u32),
        (&b"B"[..], 35),
        (&b"C"[..], 36),
        (&b"D"[..], 37),
    ] {
        assert_eq!(v.id_of(bytes), Some(want), "the bare letter");
    }
}

#[test]
fn the_ag_news_verbalizers_are_single_tokens() {
    let v = load();

    // The whole AG News protocol rests on each class name being one token.
    // `Sci/Tech` is three, which is why the reference harness remaps it to
    // `Tech`.
    for (bytes, want) in [
        (&b" Business"[..], 10518u32),
        (&b" World"[..], 3645),
        (&b" Sports"[..], 15000),
        (&b" Tech"[..], 18447),
    ] {
        assert_eq!(
            v.id_of(bytes),
            Some(want),
            "{:?} must be one token",
            Vocabulary::show_bytes(bytes)
        );
    }

    assert_eq!(v.id_of(b" Sci/Tech"), None, "Sci/Tech is not one token");
}

#[test]
fn ids_and_bytes_agree_in_both_directions() {
    let v = load();
    let mut checked = 0usize;
    for id in 0..v.id_space {
        let bytes = v.bytes_of(id);
        if bytes.is_empty() {
            continue; // an added token, or padding at the top of the id space
        }
        assert_eq!(
            v.id_of(bytes),
            Some(id),
            "id {id} ({:?}) does not round-trip, so two keys decode to the same bytes",
            Vocabulary::show_bytes(bytes)
        );
        checked += 1;
    }
    assert_eq!(checked, v.len(), "every byte-level token has a byte string");
}

#[test]
fn the_two_namespaces_stay_apart() {
    let v = load();

    // Two spaces have two spellings. `"ĠĠ"` is byte-level and comes out of a
    // merge; `"  "` is the added token at id 50276 and is matched before the
    // regex runs. Both mean the same bytes, so the loader must not let the
    // added token overwrite the merge target.
    let from_merge = v.id_of(b"  ").expect("two spaces are a byte-level token");
    assert_ne!(
        from_merge, 50276,
        "the added token must not shadow the merge"
    );

    let space = v.id_of(b" ").expect("one space is a token");
    assert_eq!(
        v.merge_to(space, space),
        Some(from_merge),
        "merging two spaces gives the byte-level token, not the added token"
    );

    // The added token keeps its own id and its literal content.
    let run = v.added_token(50276).expect("50276 is an added token");
    assert_eq!(run.content, "  ");

    // Added-token ids hold no byte-level entry.
    assert!(v.bytes_of(50276).is_empty(), "50276 is not byte-level");
    assert!(v.bytes_of(v.mask).is_empty(), "[MASK] is not byte-level");
}

#[test]
fn every_merge_names_tokens_that_exist() {
    let v = load();

    // `from_json` rejects a merge whose operands or result are missing, so
    // loading at all proves this. Check a merge that must exist, to be sure
    // the table is populated rather than empty.
    let g = v.id_of(b" ").expect("a lone space is a token");
    let t = v.id_of(b"t").expect("t is a token");
    assert_eq!(
        v.merge_rank(g, t),
        Some(1),
        "the second merge in the list joins a space and t"
    );
    assert_eq!(v.merge_to(g, t), v.id_of(b" t"), "the join gives ' t'");
}
