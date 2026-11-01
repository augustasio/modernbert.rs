//! The encoder must give the same token ids as HF `tokenizers`.
//!
//! `tests/data/golden_strings.json` holds the ids HF produces for 92 inputs
//! chosen to hit the stages that are easy to get wrong: `add_prefix_space`,
//! the whitespace-run added tokens, `[MASK]`'s `lstrip`, NFC, contractions,
//! digit runs, and the real AG News prompt. Regenerate it with
//! `tools/make_golden_strings.py`.
//!
//! This is the first of the two equality gates. The second is every AG News
//! test example. A failure here names the behaviour that broke, which a
//! failure on example 4,312 of AG News does not.

use std::path::PathBuf;

use mb_tokenizer::encode::Tokenizer;
use serde_json::Value;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .expect("the crate lives inside the repository")
}

fn tokenizer() -> Tokenizer {
    let path = repo_root().join("weights/ModernBERT-Large-Instruct/tokenizer.json");
    assert!(
        path.is_file(),
        "{} is missing. Fetch the checkpoint first.",
        path.display()
    );
    Tokenizer::from_file(path).expect("the file must load")
}

struct Case {
    label: String,
    text: String,
    ids: Vec<u32>,
    ids_no_specials: Vec<u32>,
    decoded: String,
}

fn golden() -> Vec<Case> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data/golden_strings.json");
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
    let json: Value = serde_json::from_str(&text).expect("the golden file must be valid JSON");

    let numbers = |entry: &Value, key: &str| -> Vec<u32> {
        entry[key]
            .as_array()
            .unwrap_or_else(|| panic!("{key} is not an array"))
            .iter()
            .map(|n| n.as_u64().expect("an id must be a number") as u32)
            .collect()
    };

    json["cases"]
        .as_array()
        .expect("cases must be an array")
        .iter()
        .map(|entry| Case {
            label: entry["label"].as_str().unwrap_or_default().to_string(),
            text: entry["text"].as_str().unwrap_or_default().to_string(),
            ids: numbers(entry, "ids"),
            ids_no_specials: numbers(entry, "ids_no_specials"),
            decoded: entry["decoded"].as_str().unwrap_or_default().to_string(),
        })
        .collect()
}

/// Report a mismatch with the first id that differs, which is usually enough
/// to name the stage at fault.
fn compare(label: &str, text: &str, want: &[u32], got: &[u32], what: &str) -> Option<String> {
    if want == got {
        return None;
    }
    let at = want
        .iter()
        .zip(got)
        .position(|(w, g)| w != g)
        .unwrap_or(want.len().min(got.len()));
    Some(format!(
        "{label} ({what})\n  input:  {text:?}\n  HF:     {want:?}\n  ours:   {got:?}\n  \
         first difference at index {at}: HF {:?}, ours {:?}",
        want.get(at),
        got.get(at),
    ))
}

#[test]
fn every_golden_case_matches_hf() {
    let tok = tokenizer();
    let cases = golden();
    assert!(cases.len() >= 90, "the golden file looks truncated");

    let mut failures = Vec::new();
    for case in &cases {
        // Without the specials first: a failure there is the pre-tokenizer,
        // the added tokens or the merges. With them, it is the post-processor.
        if let Some(message) = compare(
            &case.label,
            &case.text,
            &case.ids_no_specials,
            &tok.encode_without_specials(&case.text),
            "no specials",
        ) {
            failures.push(message);
            continue;
        }
        if let Some(message) = compare(
            &case.label,
            &case.text,
            &case.ids,
            &tok.encode(&case.text),
            "with specials",
        ) {
            failures.push(message);
        }
    }

    assert!(
        failures.is_empty(),
        "{} of {} golden cases differ from HF:\n\n{}",
        failures.len(),
        cases.len(),
        failures.join("\n\n")
    );
}

#[test]
fn the_post_processor_wraps_the_sequence() {
    let tok = tokenizer();
    let v = tok.vocab();
    for case in golden() {
        let ids = tok.encode(&case.text);
        assert_eq!(ids.first(), Some(&v.cls), "{}: first id", case.label);
        assert_eq!(ids.last(), Some(&v.sep), "{}: last id", case.label);
        assert_eq!(
            &ids[1..ids.len() - 1],
            case.ids_no_specials.as_slice(),
            "{}: the body must be the same with and without specials",
            case.label
        );
    }
}

/// Decoding must match HF's decoding, which is not the same as recovering the
/// input.
///
/// `add_prefix_space` is true for the pre-tokenizer and false for the
/// decoder, so decoding is lossy. It is lossy in more than the obvious way:
/// the prefix space goes on to every span, not once per input, so each
/// added-token boundary gains a space too. `"a  b"` decodes to `" a   b"`,
/// three spaces for the original two. `"x  [MASK]"` decodes to `" x[MASK]"`,
/// losing the whitespace entirely to `lstrip`.
///
/// So the oracle is HF's own output, not the input text. An earlier version
/// of this test asserted a round trip and failed on correct behaviour.
#[test]
fn decoding_matches_hf() {
    let tok = tokenizer();
    let mut failures = Vec::new();
    for case in golden() {
        let got = tok.decode(&tok.encode_without_specials(&case.text));
        if got != case.decoded {
            failures.push(format!(
                "{}\n  input: {:?}\n  HF:    {:?}\n  ours:  {:?}",
                case.label, case.text, case.decoded, got
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "{} cases decode differently from HF:\n\n{}",
        failures.len(),
        failures.join("\n\n")
    );
}

/// `decode` keeps the specials and `decode_without_specials` drops them.
///
/// Skipping them is a choice, not a rule: a `[CLS]` typed into the input is
/// an ordinary token and `decode` must give it back. Filtering by id instead
/// of by the caller's wish ate those, and this test is what noticed.
#[test]
fn skipping_specials_is_the_callers_choice() {
    let tok = tokenizer();
    for case in golden() {
        let ids = tok.encode(&case.text);
        let kept = tok.decode(&ids);
        let dropped = tok.decode_without_specials(&ids);

        // `encode` wraps every input, so the kept form always shows the wrap
        // and the dropped form never does.
        assert!(
            kept.starts_with("[CLS]") && kept.ends_with("[SEP]"),
            "{}: decode must show the wrap, got {kept:?}",
            case.label
        );
        assert!(
            !dropped.contains("[CLS]") && !dropped.contains("[SEP]"),
            "{}: decode_without_specials must not, got {dropped:?}",
            case.label
        );

        // A `[CLS]` the input itself asked for survives the skipping form
        // only as the text it was, which is to say not at all. What matters
        // is that the non-skipping form keeps it.
        if case.text.contains("[CLS]") {
            assert_eq!(
                kept.matches("[CLS]").count(),
                2,
                "{}: the wrap and the typed one, got {kept:?}",
                case.label
            );
        }
    }
}

#[test]
fn ids_stay_inside_the_id_space() {
    let tok = tokenizer();
    for case in golden() {
        for id in tok.encode(&case.text) {
            assert!(
                id < tok.vocab().id_space,
                "{}: id {id} is outside the id space of {}",
                case.label,
                tok.vocab().id_space
            );
        }
    }
}
