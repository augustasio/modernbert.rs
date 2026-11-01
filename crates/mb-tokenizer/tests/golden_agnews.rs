//! The encoder must match HF on every AG News test example.
//!
//! This is the project's primary tokenizer gate: 7,600 examples, each checked
//! twice, as raw text and as the prompt the model is actually given. The
//! hand-picked cases in `golden_strings.rs` name the behaviour they test;
//! this one is volume, and covers what the project runs rather than what
//! someone thought to write down.
//!
//! Regenerate with `tools/make_golden_agnews.py`.

use std::io::{BufRead, BufReader};
use std::path::PathBuf;

use flate2::read::GzDecoder;
use mb_tokenizer::encode::Tokenizer;
use serde_json::Value;

fn tokenizer() -> Tokenizer {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../weights/ModernBERT-Large-Instruct/tokenizer.json");
    assert!(
        path.is_file(),
        "{} is missing. Fetch the checkpoint first.",
        path.display()
    );
    Tokenizer::from_file(path).expect("the file must load")
}

struct Example {
    index: usize,
    text: String,
    ids: Vec<u32>,
    prompt_ids: Vec<u32>,
}

/// The template the ids were generated with. The Rust side rebuilds the
/// prompt rather than storing it, since storing 7,600 copies of a fixed
/// template would triple the file for no gain.
///
/// The four trailing spaces after `** Instructions **` are real. They are in
/// the reference harness, confirmed with `cat -vet`, and dropping them
/// changes the token ids.
fn prompt_for(text: &str) -> String {
    format!(
        "** Instructions **    \n\
         Given a short news report, determine which news category it belongs to.\n\
         ** Content **\n\
         What is the category of the following news report?\n\
         News Report: {text}\n\
         ** Options **\n\
         - Business: It is about business or financial news.\n\
         - World: It's about world news.\n\
         - Sports: It's about sports.\n\
         - Tech: It's about Science or Technology, or related companies.\n\
         ** Answer **\n\
         Answer: [unused0] [MASK]"
    )
}

fn examples() -> Vec<Example> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data/golden_agnews.jsonl.gz");
    let file = std::fs::File::open(&path)
        .unwrap_or_else(|e| panic!("cannot open {}: {e}", path.display()));
    let reader = BufReader::new(GzDecoder::new(file));

    let numbers = |entry: &Value, key: &str| -> Vec<u32> {
        entry[key]
            .as_array()
            .unwrap_or_else(|| panic!("{key} is not an array"))
            .iter()
            .map(|n| n.as_u64().expect("an id must be a number") as u32)
            .collect()
    };

    let mut out = Vec::with_capacity(7600);
    for line in reader.lines() {
        let line = line.expect("the gzip stream must decode");
        let entry: Value = serde_json::from_str(&line).expect("each line must be JSON");
        if entry.get("header").is_some() {
            continue;
        }
        out.push(Example {
            index: entry["i"].as_u64().expect("i") as usize,
            text: entry["text"].as_str().expect("text").to_string(),
            ids: numbers(&entry, "ids"),
            prompt_ids: numbers(&entry, "prompt_ids"),
        });
    }
    out
}

/// Describe a mismatch by the first id that differs, and show the input
/// around it, so the failure points at a piece of text rather than an index.
fn describe(example: &Example, what: &str, want: &[u32], got: &[u32]) -> String {
    let at = want
        .iter()
        .zip(got)
        .position(|(w, g)| w != g)
        .unwrap_or(want.len().min(got.len()));
    format!(
        "example {} ({what}): first difference at id {at}, HF {:?}, ours {:?}\n  \
         lengths: HF {}, ours {}\n  text: {:?}",
        example.index,
        want.get(at),
        got.get(at),
        want.len(),
        got.len(),
        example.text,
    )
}

#[test]
fn every_ag_news_example_matches_hf() {
    let tok = tokenizer();
    let examples = examples();
    assert_eq!(examples.len(), 7600, "the AG News test split");

    let mut failures = Vec::new();
    for example in &examples {
        if tok.encode(&example.text) != example.ids {
            failures.push(describe(
                example,
                "raw text",
                &example.ids,
                &tok.encode(&example.text),
            ));
        }
        let prompt = prompt_for(&example.text);
        if tok.encode(&prompt) != example.prompt_ids {
            failures.push(describe(
                example,
                "prompt",
                &example.prompt_ids,
                &tok.encode(&prompt),
            ));
        }
    }

    // Report the first few in full rather than 15,200 lines.
    assert!(
        failures.is_empty(),
        "{} of {} checks differ from HF. The first few:\n\n{}",
        failures.len(),
        examples.len() * 2,
        failures
            .iter()
            .take(5)
            .cloned()
            .collect::<Vec<_>>()
            .join("\n\n")
    );
}

/// The prompts must be long enough to put the sliding window to work.
///
/// ModernBERT's local attention sees 129 keys per query. The AG News *texts*
/// have a median of 52 tokens, which would leave local attention identical to
/// global attention, but the *prompt* around them has a median of 152, and
/// nearly every prompt crosses 129. This test records that, because the
/// numerical fixtures were chosen on the strength of it.
#[test]
fn the_prompts_engage_the_sliding_window() {
    const WINDOW_KEYS: usize = 129;

    let examples = examples();
    let over = examples
        .iter()
        .filter(|e| e.prompt_ids.len() > WINDOW_KEYS)
        .count();
    let fraction = over as f64 / examples.len() as f64;

    assert!(
        fraction > 0.95,
        "only {over} of {} prompts exceed {WINDOW_KEYS} tokens ({:.1}%). If this has \
         dropped, local attention is no longer being exercised by the real workload and \
         the long synthetic fixtures carry the whole burden.",
        examples.len(),
        fraction * 100.0,
    );

    let mut lengths: Vec<usize> = examples.iter().map(|e| e.prompt_ids.len()).collect();
    lengths.sort_unstable();
    let median = lengths[lengths.len() / 2];
    assert!(
        (140..=170).contains(&median),
        "the median prompt is {median} tokens, outside the expected 140 to 170. The \
         template may have changed."
    );
}
