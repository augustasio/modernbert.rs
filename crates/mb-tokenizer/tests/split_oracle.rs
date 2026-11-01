//! Tier-1 verification for the pre-tokenizer, against a regex oracle.
//!
//! The lexer must give the same spans as a real regex engine running
//! [`GPT2_PATTERN`]. These tests keep lexer bugs apart from merge bugs: a
//! failure here is a splitter bug, and nothing downstream needs looking at.
//!
//! The oracle is `fancy-regex`. HF runs the same pattern through `onig`.
//! The two engines agree on this pattern for every input tested here, and the
//! golden AG News test against real HF token ids is the final authority.

use std::sync::LazyLock;

use fancy_regex::Regex;
use mb_tokenizer::split::{pieces, GPT2_PATTERN};

static ORACLE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(GPT2_PATTERN).expect("the pattern must compile"));

/// Run both the oracle and the lexer on `text` and compare the pieces.
fn assert_same_pieces(text: &str, label: &str) {
    let expected: Vec<&str> = ORACLE
        .find_iter(text)
        .map(|m| m.expect("the oracle must not fail").as_str())
        .collect();
    let actual: Vec<&str> = pieces(text).collect();

    if expected != actual {
        // Report the first difference. A whole-list diff is unreadable for
        // long inputs.
        let at = expected
            .iter()
            .zip(&actual)
            .position(|(e, a)| e != a)
            .unwrap_or(expected.len().min(actual.len()));
        panic!(
            "pieces differ for {label} on input {text:?}\n\
             first difference at piece {at}\n\
             oracle: {:?}\n\
             lexer:  {:?}\n\
             oracle piece count {}, lexer piece count {}",
            expected.get(at),
            actual.get(at),
            expected.len(),
            actual.len(),
        );
    }
}

/// The pieces must cover the whole input, in order, with no gap and no
/// overlap. This holds for this pattern because the last alternative `\s+`
/// matches any character that the others reject, and every character is
/// either whitespace, a letter, a number, or in `[^\s\p{L}\p{N}]`.
fn assert_pieces_cover_input(text: &str) {
    let joined: String = pieces(text).collect();
    assert_eq!(joined, text, "the pieces must join back into the input");
}

/// Inputs chosen to hit the parts of the pattern that are easy to get wrong.
const TRICKY: &[(&str, &str)] = &[
    ("", "empty"),
    (" ", "one space"),
    ("  ", "two spaces"),
    ("   ", "three spaces"),
    ("\t", "one tab"),
    ("\n", "one newline"),
    ("\r\n", "CRLF"),
    ("\n\n\n", "newline run"),
    ("a", "one letter"),
    (" a", "space then letter"),
    ("  a", "two spaces then letter, the lookahead case"),
    ("   a", "three spaces then letter"),
    ("a ", "letter then trailing space"),
    ("a  ", "letter then two trailing spaces"),
    ("\tA", "tab then capital, asymmetric with a bare tab"),
    ("a\n\nb", "letters around a newline run"),
    ("a \n b", "mixed whitespace between letters"),
    ("Hello world", "two words"),
    ("Hello  world", "two words, two spaces"),
    // Contractions. The GPT-2 pattern is case-sensitive here, unlike cl100k.
    ("don't", "lowercase contraction"),
    (
        "DON'T",
        "uppercase contraction, must not match the contraction rule",
    ),
    ("it's", "'s"),
    ("I'm", "'m"),
    ("we're", "'re"),
    ("we've", "'ve"),
    ("we'll", "'ll"),
    ("I'd", "'d"),
    ("'s", "a bare contraction at the start"),
    ("x'y", "an apostrophe that is not a contraction"),
    ("''''", "apostrophe run"),
    // Numbers. GPT-2 keeps a whole digit run; cl100k would cut it into
    // groups of three.
    ("1", "one digit"),
    ("1234567890", "ten digits, must stay one piece"),
    (" 1234", "space then digits"),
    ("12.34", "digits around a dot"),
    ("1,234,567", "grouped digits"),
    ("a1b2", "letters and digits alternating"),
    // Punctuation and symbols.
    (".", "one dot"),
    (" .", "space then dot"),
    ("...", "dot run"),
    ("!?!?", "punctuation run"),
    (" -> ", "an arrow between spaces"),
    ("(a)", "brackets"),
    ("$100", "currency then digits"),
    ("a/b", "slash, which cl100k treats differently"),
    ("#[derive(Debug)]", "rust attribute"),
    // Non-ASCII.
    ("café", "precomposed accent"),
    (
        "cafe\u{301}",
        "combining accent, a different class from is_alphabetic",
    ),
    ("naïve", "diaeresis"),
    ("Ünicode", "leading uppercase non-ASCII"),
    ("Привет мир", "cyrillic"),
    ("γειά σου", "greek"),
    ("שלום", "hebrew, right to left"),
    ("مرحبا", "arabic"),
    ("日本語", "CJK, category Lo"),
    ("こんにちは", "hiragana"),
    ("한국어", "hangul"),
    ("中文 and English", "mixed scripts"),
    ("🎉", "emoji, category So"),
    ("👨‍👩‍👧‍👦", "emoji with zero width joiners"),
    ("🇬🇧", "regional indicators"),
    ("\u{00a0}", "non-breaking space"),
    ("a\u{00a0}b", "non-breaking space between letters"),
    ("\u{3000}", "ideographic space"),
    ("a\u{2009}b", "thin space"),
    ("\u{200b}", "zero width space, not whitespace in Unicode"),
    ("½", "vulgar fraction, category No"),
    ("Ⅻ", "roman numeral, category Nl"),
    ("٣٤٥", "arabic-indic digits, category Nd"),
    // Whitespace against text, the lookahead in anger.
    ("a  \t\n  b", "mixed whitespace run between letters"),
    ("  \t\n  ", "mixed whitespace run alone"),
    ("a   ", "trailing whitespace run"),
    ("   a   b   ", "runs on both sides and between"),
];

#[test]
fn tricky_inputs_match_the_oracle() {
    for (text, label) in TRICKY {
        assert_same_pieces(text, label);
        assert_pieces_cover_input(text);
    }
}

/// Contexts that put a code point next to each class boundary in the pattern.
const CONTEXTS: &[(&str, &str)] = &[
    ("{}", "alone"),
    (" {}", "after a space"),
    ("a{}", "after a letter"),
    ("1{}", "after a digit"),
    (".{}", "after punctuation"),
    ("{}a", "before a letter"),
    ("{} ", "before a space"),
    ("{}\n", "before a newline"),
    (" {} ", "between spaces"),
    ("a{}b", "between letters"),
];

fn check_code_point(c: char) {
    let s = c.to_string();
    for (shape, label) in CONTEXTS {
        let text = shape.replace("{}", &s);
        assert_same_pieces(&text, &format!("U+{:04X} {label}", c as u32));
        assert_pieces_cover_input(&text);
    }
}

/// Ranges that cover every general category the pattern cares about, plus the
/// boundaries between them. This runs on every `cargo test`.
#[test]
fn sampled_code_points_match_the_oracle() {
    const RANGES: &[(u32, u32, &str)] = &[
        (0x0000, 0x017F, "ASCII, Latin-1 and Latin Extended-A"),
        (0x0300, 0x036F, "combining diacritical marks"),
        (0x0370, 0x03FF, "Greek"),
        (0x0400, 0x04FF, "Cyrillic"),
        (0x0590, 0x05FF, "Hebrew"),
        (0x0600, 0x06FF, "Arabic"),
        (0x0660, 0x0669, "Arabic-Indic digits"),
        (0x0900, 0x097F, "Devanagari"),
        (
            0x2000,
            0x206F,
            "general punctuation, including exotic spaces",
        ),
        (0x2070, 0x209F, "super and subscripts"),
        (0x2150, 0x218F, "number forms, categories No and Nl"),
        (0x2190, 0x21FF, "arrows"),
        (
            0x3000,
            0x303F,
            "CJK symbols, including the ideographic space",
        ),
        (0x3040, 0x30FF, "Hiragana and Katakana"),
        (0x4E00, 0x4F00, "CJK unified ideographs, a sample"),
        (0xAC00, 0xAD00, "Hangul syllables, a sample"),
        (0xFB00, 0xFB4F, "alphabetic presentation forms"),
        (0xFE00, 0xFE0F, "variation selectors"),
        (0xFF00, 0xFFEF, "halfwidth and fullwidth forms"),
        (0x1D400, 0x1D4FF, "mathematical alphanumeric symbols"),
        (0x1F300, 0x1F5FF, "emoji and pictographs"),
        (0x1F600, 0x1F64F, "emoticons"),
        (0x1F1E6, 0x1F1FF, "regional indicators"),
        (0xE0000, 0xE007F, "tags"),
    ];

    for (lo, hi, _name) in RANGES {
        for cp in *lo..=*hi {
            if let Some(c) = char::from_u32(cp) {
                check_code_point(c);
            }
        }
    }
}

/// Every code point in Unicode, in every context. About 11 million oracle
/// calls, which take about 11 seconds in release and some minutes in debug.
/// That is too slow for the normal dev loop but fast enough to run often.
///
/// Run it after any change to the class tables, and after any change to the
/// Unicode crate. It is the test that caught `unicode-properties` reporting
/// 17,480 unassigned code points as letters or numbers:
///
/// `cargo test --release -- --ignored exhaustive`
#[test]
#[ignore = "about 11 s in release; run it after a change to the class tables"]
fn exhaustive_code_points_match_the_oracle() {
    for cp in 0..0x11_0000u32 {
        if let Some(c) = char::from_u32(cp) {
            check_code_point(c);
        }
    }
}

/// Structured text, to catch an ordering bug that single characters cannot.
#[test]
fn documents_match_the_oracle() {
    const DOCUMENTS: &[&str] = &[
        "The quick brown fox jumps over the lazy dog. It's 42 degrees!",
        "Wall St. Bears Claw Back Into the Black (Reuters) Reuters - Short-sellers,\n\
         Wall Street's dwindling band of ultra-cynics, are seeing green again.",
        "fn main() {\n    let x: Vec<u32> = (0..10).map(|i| i * 2).collect();\n\
         \tprintln!(\"{x:?}\");\n}\n",
        "{\"key\": \"value\", \"n\": [1, 2, 3], \"nested\": {\"a\": null}}",
        "https://example.com/path?a=1&b=2#frag",
        "user@example.com, +44 7700 900123",
        "** Instructions **    \nGiven a short news report, determine the category.\n\
         ** Options **\n- Business: It is about business.\n- Tech: It's about Science.\n",
        "Mixed 日本語 and English with 🎉 emoji, café accents, and  double  spaces.",
    ];

    for text in DOCUMENTS {
        assert_same_pieces(text, "document");
        assert_pieces_cover_input(text);
    }
}
