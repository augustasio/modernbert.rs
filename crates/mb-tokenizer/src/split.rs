//! GPT-2 regex pre-tokenization, written as a forward-scanning lexer.
//!
//! Before BPE merging, HF `tokenizers` cuts the text into pieces with one
//! hardcoded pattern. This is the pattern, confirmed byte-identical in
//! `tokenizers` 0.15 through 1.0.0-rc.3:
//!
//! ```text
//! 's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
//! ```
//!
//! A regex engine must backtrack for the `(?!\S)` lookahead. The tiktoken
//! maintainers say in their own source that this pattern controls most of the
//! tokenization time. This module gives the same output in one forward scan.
//!
//! [`split`] returns `(start, end)` byte offsets. The caller does not copy.
//!
//! The lexer tries the alternatives in the order of the pattern. Regex
//! alternation goes left to right, and the first match wins. A different order
//! gives different pieces.

use std::sync::LazyLock;

use unicode_general_category::{get_general_category, GeneralCategory};

/// The pattern above. Tests compile this with `fancy-regex` and compare the
/// regex pieces against the lexer pieces.
pub const GPT2_PATTERN: &str =
    r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+";

// ============================================================================
// Character classes
// ============================================================================
//
// The pattern uses three Unicode classes:
//
//   \p{L} = Lu | Ll | Lt | Lm | Lo     any letter
//   \p{N} = Nd | Nl | No               any number
//   \s                                 any whitespace
//
// Do not use `char::is_alphabetic()` for `\p{L}`. That is the `Alphabetic`
// property, which is a different set, and the token ids then differ from HF.
//
// The category tables must also be exact. The `unicode-properties` crate
// reports a letter or a number for 17,480 code points that `\p{L}` and
// `\p{N}` reject, because its ranges include the unassigned gaps inside
// letter blocks. Real text holds no unassigned code points, so a corpus test
// passes and the bug waits. `unicode-general-category` agrees with the regex
// on all 1,114,112 code points, and `exhaustive_code_points_match_the_oracle`
// in the tests proves it.
//
// One byte holds the classes of one code point. A lookup is an array index and
// a bit test. The cost is the same for ASCII and for CJK.

const LETTER: u8 = 1 << 0;
const NUMBER: u8 = 1 << 1;
const WHITESPACE: u8 = 1 << 2;

/// Classes of the 128 ASCII code points. The compiler builds this table, so
/// the common path costs nothing at run time.
const ASCII_CLASS: [u8; 128] = {
    let mut table = [0u8; 128];
    let mut i = 0;
    while i < 128 {
        let b = i as u8;
        let mut class = 0u8;
        // These are the `u8` methods, which test A-Z, a-z and 0-9 exactly.
        // They are not `char::is_alphabetic()`, which tests a wider set.
        if b.is_ascii_alphabetic() {
            class |= LETTER;
        }
        if b.is_ascii_digit() {
            class |= NUMBER;
        }
        if matches!(b, b' ' | b'\t' | b'\n' | b'\r' | 0x0b | 0x0c) {
            class |= WHITESPACE;
        }
        table[i] = class;
        i += 1;
    }
    table
};

const UNICODE_TABLE_SIZE: usize = 0x11_0000;

/// Classes of the code points above ASCII. The program builds this table on
/// the first call to [`split`]. It costs about 1.1 MB and about 50 ms once.
///
/// Surrogates and unassigned code points keep the value 0. A `&str` cannot
/// hold them, so nothing reads those entries.
static UNICODE_CLASS: LazyLock<Box<[u8]>> = LazyLock::new(|| {
    let mut table = vec![0u8; UNICODE_TABLE_SIZE].into_boxed_slice();
    for i in 128..UNICODE_TABLE_SIZE {
        if let Some(c) = char::from_u32(i as u32) {
            table[i] = class_from_category(c);
        }
    }
    table
});

fn class_from_category(c: char) -> u8 {
    let mut class = 0u8;
    match get_general_category(c) {
        GeneralCategory::UppercaseLetter
        | GeneralCategory::LowercaseLetter
        | GeneralCategory::TitlecaseLetter
        | GeneralCategory::ModifierLetter
        | GeneralCategory::OtherLetter => class |= LETTER,
        GeneralCategory::DecimalNumber
        | GeneralCategory::LetterNumber
        | GeneralCategory::OtherNumber => class |= NUMBER,
        _ => {}
    }
    if c.is_whitespace() {
        class |= WHITESPACE;
    }
    class
}

#[inline(always)]
fn class_of(c: char) -> u8 {
    let i = c as u32 as usize;
    if i < 128 {
        ASCII_CLASS[i]
    } else {
        UNICODE_CLASS[i]
    }
}

#[inline]
fn is_letter(c: char) -> bool {
    class_of(c) & LETTER != 0
}

#[inline]
fn is_number(c: char) -> bool {
    class_of(c) & NUMBER != 0
}

#[inline]
fn is_whitespace(c: char) -> bool {
    class_of(c) & WHITESPACE != 0
}

/// True for the class `[^\s\p{L}\p{N}]`: punctuation, symbols and marks.
#[inline]
fn is_other(c: char) -> bool {
    class_of(c) & (WHITESPACE | LETTER | NUMBER) == 0
}

// ============================================================================
// One function for each alternative
// ============================================================================
//
// Each function returns `Some(end)` for a match that starts at `start`, or
// `None`. The span is `[start, end)`, and `end` is always greater than
// `start`. The lexer never makes an empty piece.

/// `'s|'t|'re|'ve|'m|'ll|'d`
///
/// These are case-sensitive. The cl100k and o200k patterns wrap the same list
/// in `(?i:...)`, but the GPT-2 pattern does not, so `'S` does not match here.
fn try_contraction(input: &str, start: usize) -> Option<usize> {
    const CONTRACTIONS: [&str; 7] = ["'s", "'t", "'re", "'ve", "'m", "'ll", "'d"];
    let rest = input.get(start..)?;
    for tail in CONTRACTIONS {
        if rest.starts_with(tail) {
            return Some(start + tail.len());
        }
    }
    None
}

/// ` ?<class>+` for the three classes: `\p{L}`, `\p{N}` and `[^\s\p{L}\p{N}]`.
///
/// The leading space is optional, so the regex tries the space first and then
/// retries without it. This loop does the same. The retry matters for a piece
/// such as `" ."`: with the space the body starts at `.`, and without the
/// space the body starts at the space itself, which belongs to no class here.
fn try_space_run(input: &str, start: usize, in_class: fn(char) -> bool) -> Option<usize> {
    for take_space in [true, false] {
        let body_start = if take_space {
            if input.as_bytes().get(start) != Some(&b' ') {
                continue;
            }
            start + 1
        } else {
            start
        };
        let mut end = body_start;
        for c in input[body_start..].chars() {
            if !in_class(c) {
                break;
            }
            end += c.len_utf8();
        }
        if end > body_start {
            return Some(end);
        }
    }
    None
}

/// `\s+(?!\S)`
///
/// This is the alternative that needs backtracking in a regex engine, and it
/// is the reason the lexer exists.
///
/// The run must end at the end of the input, or at another whitespace
/// character. A greedy `\s+` would eat the whole run. The lookahead makes the
/// engine give back characters until the next character is not `\S`. In
/// practice that means: take the whole run if it reaches the end of the input,
/// and otherwise stop one character early and leave the last whitespace
/// character for the next piece. That last character is why `" a"` and
/// `"  a"` tokenize differently.
fn try_whitespace_before_text(input: &str, start: usize) -> Option<usize> {
    let mut end = start;
    let mut last_char_start = start;
    for c in input[start..].chars() {
        if !is_whitespace(c) {
            break;
        }
        last_char_start = end;
        end += c.len_utf8();
    }
    if end == start {
        return None;
    }
    if end == input.len() {
        return Some(end);
    }
    // Non-whitespace follows, so give back the last whitespace character.
    // A run of one character gives back everything, which is no match.
    if last_char_start > start {
        Some(last_char_start)
    } else {
        None
    }
}

/// `\s+`
///
/// The last alternative, and the fallback. Only one case reaches it: a run of
/// exactly one whitespace character with non-whitespace after it, because
/// every other run already matched `\s+(?!\S)` above.
fn try_whitespace_run(input: &str, start: usize) -> Option<usize> {
    let mut end = start;
    for c in input[start..].chars() {
        if !is_whitespace(c) {
            break;
        }
        end += c.len_utf8();
    }
    if end > start {
        Some(end)
    } else {
        None
    }
}

// ============================================================================
// The lexer
// ============================================================================

/// Cut `text` into pre-token spans. The spans are the same as the spans from
/// `fancy_regex::Regex::find_iter` on [`GPT2_PATTERN`].
pub fn split(text: &str) -> Split<'_> {
    Split {
        input: text,
        pos: 0,
    }
}

pub struct Split<'a> {
    input: &'a str,
    pos: usize,
}

impl Iterator for Split<'_> {
    /// `(start, end)` byte offsets into the input.
    type Item = (usize, usize);

    fn next(&mut self) -> Option<(usize, usize)> {
        if self.pos >= self.input.len() {
            return None;
        }
        let start = self.pos;
        let input = self.input;

        // The order is the order of the alternatives in the pattern.
        let end = try_contraction(input, start)
            .or_else(|| try_space_run(input, start, is_letter))
            .or_else(|| try_space_run(input, start, is_number))
            .or_else(|| try_space_run(input, start, is_other))
            .or_else(|| try_whitespace_before_text(input, start))
            .or_else(|| try_whitespace_run(input, start))?;

        debug_assert!(
            end > start,
            "the lexer made no progress at byte {start} of {:?}",
            &input[start..(start + 32).min(input.len())]
        );
        self.pos = end;
        Some((start, end))
    }
}

/// Cut `text` and return the pieces as string slices. This is a convenience
/// for tests and for the merge loop.
pub fn pieces(text: &str) -> impl Iterator<Item = &str> {
    split(text).map(move |(start, end)| &text[start..end])
}
