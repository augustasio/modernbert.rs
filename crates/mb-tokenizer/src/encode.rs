//! Turning text into token ids.
//!
//! Five stages, in this order:
//!
//! 1. NFC normalization of the whole input.
//! 2. Cut out the added tokens. These are matched as literal text, never
//!    merged, and `[MASK]` eats the whitespace in front of it.
//! 3. For each surviving span, put a space in front if it has none.
//! 4. Split the span with the GPT-2 lexer and merge each piece by pair rank.
//! 5. Wrap the result in `[CLS]` and `[SEP]`.
//!
//! Stages 1 and 2 commute for this tokenizer, which is why stage 1 can run
//! once over the whole input instead of once per span. See
//! [`Tokenizer::segments`].

use crate::load::{AddedToken, LoadError, Vocabulary};
use crate::split::split;

use std::path::Path;

use unicode_normalization::{is_nfc_quick, IsNormalized, UnicodeNormalization};

/// A stretch of the input, after the added tokens have been cut out.
#[derive(Debug, PartialEq, Eq)]
enum Segment<'a> {
    /// An added token, which goes straight to its id with no merging.
    Added(u32),
    /// Ordinary text, for stages 3 and 4.
    Text(&'a str),
}

pub struct Tokenizer {
    vocab: Vocabulary,
    /// The added tokens with `normalized: false`: the seven specials. Cut out
    /// of the text first. See [`Tokenizer::segments`].
    specials: Vec<AddedToken>,
    /// The other 109 added tokens, cut out second.
    others: Vec<AddedToken>,
    /// The id of each one-byte token, or `u32::MAX` for the 13 byte values
    /// valid UTF-8 cannot hold. Having a token for every reachable byte is
    /// what lets any input encode without an unknown token.
    byte_ids: [u32; 256],
}

impl Tokenizer {
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, LoadError> {
        Self::new(Vocabulary::from_file(path)?)
    }

    pub fn new(vocab: Vocabulary) -> Result<Self, LoadError> {
        // Stage 2 matches added tokens against text that stage 1 has already
        // normalized. That is only the same as HF's order, which normalizes
        // each span after cutting, because every added token is ASCII: NFC
        // never turns non-ASCII into ASCII or the other way round, so it can
        // neither create nor destroy a match. Check the premise rather than
        // trust it, because a different tokenizer.json could break it.
        for token in &vocab.added {
            if !token.content.is_ascii() {
                return Err(LoadError::new(format!(
                    "the added token {:?} (id {}) is not ASCII. The encoder normalizes \
                     before cutting out added tokens, which is only safe while they are \
                     all ASCII. Normalize each span after cutting instead.",
                    token.content, token.id
                )));
            }
        }

        // Byte-level BPE gives a token to every byte the input can hold, so
        // nothing ever falls back to an unknown token. "Can hold" is the
        // point: the input is a `&str`, so 13 byte values never reach here
        // and the file rightly has no token for them. Expect 243.
        let mut byte_ids = [u32::MAX; 256];
        let mut found = 0usize;
        for byte in 0u8..=255 {
            match vocab.id_of(&[byte]) {
                Some(id) => {
                    byte_ids[byte as usize] = id;
                    found += 1;
                }
                None if never_in_utf8(byte) => {}
                None => {
                    return Err(LoadError::new(format!(
                        "byte 0x{byte:02X} can appear in valid UTF-8 but has no one-byte \
                         token, so some inputs could not be encoded at all"
                    )))
                }
            }
        }
        if found != 243 {
            return Err(LoadError::new(format!(
                "expected 243 one-byte tokens, the number of byte values valid UTF-8 can \
                 hold, but found {found}"
            )));
        }

        // `vocab.added` is already sorted longest first, and `filter` keeps
        // that order, so both lists match the longest token at a position.
        let specials = vocab
            .added
            .iter()
            .filter(|t| !t.normalized)
            .cloned()
            .collect();
        let others = vocab
            .added
            .iter()
            .filter(|t| t.normalized)
            .cloned()
            .collect();

        Ok(Tokenizer {
            vocab,
            specials,
            others,
            byte_ids,
        })
    }

    pub fn vocab(&self) -> &Vocabulary {
        &self.vocab
    }

    /// Encode `text`, wrapped in `[CLS]` and `[SEP]`.
    ///
    /// This is what the model is fed. The template comes from the file's
    /// post-processor, which is `[CLS] $A [SEP]` for a single sequence.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut ids = Vec::with_capacity(text.len() / 3 + 2);
        ids.push(self.vocab.cls);
        self.encode_into(text, &mut ids);
        ids.push(self.vocab.sep);
        ids
    }

    /// Encode `text` with no `[CLS]` or `[SEP]`. Useful for testing the
    /// earlier stages on their own.
    pub fn encode_without_specials(&self, text: &str) -> Vec<u32> {
        let mut ids = Vec::with_capacity(text.len() / 3);
        self.encode_into(text, &mut ids);
        ids
    }

    fn encode_into(&self, text: &str, ids: &mut Vec<u32>) {
        // Stage 1. `is_nfc_quick` answers without allocating for the common
        // case, which is text that is already normalized.
        let normalized: String;
        let text = if is_nfc_quick(text.chars()) == IsNormalized::Yes {
            text
        } else {
            normalized = text.nfc().collect();
            &normalized
        };

        for segment in self.segments(text) {
            match segment {
                Segment::Added(id) => ids.push(id),
                Segment::Text(span) => self.encode_text(span, ids),
            }
        }
    }

    /// Stage 2: cut the added tokens out of `text`.
    ///
    /// Two passes, and the order is not an implementation detail. The added
    /// tokens with `normalized: false`, which are the seven specials, are cut
    /// out first. The other 109 are cut out of what is left.
    ///
    /// `"x  [MASK]"` is the case that forces this. `[MASK]` has `lstrip`, so
    /// it must eat the two spaces in front of it and the result is
    /// `[" x", [MASK]]`. In one pass the run of two spaces is also an added
    /// token starting at the same place, it matches first, and the output
    /// gains a stray token 50276 that HF does not produce. Cutting the
    /// specials first gives `[MASK]` the chance to claim that whitespace.
    fn segments<'a>(&self, text: &'a str) -> Vec<Segment<'a>> {
        let mut out = Vec::new();
        for segment in self.cut(text, &self.specials) {
            match segment {
                Segment::Added(id) => out.push(Segment::Added(id)),
                Segment::Text(part) => out.extend(self.cut(part, &self.others)),
            }
        }
        out
    }

    /// Cut `tokens` out of `text`, leaving the rest as `Segment::Text`.
    ///
    /// `tokens` is sorted longest first, so the first match at a position is
    /// the longest one. That matters: a run of 24 spaces must win over the
    /// run of 2 that also starts there.
    fn cut<'a>(&self, text: &'a str, tokens: &[AddedToken]) -> Vec<Segment<'a>> {
        let mut out = Vec::new();
        // Where the current run of ordinary text began.
        let mut text_start = 0usize;
        let mut pos = 0usize;

        while pos < text.len() {
            let Some(token) = tokens.iter().find(|t| text[pos..].starts_with(&t.content)) else {
                // No added token here. Advance one character; `pos` must stay
                // on a character boundary for the slicing above.
                pos += next_char_len(text, pos);
                continue;
            };

            // Close the text run that ends here. `lstrip` eats the whitespace
            // at its end, which is why "x  [MASK]" holds no whitespace token:
            // the two spaces vanish rather than becoming token 50276.
            let mut text_end = pos;
            if token.lstrip {
                text_end = text[text_start..text_end].trim_end().len() + text_start;
            }
            if text_end > text_start {
                out.push(Segment::Text(&text[text_start..text_end]));
            }

            out.push(Segment::Added(token.id));
            pos += token.content.len();

            // `rstrip` eats the whitespace that follows the token.
            if token.rstrip {
                let rest = &text[pos..];
                pos += rest.len() - rest.trim_start().len();
            }
            text_start = pos;
        }

        if text_start < text.len() {
            out.push(Segment::Text(&text[text_start..]));
        }
        out
    }

    /// Stages 3 and 4 for one span of ordinary text.
    fn encode_text(&self, span: &str, ids: &mut Vec<u32>) {
        // Stage 3. `add_prefix_space` is true for this model, and it only
        // acts when the span does not already start with a space. It runs per
        // span, not once per input, which is why the "b" in "a  b" comes out
        // as " b" even though it sits in the middle of the text.
        let prefixed: String;
        let span = if self.vocab.add_prefix_space && !span.starts_with(' ') {
            prefixed = format!(" {span}");
            &prefixed
        } else {
            span
        };

        // Stage 4.
        let mut symbols = Vec::new();
        for (start, end) in split(span) {
            self.merge_piece(&span.as_bytes()[start..end], &mut symbols);
            ids.extend_from_slice(&symbols);
        }
    }

    /// The BPE merge loop for one piece, writing the result into `out`.
    ///
    /// Start with one token per byte, then join the adjacent pair with the
    /// lowest merge rank, over and over, until no adjacent pair has a rank.
    ///
    /// Two things here are easy to get wrong.
    ///
    /// The rank is the pair's position in `model.merges`. tiktoken instead
    /// ranks by the id of the joined token, which is a different algorithm
    /// and can pick a different pair. Using the wrong one gives plausible
    /// tokens that do not match HF.
    ///
    /// There is no shortcut for a piece that is already a token. HF only
    /// takes that shortcut when `ignore_merges` is true, and it is false
    /// here, so the loop always runs. Adding the shortcut would again give
    /// tiktoken's answer rather than HF's.
    fn merge_piece(&self, piece: &[u8], out: &mut Vec<u32>) {
        out.clear();
        for &byte in piece {
            let id = self.byte_ids[byte as usize];
            debug_assert_ne!(
                id,
                u32::MAX,
                "byte 0x{byte:02X} has no token, but it reached the merge loop, so it came \
                 from something that was not valid UTF-8"
            );
            out.push(id);
        }

        // A piece is a word, so this stays short and a plain scan beats a
        // heap. The loop runs once per merge, and each scan is one pass.
        while out.len() > 1 {
            let mut best: Option<(usize, u32)> = None;
            for i in 0..out.len() - 1 {
                if let Some(rank) = self.vocab.merge_rank(out[i], out[i + 1]) {
                    if best.is_none_or(|(_, best_rank)| rank < best_rank) {
                        best = Some((i, rank));
                    }
                }
            }
            let Some((i, _)) = best else { break };

            // `merge_to` is present whenever `merge_rank` is: the loader
            // fills both from the same entry and rejects a merge whose result
            // is not a token.
            let joined = match self.vocab.merge_to(out[i], out[i + 1]) {
                Some(id) => id,
                None => break,
            };
            out[i] = joined;
            out.remove(i + 1);
        }
    }

    /// Turn ids back into text.
    ///
    /// The file's decoder has `add_prefix_space` false while the
    /// pre-tokenizer has it true, so a round trip puts a space at the front:
    /// `decode(encode("Hello"))` is `" Hello"`. That is HF's behaviour, not a
    /// bug here.
    /// Decoding is lossy in a second way too. The prefix space goes on to
    /// every span, not once per input, so each added-token boundary gains
    /// one: `"a  b"` comes back as `" a   b"`, three spaces for the original
    /// two. And `lstrip` loses whitespace outright, so `"x  [MASK]"` comes
    /// back as `" x[MASK]"`.
    pub fn decode(&self, ids: &[u32]) -> String {
        self.decode_inner(ids, false)
    }

    /// Decode, leaving out the tokens marked `special` in the file. This is
    /// HF's `skip_special_tokens=True`.
    ///
    /// Note that it drops every special, not only `[CLS]` and `[SEP]`, so
    /// `[MASK]` disappears as well.
    pub fn decode_without_specials(&self, ids: &[u32]) -> String {
        self.decode_inner(ids, true)
    }

    fn decode_inner(&self, ids: &[u32], skip_specials: bool) -> String {
        let mut bytes = Vec::new();
        for &id in ids {
            match self.vocab.added_token(id) {
                Some(token) => {
                    // An added token stands for its literal content. Whether
                    // it is written out is the caller's choice, and it is a
                    // choice rather than a rule: `[CLS]` typed into the input
                    // is an ordinary token and must come back.
                    if skip_specials && token.special {
                        continue;
                    }
                    bytes.extend_from_slice(token.content.as_bytes());
                }
                None => bytes.extend_from_slice(self.vocab.bytes_of(id)),
            }
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }
}

/// The byte length of the character starting at `pos`.
fn next_char_len(text: &str, pos: usize) -> usize {
    match text[pos..].chars().next() {
        Some(c) => c.len_utf8(),
        None => 1,
    }
}

/// The byte values valid UTF-8 can never hold: 0xC0 and 0xC1 are overlong
/// lead bytes, and 0xF5 upwards would encode above U+10FFFF. `tokenizer.json`
/// has no one-byte token for these 13, which is correct, because the encoder
/// is only ever given a `&str`.
fn never_in_utf8(byte: u8) -> bool {
    matches!(byte, 0xC0 | 0xC1 | 0xF5..=0xFF)
}
