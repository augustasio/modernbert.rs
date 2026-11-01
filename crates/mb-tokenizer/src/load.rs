//! Reading `tokenizer.json`.
//!
//! The file holds the vocabulary, the merge list, the added tokens and the
//! configuration of each stage. This module reads it and puts it into the
//! shape the encoder wants. It does no encoding itself.
//!
//! Two decisions shape the whole module.
//!
//! **Byte-level strings become raw bytes here.** HF writes vocabulary keys in
//! the GPT-2 printable alphabet, where a space is `Ġ` and a newline is `Ċ`.
//! Decoding those keys back to bytes once, at load time, lets the encoder work
//! on the input bytes directly. The byte offsets from the lexer then index the
//! input with no conversion, and no piece is ever re-encoded.
//!
//! **Merges become pairs of token ids.** HF ranks a merge by its position in
//! the merge list. tiktoken instead ranks by the id of the joined token, which
//! is a different algorithm and can disagree. `ignore_merges` is `false` for
//! this model, so the merge list is what counts, and
//! [`Vocabulary::merge_rank`] is keyed on the two ids being joined.

use std::collections::HashMap;
use std::fmt;
use std::path::Path;

use serde_json::Value;

/// A problem with the file. The message names the key that was wrong, because
/// a `tokenizer.json` is 3.5 MB and "invalid JSON" does not help anyone.
#[derive(Debug)]
pub struct LoadError(String);

impl LoadError {
    pub fn new(message: impl Into<String>) -> Self {
        LoadError(message.into())
    }
}

impl fmt::Display for LoadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for LoadError {}

fn err<T>(message: impl Into<String>) -> Result<T, LoadError> {
    Err(LoadError(message.into()))
}

// ============================================================================
// The GPT-2 byte alphabet
// ============================================================================

/// The 256 printable characters GPT-2 uses to write arbitrary bytes as text.
///
/// Byte values that are already printable ASCII or printable Latin-1 stand for
/// themselves. The remaining 68 bytes, which are control characters, the
/// space, and a few gaps, map to U+0100 and upwards in order. That is why a
/// space appears as `Ġ` (U+0120 = 0x100 + 32) in the vocabulary.
///
/// Returns the byte for each of the 256 characters, as a map. The reverse
/// direction is [`byte_to_char`].
fn char_to_byte() -> HashMap<char, u8> {
    let mut map = HashMap::with_capacity(256);
    let mut next_spare = 0u32;
    for byte in 0u8..=255 {
        let printable = matches!(byte, 0x21..=0x7E | 0xA1..=0xAC | 0xAE..=0xFF);
        let c = if printable {
            // Latin-1 byte values equal their Unicode scalar values.
            char::from_u32(byte as u32).expect("Latin-1 is valid Unicode")
        } else {
            let c = char::from_u32(0x100 + next_spare).expect("below the Unicode maximum");
            next_spare += 1;
            c
        };
        map.insert(c, byte);
    }
    debug_assert_eq!(next_spare, 68, "68 byte values are not printable");
    map
}

/// The character each byte is written as. The inverse of [`char_to_byte`].
fn byte_to_char() -> [char; 256] {
    let mut table = ['\0'; 256];
    for (c, byte) in char_to_byte() {
        table[byte as usize] = c;
    }
    table
}

/// Turn a byte-level key into the bytes it stands for, or `None` if the key
/// holds a character outside the 256-character alphabet.
///
/// `None` is not always an error. `model.vocab` also lists the 23
/// whitespace-run added tokens under their literal content, where a space is
/// a real space rather than `Ġ`. Those entries belong to the added-token
/// namespace, so the byte-level maps must leave them out. See
/// [`Vocabulary::from_json`].
fn try_decode_key(key: &str, map: &HashMap<char, u8>) -> Option<Vec<u8>> {
    let mut bytes = Vec::with_capacity(key.len());
    for c in key.chars() {
        bytes.push(*map.get(&c)?);
    }
    Some(bytes)
}

/// Turn a byte-level key into bytes, where anything else is a broken file.
fn decode_key(key: &str, map: &HashMap<char, u8>) -> Result<Vec<u8>, LoadError> {
    match try_decode_key(key, map) {
        Some(bytes) => Ok(bytes),
        None => err(format!(
            "{key:?} is not written in the 256-character byte-level alphabet"
        )),
    }
}

// ============================================================================
// Added tokens
// ============================================================================

/// A token that is matched in the text before the regex runs, and that never
/// takes part in a merge.
///
/// For this model there are 116 of these. Most are unremarkable, but two
/// groups change the output in ways the regex alone cannot explain:
///
/// * 23 whitespace runs, for runs of 2 to 24 spaces. They are the reason
///   `"  A"` encodes as `[50276, 329]` where the plain GPT-2 regex would give
///   `[209, 329]`.
/// * `[MASK]`, whose `lstrip` flag eats the whitespace in front of it, so
///   `"x  [MASK]"` has no whitespace token at all.
#[derive(Debug, Clone)]
pub struct AddedToken {
    pub id: u32,
    pub content: String,
    /// Eat whitespace to the left of the match.
    pub lstrip: bool,
    /// Eat whitespace to the right of the match.
    pub rstrip: bool,
    /// Whether the normalizer may change this token's text.
    pub normalized: bool,
    /// Whether this is a special token, such as `[CLS]`.
    pub special: bool,
}

// ============================================================================
// The loaded file
// ============================================================================

/// Everything `tokenizer.json` says, in the shape the encoder wants.
pub struct Vocabulary {
    /// The bytes of each token, indexed by id. Ids with no token, which are
    /// the added tokens and the padding at the top of the id space, hold an
    /// empty slice.
    token_bytes: Vec<Vec<u8>>,
    /// The id of each token, keyed on its bytes.
    ids: HashMap<Vec<u8>, u32>,
    /// The rank of each merge, keyed on the two ids it joins. A lower rank
    /// happens first.
    merges: HashMap<(u32, u32), u32>,
    /// The id the two ids of a merge produce.
    merge_result: HashMap<(u32, u32), u32>,
    /// Tokens matched before the regex runs.
    pub added: Vec<AddedToken>,
    /// How many `model.vocab` keys were listed under their literal content
    /// rather than in the byte-level alphabet. These are added tokens written
    /// twice, and they are not in the byte-level maps. 23 for this model.
    pub literal_key_count: usize,
    /// Size of the whole id space, including added tokens and the padding that
    /// rounds it to a multiple of 64. Larger than the number of real tokens.
    pub id_space: u32,
    /// Put a space in front of a piece that does not already start with one.
    pub add_prefix_space: bool,
    /// The name of the normalizer, for example `NFC`.
    pub normalizer: Option<String>,
    /// `true` would make the encoder look a whole piece up in the vocabulary
    /// and skip merging when it hits. It is `false` for this model, so the
    /// merge loop always runs.
    pub ignore_merges: bool,
    pub cls: u32,
    pub sep: u32,
    pub mask: u32,
    pub pad: u32,
    pub unk: u32,
}

impl Vocabulary {
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, LoadError> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path)
            .map_err(|e| LoadError(format!("cannot read {}: {e}", path.display())))?;
        let json: Value = serde_json::from_str(&text)
            .map_err(|e| LoadError(format!("{} is not valid JSON: {e}", path.display())))?;
        Self::from_json(&json)
    }

    pub fn from_json(json: &Value) -> Result<Self, LoadError> {
        let alphabet = char_to_byte();

        // ---- the BPE model -------------------------------------------------
        let model = match json.get("model") {
            Some(m) => m,
            None => return err("tokenizer.json has no \"model\" key"),
        };

        let ignore_merges = model
            .get("ignore_merges")
            .and_then(Value::as_bool)
            .unwrap_or(false);

        let vocab = match model.get("vocab").and_then(Value::as_object) {
            Some(v) => v,
            None => return err("model.vocab is missing or is not an object"),
        };

        let mut ids: HashMap<Vec<u8>, u32> = HashMap::with_capacity(vocab.len());
        let mut highest = 0u32;
        let mut by_id: HashMap<u32, Vec<u8>> = HashMap::with_capacity(vocab.len());

        // Keys that are not byte-level. `model.vocab` lists the 23
        // whitespace-run added tokens twice over: once as a byte-level key,
        // and once under their literal content. Both spellings mean the same
        // bytes, so putting the literal ones in `ids` would overwrite the
        // byte-level entry the merge loop needs. `"ĠĠ"` and `"  "` are both
        // two spaces, and only the first can come out of a merge.
        //
        // These are collected and checked against `added_tokens` below,
        // rather than skipped quietly.
        let mut literal_keys: Vec<(String, u32)> = Vec::new();

        for (key, id) in vocab {
            let id = match id.as_u64() {
                Some(n) => n as u32,
                None => return err(format!("model.vocab[{key:?}] is not a number")),
            };
            highest = highest.max(id);
            match try_decode_key(key, &alphabet) {
                Some(bytes) => {
                    ids.insert(bytes.clone(), id);
                    by_id.insert(id, bytes);
                }
                None => literal_keys.push((key.clone(), id)),
            }
        }

        // ---- the merge list ------------------------------------------------
        //
        // Each entry is a two-element array of byte-level strings. The
        // position in the list is the rank, and a lower rank merges first.
        let merge_list = match model.get("merges").and_then(Value::as_array) {
            Some(m) => m,
            None => return err("model.merges is missing or is not an array"),
        };

        let mut merges = HashMap::with_capacity(merge_list.len());
        let mut merge_result = HashMap::with_capacity(merge_list.len());
        for (rank, entry) in merge_list.iter().enumerate() {
            let (left, right) = match entry.as_array() {
                Some(pair) if pair.len() == 2 => {
                    let left = pair[0].as_str();
                    let right = pair[1].as_str();
                    match (left, right) {
                        (Some(l), Some(r)) => (l, r),
                        _ => return err(format!("model.merges[{rank}] holds non-strings")),
                    }
                }
                // Older files join the pair with a space instead.
                _ => match entry.as_str() {
                    Some(s) => match s.split_once(' ') {
                        Some(pair) => pair,
                        None => return err(format!("model.merges[{rank}] = {s:?} has no space")),
                    },
                    None => {
                        return err(format!(
                            "model.merges[{rank}] is neither a two-element array nor a string"
                        ))
                    }
                },
            };

            let left_bytes = decode_key(left, &alphabet)?;
            let right_bytes = decode_key(right, &alphabet)?;
            let mut joined = left_bytes.clone();
            joined.extend_from_slice(&right_bytes);

            // Every operand and every result of a merge must be a token. If
            // one is missing the merge can never fire, and the file is broken.
            let left_id = match ids.get(&left_bytes) {
                Some(&id) => id,
                None => {
                    return err(format!(
                        "model.merges[{rank}]: {left:?} is not in the vocab"
                    ))
                }
            };
            let right_id = match ids.get(&right_bytes) {
                Some(&id) => id,
                None => {
                    return err(format!(
                        "model.merges[{rank}]: {right:?} is not in the vocab"
                    ))
                }
            };
            let joined_id = match ids.get(&joined) {
                Some(&id) => id,
                None => {
                    return err(format!(
                        "model.merges[{rank}]: the join of {left:?} and {right:?} is not in the \
                         vocab, so the merge could never fire"
                    ))
                }
            };

            // The first entry wins if a pair somehow appears twice, because
            // the lowest rank is the one that fires.
            merges.entry((left_id, right_id)).or_insert(rank as u32);
            merge_result.entry((left_id, right_id)).or_insert(joined_id);
        }

        // ---- added tokens --------------------------------------------------
        let added_list = match json.get("added_tokens").and_then(Value::as_array) {
            Some(a) => a,
            None => return err("added_tokens is missing or is not an array"),
        };

        let mut added = Vec::with_capacity(added_list.len());
        for (i, entry) in added_list.iter().enumerate() {
            let field = |name: &str| entry.get(name);
            let id = match field("id").and_then(Value::as_u64) {
                Some(n) => n as u32,
                None => return err(format!("added_tokens[{i}].id is missing")),
            };
            let content = match field("content").and_then(Value::as_str) {
                Some(s) => s.to_string(),
                None => return err(format!("added_tokens[{i}].content is missing")),
            };
            highest = highest.max(id);
            added.push(AddedToken {
                id,
                content,
                lstrip: field("lstrip").and_then(Value::as_bool).unwrap_or(false),
                rstrip: field("rstrip").and_then(Value::as_bool).unwrap_or(false),
                normalized: field("normalized").and_then(Value::as_bool).unwrap_or(true),
                special: field("special").and_then(Value::as_bool).unwrap_or(false),
            });
        }

        // The encoder must try the longest content first. `"[MASK]"` and a run
        // of 24 spaces must both win over any shorter token that starts the
        // same way, so sort once here instead of at every call.
        added.sort_by(|a, b| {
            b.content
                .len()
                .cmp(&a.content.len())
                .then_with(|| a.id.cmp(&b.id))
        });

        // Every non-byte-level vocabulary key must be an added token with the
        // same id. If one is not, the two namespaces disagree and the file is
        // not what this loader assumes.
        for (key, id) in &literal_keys {
            match added.iter().find(|t| &t.content == key) {
                Some(token) if token.id == *id => {}
                Some(token) => {
                    return err(format!(
                        "model.vocab[{key:?}] has id {id}, but the added token with the same \
                         content has id {}",
                        token.id
                    ))
                }
                None => {
                    return err(format!(
                        "model.vocab[{key:?}] is not byte-level and is not an added token \
                         either, so nothing could ever produce it"
                    ))
                }
            }
        }
        let literal_key_count = literal_keys.len();

        let special_id = |want: &str| -> Result<u32, LoadError> {
            match added.iter().find(|t| t.content == want) {
                Some(t) => Ok(t.id),
                None => err(format!("added_tokens has no {want}")),
            }
        };

        // ---- stage configuration -------------------------------------------
        let add_prefix_space = json
            .get("pre_tokenizer")
            .and_then(|p| p.get("add_prefix_space"))
            .and_then(Value::as_bool)
            .unwrap_or(false);

        let normalizer = json
            .get("normalizer")
            .and_then(|n| n.get("type"))
            .and_then(Value::as_str)
            .map(str::to_string);

        // Fill the byte table. Ids with no token keep an empty slice: the
        // added tokens are not in model.vocab, and the id space is padded to a
        // multiple of 64 so the top ids stand for nothing at all.
        let id_space = highest + 1;
        let mut token_bytes = vec![Vec::new(); id_space as usize];
        for (id, bytes) in by_id {
            token_bytes[id as usize] = bytes;
        }

        Ok(Vocabulary {
            token_bytes,
            ids,
            merges,
            merge_result,
            cls: special_id("[CLS]")?,
            sep: special_id("[SEP]")?,
            mask: special_id("[MASK]")?,
            pad: special_id("[PAD]")?,
            unk: special_id("[UNK]")?,
            added,
            literal_key_count,
            id_space,
            add_prefix_space,
            normalizer,
            ignore_merges,
        })
    }

    /// The id of the token whose bytes are exactly `bytes`.
    pub fn id_of(&self, bytes: &[u8]) -> Option<u32> {
        self.ids.get(bytes).copied()
    }

    /// The bytes of token `id`, or an empty slice for an id that stands for no
    /// token.
    pub fn bytes_of(&self, id: u32) -> &[u8] {
        match self.token_bytes.get(id as usize) {
            Some(bytes) => bytes,
            None => &[],
        }
    }

    /// How early the merge of `left` and `right` happens. Lower goes first.
    /// `None` means the two never merge.
    pub fn merge_rank(&self, left: u32, right: u32) -> Option<u32> {
        self.merges.get(&(left, right)).copied()
    }

    /// The id that joining `left` and `right` produces.
    pub fn merge_to(&self, left: u32, right: u32) -> Option<u32> {
        self.merge_result.get(&(left, right)).copied()
    }

    /// The added token with this id, if any. 116 entries, so a scan is
    /// cheaper than a map and easier to read.
    pub fn added_token(&self, id: u32) -> Option<&AddedToken> {
        self.added.iter().find(|t| t.id == id)
    }

    /// The number of byte-level tokens: the ones a merge can produce. Smaller
    /// than [`Self::id_space`], and smaller than the number of `model.vocab`
    /// keys by [`Self::literal_key_count`].
    pub fn len(&self) -> usize {
        self.ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    /// The number of merges.
    pub fn merge_count(&self) -> usize {
        self.merges.len()
    }

    /// Write `bytes` in the GPT-2 printable alphabet. This is only for error
    /// messages and test output, where raw bytes are unreadable.
    pub fn show_bytes(bytes: &[u8]) -> String {
        let table = byte_to_char();
        bytes.iter().map(|&b| table[b as usize]).collect()
    }
}
