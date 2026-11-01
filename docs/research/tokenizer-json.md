# `answerdotai/ModernBERT-Large-Instruct` tokenizer.json — exact spec

Retrieved 2026-11-01 from `https://huggingface.co/answerdotai/ModernBERT-Large-Instruct/raw/main/`.
Files saved in `/private/tmp/claude-501/-Users-augustasm-Documents-code-personal/74cb3aa9-3b78-400a-8b7a-755389ec9e90/scratchpad/hf_download/`
(`tokenizer.json` 3,583,227 bytes; `tokenizer_config.json`; `special_tokens_map.json`; `config.json`).

Ground truth encodings were produced with real HF `tokenizers` **0.23.3** (pip, throwaway venv at
`/private/tmp/claude-501/-Users-augustasm-Documents-code-personal/74cb3aa9-3b78-400a-8b7a-755389ec9e90/scratchpad/venv`).

Top-level keys: `version`, `truncation`, `padding`, `added_tokens`, `normalizer`, `pre_tokenizer`,
`post_processor`, `decoder`, `model`. `version` = `"1.0"`, `truncation` = `null`, `padding` = `null`.

---

## 1. `pre_tokenizer` (verbatim)

```json
{
  "type": "ByteLevel",
  "add_prefix_space": true,
  "trim_offsets": true,
  "use_regex": true
}
```

A **single `ByteLevel`**, not a `Sequence`.

- `add_prefix_space`: **`true`**  <- differs from GPT-2's own `tokenizer.json`, which has `false`
- `trim_offsets`: `true`
- `use_regex`: `true`

(`tokenizer_config.json` also carries `"add_prefix_space": true`.)

## 2. `normalizer` (verbatim)

```json
{
  "type": "NFC"
}
```

**Not null.** NFC Unicode normalization, no lowercasing, no accent stripping.
Verified live: `"café"` (decomposed) and `"café"` (precomposed) both encode to `[36241]`.

## 3. `model` settings (everything except `vocab`/`merges`)

| key | value |
|---|---|
| `type` | `"BPE"` |
| `dropout` | `null` |
| `unk_token` | `null` |
| `continuing_subword_prefix` | `null` |
| `end_of_word_suffix` | `null` |
| `fuse_unk` | `false` |
| `byte_fallback` | `false` |
| **`ignore_merges`** | **`false`** |

All eight keys are explicitly present in the JSON. `ignore_merges` is **`false`**, so the standard
merge-ranking BPE loop applies: no "if the whole pre-token is already in vocab, emit it directly"
shortcut. Same as vanilla GPT-2.

`len(model.vocab)` = 50280 (ids 0..50279). `len(model.merges)` = 50009.
Merges are stored as **2-element arrays** (`[["Ġ","Ġ"],["Ġ","t"],["Ġ","a"],["h","e"],["i","n"],["r","e"], ...]`),
not space-joined strings. Last three: `[["Ġfort","ified"],["feren","ced"],["ĠOut","comes"]]`.

## 4. `post_processor` (verbatim)

```json
{
  "type": "TemplateProcessing",
  "single": [
    { "SpecialToken": { "id": "[CLS]", "type_id": 0 } },
    { "Sequence":     { "id": "A",     "type_id": 0 } },
    { "SpecialToken": { "id": "[SEP]", "type_id": 0 } }
  ],
  "pair": [
    { "SpecialToken": { "id": "[CLS]", "type_id": 0 } },
    { "Sequence":     { "id": "A",     "type_id": 0 } },
    { "SpecialToken": { "id": "[SEP]", "type_id": 0 } },
    { "Sequence":     { "id": "B",     "type_id": 0 } },
    { "SpecialToken": { "id": "[SEP]", "type_id": 0 } }
  ],
  "special_tokens": {
    "[CLS]":  { "id": "[CLS]",  "ids": [50281], "tokens": ["[CLS]"] },
    "[MASK]": { "id": "[MASK]", "ids": [50284], "tokens": ["[MASK]"] },
    "[PAD]":  { "id": "[PAD]",  "ids": [50283], "tokens": ["[PAD]"] },
    "[SEP]":  { "id": "[SEP]",  "ids": [50282], "tokens": ["[SEP]"] },
    "[UNK]":  { "id": "[UNK]",  "ids": [50280], "tokens": ["[UNK]"] }
  }
}
```

Yes: BERT-style wrapping via `TemplateProcessing`.

- single-sequence template: `[CLS] $A [SEP]`  ->  `50281 … 50282`
- pair template: `[CLS] $A [SEP] $B [SEP]`

Note **`type_id` is 0 for sequence B as well** (not 1). Verified live: `encode("foo","bar")` gives
`type_ids = [0,0,0,0,0]`. `tokenizer_config.json` has
`"model_input_names": ["input_ids", "attention_mask"]` — no `token_type_ids`.

## 5. `decoder` (verbatim)

```json
{
  "type": "ByteLevel",
  "add_prefix_space": false,
  "trim_offsets": true,
  "use_regex": true
}
```

(The decoder's `add_prefix_space` is `false` while the pre-tokenizer's is `true`; decoding is
therefore lossy for strings without a leading space — `decode(encode("Hello world"))` returns
`" Hello world"`. Irrelevant if you only need encode parity.)

## 6. `added_tokens` — 116 entries

Full list, grouped. Every entry has `single_word: false`.

| id(s) | content | lstrip | rstrip | normalized | special |
|---|---|---|---|---|---|
| 0 | `\|\|\|IP_ADDRESS\|\|\|` | false | false | **true** | false |
| 1 | `<\|padding\|>` | false | false | false | **true** |
| 50254 | 24 spaces | false | false | **true** | false |
| 50255 | 23 spaces | false | false | **true** | false |
| 50256 | 22 spaces | false | false | **true** | false |
| 50257 | 21 spaces | false | false | **true** | false |
| 50258 | 20 spaces | false | false | **true** | false |
| 50259 | 19 spaces | false | false | **true** | false |
| 50260 | 18 spaces | false | false | **true** | false |
| 50261 | 17 spaces | false | false | **true** | false |
| 50262 | 16 spaces | false | false | **true** | false |
| 50263 | 15 spaces | false | false | **true** | false |
| 50264 | 14 spaces | false | false | **true** | false |
| 50265 | 13 spaces | false | false | **true** | false |
| 50266 | 12 spaces | false | false | **true** | false |
| 50267 | 11 spaces | false | false | **true** | false |
| 50268 | 10 spaces | false | false | **true** | false |
| 50269 | 9 spaces | false | false | **true** | false |
| 50270 | 8 spaces | false | false | **true** | false |
| 50271 | 7 spaces | false | false | **true** | false |
| 50272 | 6 spaces | false | false | **true** | false |
| 50273 | 5 spaces | false | false | **true** | false |
| 50274 | 4 spaces | false | false | **true** | false |
| 50275 | 3 spaces | false | false | **true** | false |
| 50276 | 2 spaces | false | false | **true** | false |
| 50277 | `\|\|\|EMAIL_ADDRESS\|\|\|` | false | false | **true** | false |
| 50278 | `\|\|\|PHONE_NUMBER\|\|\|` | false | false | **true** | false |
| 50279 | `<\|endoftext\|>` | false | false | false | **true** |
| 50280 | `[UNK]` | false | false | false | **true** |
| 50281 | `[CLS]` | false | false | false | **true** |
| 50282 | `[SEP]` | false | false | false | **true** |
| 50283 | `[PAD]` | false | false | false | **true** |
| 50284 | `[MASK]` | **true** | false | false | **true** |
| 50285..50367 | `[unused0]` .. `[unused82]` (83 tokens, contiguous) | false | false | **true** | false |

Counts: 116 total = 2 (ids 0,1) + 23 whitespace runs (2..24 spaces, ids 50254..50276) +
2 PII placeholders + `<|endoftext|>` + 5 BERT specials + 83 `[unused*]`.

Which ones also live in `model.vocab`: everything **except** `[UNK]`, `[CLS]`, `[SEP]`, `[PAD]`,
`[MASK]` and the 83 `[unused*]` tokens. In particular the 23 whitespace-run tokens and the PII
placeholders **are** in `model.vocab` at those same ids, but they are *also* in `added_tokens`,
which is what matters (see §9).

## 7. Sizes and special-token IDs

| | |
|---|---|
| `get_vocab_size(with_added_tokens=True)` | **50368** |
| `get_vocab_size(with_added_tokens=False)` | 50280 |
| `config.json` `vocab_size` | **50368** |
| id space | contiguous 0..50367 |
| `[UNK]` | **50280** |
| `[CLS]` | **50281** |
| `[SEP]` | **50282** |
| `[PAD]` | **50283** |
| `[MASK]` | **50284** |
| `<|endoftext|>` | 50279 |
| `<|padding|>` | 1 |

`config.json` also: `bos_token_id`/`cls_token_id` = 50281, `eos_token_id`/`sep_token_id` = 50282,
`pad_token_id` = 50283, `model_type` = `modernbert`, `max_position_embeddings` = 8192.

## 8. Token IDs for A/B/C/D

Raw `model.vocab` lookups:

| token | id |
|---|---|
| `A` | 34 |
| `B` | 35 |
| `C` | 36 |
| `D` | 37 |
| `ĠA` | **329** |
| `ĠB` | **378** |
| `ĠC` | **330** |
| `ĠD` | **399** |

Note the non-monotonic ordering of the `Ġ` forms: C (330) < B (378) < D (399).

**What the tokenizer actually emits**, because `add_prefix_space: true` (verified live):

| input | ids (no specials) | ids (with specials) |
|---|---|---|
| `"A"` | `[329]` | `[50281, 329, 50282]` |
| `" A"` | `[329]` | `[50281, 329, 50282]` |
| `"B"` | `[378]` | `[50281, 378, 50282]` |
| `" B"` | `[378]` | `[50281, 378, 50282]` |
| `"C"` | `[330]` | `[50281, 330, 50282]` |
| `" C"` | `[330]` | `[50281, 330, 50282]` |
| `"D"` | `[399]` | `[50281, 399, 50282]` |
| `" D"` | `[399]` | `[50281, 399, 50282]` |

`"A"` and `" A"` are **indistinguishable**. The bare ids 34/35/36/37 are only reachable
mid-string, e.g. `"xA"` -> `[1269, 34]`, `"fooA"` -> `[17374, 34]`.

This matters for ModernBERT-Large-Instruct specifically: its MCQ answer-option logits are
on 329/378/330/399, not 34/35/36/37.

## 9. Is HF's ByteLevel regex the one that applies? Yes.

`use_regex: true`, so the hardcoded GPT-2 pattern is used. Confirmed from the crate source at
several refs:

- **v0.23.3** (latest stable release, 2026-10-09), `tokenizers/src/pre_tokenizers/byte_level.rs:44`:

  ```rust
      SysRegex::new(r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+")
  ```

- v0.21.0 and v0.20.3, same file, **line 44**; v0.19.1 and v0.15.2, same file, **line 41** —
  byte-for-byte identical pattern, inside `lazy_static! { static ref RE: SysRegex = SysRegex::new(...) }`.

- **Current `main` / v1.0.0-rc.3**: the repo has been restructured (the old
  `tokenizers/src/pre_tokenizers/byte_level.rs` path now 404s; crates are `tokenizers/tk-encode`,
  `tokenizers/tk-serialize`, `tokenizers/bitcannon`, ...). The pattern moved to
  `tokenizers/bitcannon/src/regexes.rs:9-10`:

  ```rust
  /// GPT-2 / ByteLevel. Reproduced by [`crate::fsm::fsm_byte_level`].
  pub const GPT2: &str =
      r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+";
  ```

  and is re-exported by `tokenizers/tk-encode/src/utils/byte_level.rs:8` as
  `pub use bitcannon::regexes::GPT2 as GPT2_REGEX_STR;`. On `main` the regex is no longer *run*
  at runtime — it is reproduced by a hand-written FSM (`fsm_byte_level`) with the const kept as the
  tested specification. The pattern string is unchanged.

So the pattern is identical to the one you quoted, in every version checked. **Your existing
verified GPT-2-pattern pre-tokenizer applies unchanged** — but only as the regex stage. Two stages
wrap around it:

### 9a. `add_prefix_space` semantics (v0.23.3 `byte_level.rs:123-125`)

```rust
pretokenized.split(|_, mut normalized| {
    if self.add_prefix_space && !normalized.get().starts_with(' ') {
        normalized.prepend(" ");
    }
    if self.use_regex {
        normalized.split(re_ref, SplitDelimiterBehavior::Isolated)
    } ...
```

Conditional, not unconditional: a space is prepended **only if the piece does not already start
with `' '`**. So `"A"` -> `" A"` but `" A"` is untouched — hence `"A"` == `" A"`. And it is applied
**per piece handed to the pre-tokenizer**, i.e. per span surviving added-token extraction, not once
per input string. `SplitDelimiterBehavior::Isolated` means every regex match is kept as its own
pre-token and nothing is dropped.

### 9b. Added-token extraction runs *before* the regex, and bites

`AddedVocabulary` splits the (NFC-normalized) text on added tokens first, leftmost-longest. Because
2..24-space runs are added tokens, whitespace handling diverges from GPT-2 even though the regex is
identical. Verified live:

| input | ModernBERT ids | tokens |
|---|---|---|
| `"  A"` | `[50276, 329]` | `['  ', 'ĠA']` |
| `"   A"` | `[50275, 329]` | `['   ', 'ĠA']` |
| `"a  b"` | `[247, 50276, 270]` | `['Ġa', '  ', 'Ġb']` |
| `"    x"` | `[50274, 1269]` | `['    ', 'Ġx']` |
| 23 spaces | `[50255]` | one token |
| 24 spaces | `[50254]` | one token |
| 25 spaces | `[50254, 209]` | 24-run + `'Ġ'` |
| 26 spaces | `[50254, 50276]` | 24-run + 2-run |

Pure GPT-2 regex on `"  A"` would give `['Ġ', 'ĠA']` = `[209, 329]`. Actual is `[50276, 329]`.
Note also that after the `"  "` is carved out, the following `"b"`/`"A"` gets its own prefix space
(`'Ġb'`, `'ĠA'`), which is a second-order consequence of 9a being per-piece.

Single spaces, tabs and newlines are **not** added tokens and go through the regex normally:

| input | ids | tokens |
|---|---|---|
| `" "` | `[209]` | `['Ġ']` |
| `"\t"` | `[28910]` | `['Ġĉ']` |
| `"\n"` | `[2490]` | `['ĠĊ']` |
| `"\tA"` | `[209, 186, 34]` | `['Ġ', 'ĉ', 'A']` |
| `"\nA"` | `[209, 187, 34]` | `['Ġ', 'Ċ', 'A']` |
| `"a\n\nb"` | `[247, 187, 187, 67]` | `['Ġa', 'Ċ', 'Ċ', 'b']` |
| `"a\t\tb"` | `[247, 186, 186, 67]` | `['Ġa', 'ĉ', 'ĉ', 'b']` |
| `"a \n b"` | `[247, 2490, 270]` | `['Ġa', 'ĠĊ', 'Ġb']` |

(`"\tA"` -> 3 tokens while `"\t"` -> 1 is pure `\s+(?!\S)` backtracking on `" \tA"`; a correct GPT-2
regex implementation reproduces it. Good differential test cases.)

### 9c. `[MASK]` has `lstrip: true`

It swallows preceding whitespace, and the swallowed whitespace does **not** become a whitespace
added token:

| input | ids | tokens |
|---|---|---|
| `"x [MASK] y"` | `[1269, 50284, 340]` | `['Ġx', ' [MASK]', 'Ġy']` |
| `"x[MASK]y"` | `[1269, 50284, 340]` | `['Ġx', '[MASK]', 'Ġy']` |
| `"x  [MASK]"` | `[1269, 50284]` | `['Ġx', '  [MASK]']` |

The last row is the one to watch: the two spaces are consumed by `[MASK]`'s lstrip rather than
becoming token 50276.

Non-lstrip added tokens keep the space as its own pre-token:
`"mail me at |||EMAIL_ADDRESS||| ok"` -> `[8888, 479, 387, 209, 50277, 8718]` =
`['Ġmail', 'Ġme', 'Ġat', 'Ġ', '|||EMAIL_ADDRESS|||', 'Ġok']`.

## 10. What differs from a vanilla GPT-2 tokenizer

The BPE core is pure GPT-2: byte-level alphabet, the same hardcoded
`'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+` pattern, merge-rank
greedy merging, `ignore_merges: false`, `byte_fallback: false`, `fuse_unk: false`, no `unk_token`,
no subword prefix or suffix, so your verified pre-tokenizer regex and merge loop carry over
untouched. Four things wrap around it. First, there is an **NFC normalizer** before anything else
(GPT-2 has `normalizer: null`). Second, **`add_prefix_space` is `true`** rather than GPT-2's
`false`, applied conditionally (only when the piece does not already begin with a space) and
per-piece, which makes `"A"` and `" A"` collide on id 329 and is why ModernBERT-Large-Instruct's
answer-option ids are 329/378/330/399 and not 34/35/36/37. Third, a 116-entry **`added_tokens`
layer is extracted before the regex ever runs**: 23 tokens covering runs of 2 to 24 spaces, two PII
placeholders, `<|endoftext|>`/`<|padding|>`, the five BERT specials and 83 `[unused*]` slots. The
space-run tokens are the real trap, because they change whitespace output even though the regex is
byte-identical, and `[MASK]` carries `lstrip: true` so it eats the whitespace in front of it.
Fourth, there is a **`TemplateProcessing` post-processor** wrapping every sequence as
`[CLS] $A [SEP]` (50281 … 50282), with `type_id` 0 on both sequences of a pair, and the vocabulary
is 50368 rather than 50257.

## Not retrieved / caveats

- `crates.io` API returned HTTP 403 (needs a User-Agent), so the "latest published crate version"
  came from GitHub releases instead: v0.23.3 (2026-10-09) stable, v1.0.0-rc.3 (2026-10-19) pre-release.
- The live-encoding ground truth was produced with `tokenizers` 0.23.3 only. The v1.0.0-rc line was
  inspected by source, not executed.
- No attempt was made to diff ModernBERT's `vocab`/`merges` against OLMo's or GPT-2's beyond the
  counts above (50280 vocab / 50009 merges vs GPT-2's 50257 / 50000).
