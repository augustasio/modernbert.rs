"""Write the golden token ids that the Rust tokenizer tests assert against.

HF `tokenizers` is the oracle. This script asks it for the ids of a list of
inputs chosen to hit the parts of the pipeline that are easy to get wrong, and
writes them to a JSON file the Rust tests read.

The Rust encoder must reproduce these exactly. A case here is cheaper to debug
than a mismatch somewhere in the 7,600 AG News examples, so this file is the
first gate and the AG News golden file is the second.

Run it after any change to the case list:

    python tools/make_golden_strings.py

It needs the `tokenizers` package and the checkpoint in `weights/`.
"""

import json
import pathlib

import tokenizers

REPO = pathlib.Path(__file__).resolve().parent.parent
TOKENIZER_JSON = REPO / "weights/ModernBERT-Large-Instruct/tokenizer.json"
OUT = REPO / "crates/mb-tokenizer/tests/data/golden_strings.json"

# Each case is (label, text). The label says what the case is for, so a Rust
# test failure names the behaviour rather than just the bytes.
CASES: list[tuple[str, str]] = [
    # --- the empty and near-empty inputs ---
    ("empty", ""),
    ("one space", " "),
    ("one letter", "a"),
    # --- add_prefix_space, which is true for this model ---
    ("bare letter gets a prefix space", "A"),
    ("letter already has a space", " A"),
    ("bare word", "Hello"),
    ("word with a space", " Hello"),
    ("two words", "Hello world"),
    # --- the whitespace-run added tokens, 2 to 24 spaces ---
    ("two spaces then a letter", "  A"),
    ("three spaces then a letter", "   A"),
    ("four spaces then a letter", "    x"),
    ("eight spaces", "        "),
    ("24 spaces, the longest run token", " " * 24),
    ("25 spaces, one over the longest", " " * 25),
    ("26 spaces", " " * 26),
    ("a run between letters", "a  b"),
    ("a long run between letters", "a" + " " * 10 + "b"),
    # --- single whitespace characters, which are not added tokens ---
    ("one tab", "\t"),
    ("tab then a capital", "\tA"),
    ("one newline", "\n"),
    ("CRLF", "\r\n"),
    ("two newlines between letters", "a\n\nb"),
    ("mixed whitespace between letters", "a \n b"),
    ("trailing newline", "hello\n"),
    ("leading newline", "\nhello"),
    # --- [MASK] and its lstrip flag ---
    ("mask with spaces around it", "x [MASK] y"),
    ("mask with no spaces", "x[MASK]y"),
    ("mask after two spaces, which get eaten", "x  [MASK]"),
    ("mask at the start", "[MASK] y"),
    ("mask alone", "[MASK]"),
    ("the real answer tail", "Answer: [unused0] [MASK]"),
    # --- other added tokens, which do not lstrip ---
    ("an email placeholder", "mail me at |||EMAIL_ADDRESS||| ok"),
    ("an ip placeholder", "host |||IP_ADDRESS||| up"),
    ("unused0 alone", "[unused0]"),
    ("endoftext", "a <|endoftext|> b"),
    # --- the specials written into the text ---
    ("CLS written by hand", "[CLS] hello"),
    ("SEP written by hand", "hello [SEP]"),
    # --- NFC normalization ---
    ("precomposed e acute", "café"),
    ("decomposed e acute, must match the precomposed one", "café"),
    ("precomposed a diaeresis", "naïve"),
    ("decomposed a diaeresis", "naïve"),
    ("a combining mark with no base", "́"),
    ("hangul, which NFC composes", "가"),
    # --- contractions, case-sensitive in the GPT-2 pattern ---
    ("lowercase contraction", "don't"),
    ("uppercase contraction", "DON'T"),
    ("mixed case contraction", "Don'T"),
    ("every contraction", "it's it't we're we've I'm we'll I'd"),
    ("a bare contraction", "'s"),
    ("an apostrophe that is not a contraction", "x'y"),
    ("a run of apostrophes", "''''"),
    # --- digits, which GPT-2 keeps whole ---
    ("one digit", "1"),
    ("ten digits", "1234567890"),
    ("digits with a space", " 1234"),
    ("a decimal", "12.34"),
    ("grouped digits", "1,234,567"),
    ("letters and digits", "a1b2c3"),
    ("a year in a sentence", "In 2024 the model shipped."),
    # --- punctuation and symbols ---
    ("one dot", "."),
    ("space then dot", " ."),
    ("dots", "..."),
    ("punctuation run", "!?!?"),
    ("an arrow", " -> "),
    ("brackets", "(a)"),
    ("currency", "$100"),
    ("a slash", "a/b"),
    ("a url", "https://example.com/path?a=1&b=2#frag"),
    ("rust code", "fn main() { let x: Vec<u32> = vec![1, 2]; }"),
    ("json", '{"key": "value", "n": [1, 2, 3]}'),
    # --- non-ASCII scripts ---
    ("cyrillic", "Привет мир"),
    ("greek", "γειά σου"),
    ("hebrew", "שלום"),
    ("arabic", "مرحبا"),
    ("CJK", "日本語"),
    ("hiragana", "こんにちは"),
    ("hangul syllables", "한국어"),
    ("mixed scripts", "中文 and English"),
    ("emoji", "\U0001f389"),
    ("emoji with joiners", "\U0001f468‍\U0001f469‍\U0001f467‍\U0001f466"),
    ("regional indicators", "\U0001f1ec\U0001f1e7"),
    ("non-breaking space", "a b"),
    ("ideographic space", "a　b"),
    ("thin space", "a b"),
    ("zero width space, not whitespace", "a​b"),
    ("a vulgar fraction", "½"),
    ("a roman numeral", "Ⅻ"),
    ("arabic-indic digits", "٣٤٥"),
    # --- the AG News verbalizers, which must each be one token ---
    ("verbalizer Business", " Business"),
    ("verbalizer World", " World"),
    ("verbalizer Sports", " Sports"),
    ("verbalizer Tech", " Tech"),
    ("the label that is not one token", " Sci/Tech"),
    # --- the real prompt, including its four trailing spaces ---
    (
        "the AG News template at inference",
        "** Instructions **    \n"
        "Given a short news report, determine which news category it belongs to.\n"
        "** Content **\n"
        "What is the category of the following news report?\n"
        "News Report: Wall St. Bears Claw Back Into the Black (Reuters) Reuters - "
        "Short-sellers, Wall Street's dwindling band of ultra-cynics, are seeing "
        "green again.\n"
        "** Options **\n"
        "- Business: It is about business or financial news.\n"
        "- World: It's about world news.\n"
        "- Sports: It's about sports.\n"
        "- Tech: It's about Science or Technology, or related companies.\n"
        "** Answer **\n"
        "Answer: [unused0] [MASK]",
    ),
]


def main() -> None:
    if not TOKENIZER_JSON.is_file():
        raise SystemExit(f"{TOKENIZER_JSON} is missing; fetch the checkpoint first")

    tok = tokenizers.Tokenizer.from_file(str(TOKENIZER_JSON))

    seen: set[str] = set()
    cases = []
    for label, text in CASES:
        if text in seen:
            raise SystemExit(f"duplicate case text for {label!r}")
        seen.add(text)
        cases.append(
            {
                "label": label,
                "text": text,
                # The post-processor adds [CLS] and [SEP] here.
                "ids": tok.encode(text, add_special_tokens=True).ids,
                # And leaves them off here, so the Rust tests can check the
                # pre-tokenizer, the merges and the post-processor apart.
                "ids_no_specials": tok.encode(text, add_special_tokens=False).ids,
                # Decoding is lossy, and in more ways than the leading space
                # that `add_prefix_space` explains: it injects a space into
                # every span, so each added-token boundary gains one too.
                # `"a  b"` decodes to `" a   b"`. HF is the oracle for this
                # as well, so record its answer rather than assume a round
                # trip.
                "decoded": tok.decode(
                    tok.encode(text, add_special_tokens=False).ids,
                    skip_special_tokens=False,
                ),
            }
        )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(
            {
                "model": "answerdotai/ModernBERT-Large-Instruct",
                "tokenizers_version": tokenizers.__version__,
                "generated_by": "tools/make_golden_strings.py",
                "cases": cases,
            },
            indent=1,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(cases)} cases to {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
