"""Write the HF token ids for every AG News test example.

This is the second and larger of the two tokenizer equality gates. The first,
`golden_strings.json`, is 92 hand-picked cases that name the behaviour they
test. This one is volume: all 7,600 test examples, as raw text and as the
prompt the model is actually given, so the gate covers what the project runs
rather than what someone thought to write down.

The output is JSON Lines, one object per example, which keeps the diff
readable and lets the Rust test stream it.

    python tools/make_golden_agnews.py

Needs `datasets` and `tokenizers`, and the checkpoint in `weights/`.
"""

import gzip
import json
import pathlib

import tokenizers
from datasets import load_dataset

REPO = pathlib.Path(__file__).resolve().parent.parent
TOKENIZER_JSON = REPO / "weights/ModernBERT-Large-Instruct/tokenizer.json"
# Gzipped: 11 MB of JSON Lines compresses to under 2 MB, which is a
# reasonable thing to keep in the repository and keeps the gate
# self-contained.
OUT = REPO / "crates/mb-tokenizer/tests/data/golden_agnews.jsonl.gz"

DATASET = "fancyzhx/ag_news"

# The reference harness remaps `Sci/Tech` to `Tech`, because `Sci/Tech` is
# three tokens while the other three class names are one each. The whole
# protocol rests on each verbalizer being a single token.
LABELS = ["World", "Sports", "Business", "Tech"]

TEMPLATE = (
    "** Instructions **    \n"
    "Given a short news report, determine which news category it belongs to.\n"
    "** Content **\n"
    "What is the category of the following news report?\n"
    "News Report: {text}\n"
    "** Options **\n"
    "- Business: It is about business or financial news.\n"
    "- World: It's about world news.\n"
    "- Sports: It's about sports.\n"
    "- Tech: It's about Science or Technology, or related companies.\n"
    "** Answer **\n"
    "Answer: [unused0] [MASK]"
)


def main() -> None:
    if not TOKENIZER_JSON.is_file():
        raise SystemExit(f"{TOKENIZER_JSON} is missing; fetch the checkpoint first")

    tok = tokenizers.Tokenizer.from_file(str(TOKENIZER_JSON))
    # `datasets` 5.x requires the canonical namespace/name form; the bare
    # "ag_news" alias no longer resolves.
    split = load_dataset(DATASET, split="test")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(OUT, "wt", encoding="utf-8", compresslevel=9) as f:
        f.write(
            json.dumps(
                {
                    "header": True,
                    "model": "answerdotai/ModernBERT-Large-Instruct",
                    "tokenizers_version": tokenizers.__version__,
                    "dataset": f"{DATASET}/test",
                    "count": len(split),
                    "labels": LABELS,
                    "generated_by": "tools/make_golden_agnews.py",
                },
                ensure_ascii=False,
            )
            + "\n"
        )
        for i, row in enumerate(split):
            text = row["text"]
            prompt = TEMPLATE.format(text=text)
            f.write(
                json.dumps(
                    {
                        "i": i,
                        "label": LABELS[row["label"]],
                        "text": text,
                        # Raw text, which gives the widest variety of input.
                        "ids": tok.encode(text, add_special_tokens=True).ids,
                        # And the real prompt, which is what gets measured.
                        "prompt_ids": tok.encode(prompt, add_special_tokens=True).ids,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    size_mb = OUT.stat().st_size / 1e6
    print(f"wrote {len(split)} examples to {OUT.relative_to(REPO)} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
