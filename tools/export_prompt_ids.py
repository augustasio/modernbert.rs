"""Write the AG News prompt token ids once, for every rung to read.

Hard rule 1: every rung consumes byte-identical token ids, so tokenizer
differences can never reach a model timing or a model comparison. The ids come
from `crates/mb-tokenizer/tests/data/golden_agnews.jsonl.gz`, which is HF's
output and which our own tokenizer matches on all 7,600 prompts.

Output: `data/agnews_test_prompt_ids.tsv` (gitignored, regenerable), one line per
test example in dataset order:

    index <TAB> gold label <TAB> space-separated token ids

and its SHA-256 next to it, which every rung records with its results.

    python tools/export_prompt_ids.py
"""

import gzip
import hashlib
import json
import pathlib

REPO = pathlib.Path(__file__).resolve().parent.parent
GOLDEN = REPO / "crates/mb-tokenizer/tests/data/golden_agnews.jsonl.gz"
OUT = REPO / "data/agnews_test_prompt_ids.tsv"


def main() -> None:
    lines = []
    with gzip.open(GOLDEN, "rt", encoding="utf-8") as f:
        for raw in f:
            row = json.loads(raw)
            if row.get("header"):
                continue
            ids = " ".join(str(i) for i in row["prompt_ids"])
            lines.append(f"{row['i']}\t{row['label']}\t{ids}\n")

    if len(lines) != 7600:
        raise SystemExit(f"expected 7600 examples, found {len(lines)}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    body = "".join(lines).encode("utf-8")
    OUT.write_bytes(body)
    digest = hashlib.sha256(body).hexdigest()
    OUT.with_suffix(".sha256").write_text(digest + "\n")
    print(f"wrote {len(lines)} examples to {OUT.relative_to(REPO)}, sha256 {digest}")


if __name__ == "__main__":
    main()
