"""Why is `Tech` never predicted?

Rung 0 never picks verbalizer index 3. All gold-Tech examples go to `World` or `Business`,
while the other three classes sit almost perfectly on the diagonal. This script collects
the evidence to tell the two possible causes apart:

  a) a bug in the prompt, the mask position, or the verbalizer ids, or
  b) `ĠTech` is a genuinely weak verbalizer for this model, zero-shot.

It changes nothing. It only measures. The write-up is in TECH_VERBALIZER.md.

Four checks, in the order they would invalidate each other:

  1. Is the prompt byte-identical to the template in PLAN.md? Compared against the string
     extracted from PLAN.md itself, not against a copy, including the four trailing spaces
     after `** Instructions **`.
  2. Is the mask position right? Exactly one [MASK] (50284), preceded by [unused0] (50285).
  3. What does the model actually want to say at that position? Top 10 unrestricted over
     the whole vocabulary, decoded, with logits, for gold-Tech examples.
  4. Would a different single-token verbalizer for the Sci/Tech class do better? Measured
     over a balanced seeded sample, as evidence for a recommendation. This script does not
     act on it.

Memory: one model instance, fp32, batch 1 (CLAUDE.md hard rules 8 and 9).

Usage:
    python/.venv/bin/python python/diagnose_tech.py
"""

import os
import re

import numpy as np
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

from agnews import (
    MASK_TOKEN_ID,
    MODEL_DIR,
    PROMPT_TEMPLATE,
    REPO_ROOT,
    VERBALIZERS,
    build_prompt,
    find_mask_position,
    load_agnews_test,
    resolve_verbalizer_ids,
)

UNUSED0_TOKEN_ID = 50285

# Candidate replacements for the Sci/Tech class only. Each must be a single token to work
# in the single-mask scheme. `Sci/Tech` itself is three tokens, which is why the cookbook
# remapped it in the first place.
TECH_CANDIDATES = [
    "Tech",
    "Technology",
    "Science",
    "Sci",
    "Computer",
    "Internet",
    "Software",
    "Scientific",
]

# Balanced, seeded, and small enough to stay honest about run time.
SAMPLE_PER_CLASS = 50
SAMPLE_SEED = 0

# Examples to print the full top-10 for.
N_DETAILED = 6


def check_prompt_against_plan() -> bool:
    """Compare PROMPT_TEMPLATE to the fenced template block in PLAN.md, byte for byte.

    PLAN.md is the source of truth, so the comparison is against the file rather than
    against a second copy of the string. A diff in the four trailing spaces after
    `** Instructions **` would change the token ids, which is exactly the kind of silent
    divergence this check exists to catch.
    """
    print("=" * 78)
    print("1. Is the prompt byte-identical to PLAN.md?")
    print("=" * 78)

    with open(os.path.join(REPO_ROOT, "PLAN.md")) as handle:
        plan_text = handle.read()

    # The AG News template is the fenced block right after the "Prompt template" heading.
    after_heading = plan_text.split("### Prompt template", 1)[1]
    fenced = re.search(r"```\n(.*?)```", after_heading, re.DOTALL)
    if fenced is None:
        print("  could not find the fenced template block in PLAN.md")
        return False
    plan_template = fenced.group(1)

    # PLAN.md's block is a display form with a trailing newline after `{label}`; the
    # template is used without it. Compare the stripped-of-final-newline form.
    plan_template = plan_template.removesuffix("\n")

    identical = plan_template == PROMPT_TEMPLATE
    print(f"  PLAN.md block length   : {len(plan_template)} chars")
    print(f"  PROMPT_TEMPLATE length : {len(PROMPT_TEMPLATE)} chars")
    print(f"  byte-identical         : {identical}")

    if not identical:
        for line_no, (a, b) in enumerate(
            zip(plan_template.splitlines(), PROMPT_TEMPLATE.splitlines(), strict=False)
        ):
            if a != b:
                print(f"  first differing line {line_no}:")
                print(f"    PLAN.md : {a!r}")
                print(f"    ours    : {b!r}")
                break

    # The four trailing spaces, checked explicitly because they are invisible and because
    # an editor stripping trailing whitespace is the likely way to lose them.
    first_line = PROMPT_TEMPLATE.split("\n", 1)[0]
    trailing = len(first_line) - len(first_line.rstrip(" "))
    print(f"  trailing spaces after '** Instructions **': {trailing} (expected 4)")
    print(f"  first line repr: {first_line!r}")
    print()
    return identical and trailing == 4


def check_mask_position(tokenizer, text: str) -> None:
    """Exactly one [MASK], and it sits where the answer token sat in training."""
    print("=" * 78)
    print("2. Is the mask position right?")
    print("=" * 78)

    ids = tokenizer(build_prompt(text))["input_ids"]
    mask_positions = [i for i, token_id in enumerate(ids) if token_id == MASK_TOKEN_ID]

    print(f"  sequence length        : {len(ids)}")
    print(f"  [MASK] (50284) count   : {len(mask_positions)} at {mask_positions}")
    print(f"  [unused0] (50285) count: {ids.count(UNUSED0_TOKEN_ID)}")
    print(f"  find_mask_position()   : {find_mask_position(torch.tensor(ids))}")
    print("  note: with exactly one [MASK], first and last occurrence are the same slot, so")
    print("        this agrees with the reference harness, which takes the last.")
    print(f"  final 8 tokens         : {tokenizer.convert_ids_to_tokens(ids[-8:])}")
    print(
        f"  token before [MASK]    : {tokenizer.convert_ids_to_tokens([ids[mask_positions[0] - 1]])}"
    )
    print()


def report_candidate_ids(tokenizer) -> dict:
    """Which candidate replacements are single tokens, and what are their ids."""
    print("=" * 78)
    print("3. Candidate verbalizers for the Sci/Tech class")
    print("=" * 78)
    candidate_ids = {}
    for word in [*TECH_CANDIDATES, "Sci/Tech"]:
        encoded = tokenizer.encode(word, add_special_tokens=False)
        tokens = tokenizer.convert_ids_to_tokens(encoded)
        usable = len(encoded) == 1
        print(f"  {word:12s} -> {encoded!s:22s} {tokens!s:34s} single_token={usable}")
        if usable and word in TECH_CANDIDATES:
            candidate_ids[word] = encoded[0]
    print()
    return candidate_ids


def main() -> None:
    torch.set_num_threads(4)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    verbalizer_ids = resolve_verbalizer_ids(tokenizer)
    texts, gold = load_agnews_test()

    prompt_ok = check_prompt_against_plan()
    check_mask_position(tokenizer, texts[0])
    candidate_ids = report_candidate_ids(tokenizer)

    # One model instance only.
    model = AutoModelForMaskedLM.from_pretrained(MODEL_DIR, dtype=torch.float32)
    model.eval()

    verbalizer_id_tensor = torch.tensor(verbalizer_ids)
    candidate_id_tensor = torch.tensor(list(candidate_ids.values()))

    gold_array = np.array(gold)
    tech_indices = np.where(gold_array == VERBALIZERS.index("Tech"))[0]

    print("=" * 78)
    print(f"4. What the model wants to say, on {N_DETAILED} gold-Tech examples")
    print("=" * 78)
    with torch.inference_mode():
        for example_index in tech_indices[:N_DETAILED]:
            encoded = tokenizer(build_prompt(texts[example_index]), return_tensors="pt")
            mask_position = find_mask_position(encoded["input_ids"][0])
            row = model(**encoded).logits[0, mask_position]

            top = torch.topk(row, 10)
            decoded = [
                (tokenizer.convert_ids_to_tokens([int(i)])[0], round(float(v), 2))
                for v, i in zip(top.values, top.indices, strict=True)
            ]
            four = row[verbalizer_id_tensor]
            tech_rank = int((four > four[VERBALIZERS.index("Tech")]).sum()) + 1

            print(f"  example {example_index} (gold Tech)")
            print(f"    top10 unrestricted : {decoded}")
            print(
                "    four verbalizers   : "
                + ", ".join(f"{n} {float(v):.2f}" for n, v in zip(VERBALIZERS, four, strict=True))
            )
            print(f"    Tech rank of 4     : {tech_rank}")
            print(
                "    candidates         : "
                + ", ".join(f"{w} {float(row[i]):.2f}" for w, i in candidate_ids.items())
            )
    print()

    # A balanced seeded sample, so the comparison between candidate verbalizers is not
    # dominated by whichever class happens to be over-represented.
    rng = np.random.default_rng(SAMPLE_SEED)
    sample = []
    for class_index in range(len(VERBALIZERS)):
        class_indices = np.where(gold_array == class_index)[0]
        sample.extend(rng.choice(class_indices, SAMPLE_PER_CLASS, replace=False).tolist())
    sample.sort()

    print("=" * 78)
    print(
        f"5. Would another verbalizer help? Balanced sample, "
        f"{SAMPLE_PER_CLASS} per class, seed {SAMPLE_SEED}"
    )
    print("=" * 78)

    three_logits = np.zeros((len(sample), 3), dtype=np.float32)
    candidate_logits = np.zeros((len(sample), len(candidate_ids)), dtype=np.float32)
    sample_gold = gold_array[sample]

    with torch.inference_mode():
        for row_index, example_index in enumerate(sample):
            encoded = tokenizer(build_prompt(texts[example_index]), return_tensors="pt")
            mask_position = find_mask_position(encoded["input_ids"][0])
            row = model(**encoded).logits[0, mask_position]
            # Business, World, Sports stay fixed; only the fourth slot varies.
            three_logits[row_index] = row[verbalizer_id_tensor[:3]].numpy()
            candidate_logits[row_index] = row[candidate_id_tensor].numpy()

    print(
        f"  {'4th verbalizer':14s} {'overall acc':>12s} {'Tech recall':>12s} {'Tech predicted':>15s}"
    )
    results = {}
    for position, word in enumerate(candidate_ids):
        four = np.concatenate([three_logits, candidate_logits[:, position : position + 1]], axis=1)
        predicted = four.argmax(axis=1)
        accuracy = float((predicted == sample_gold).mean())
        tech_mask = sample_gold == 3
        tech_recall = float((predicted[tech_mask] == 3).mean())
        n_predicted_tech = int((predicted == 3).sum())
        results[word] = (accuracy, tech_recall, n_predicted_tech)
        print(f"  {word:14s} {accuracy:12.4f} {tech_recall:12.4f} {n_predicted_tech:15d}")

    print()
    print("  mean logit per class over the balanced sample:")
    for name, value in zip(VERBALIZERS[:3], three_logits.mean(axis=0), strict=True):
        print(f"    {name:12s} {value:7.3f}")
    for position, word in enumerate(candidate_ids):
        print(f"    {word:12s} {candidate_logits[:, position].mean():7.3f}  (candidate)")
    print()

    # If no synonym helps, the gap is a class prior rather than anything about the word.
    # The standard test for that is contextual calibration: score the same prompt with a
    # content-free content slot, which gives the model's preference with no news report to
    # go on, and subtract it. This measures whether the task is recoverable without
    # touching the prompt or the verbalizer set. It is a measurement, not a change: rung 0
    # keeps reporting the uncalibrated argmax.
    print("=" * 78)
    print("6. Is the gap a class prior? Contextual calibration probe")
    print("=" * 78)

    # The usual third null token in the literature is "[MASK]", but here that would put a
    # second [MASK] in the prompt and there would be no single answer slot. `find_mask_position`
    # catches it. "none" stands in.
    null_contents = ["N/A", "", "none"]
    null_rows = []
    with torch.inference_mode():
        for null_content in null_contents:
            encoded = tokenizer(build_prompt(null_content), return_tensors="pt")
            mask_position = find_mask_position(encoded["input_ids"][0])
            row = model(**encoded).logits[0, mask_position]
            null_rows.append(row[verbalizer_id_tensor].numpy())
            print(
                f"  content {null_content!r:10s} -> "
                + ", ".join(
                    f"{n} {float(v):.2f}"
                    for n, v in zip(VERBALIZERS, row[verbalizer_id_tensor], strict=True)
                )
            )

    null_bias = np.mean(null_rows, axis=0)
    print(
        "  mean content-free bias: "
        + ", ".join(f"{n} {v:.2f}" for n, v in zip(VERBALIZERS, null_bias, strict=True))
    )
    print()

    four_real = np.concatenate([three_logits, candidate_logits[:, 0:1]], axis=1)
    for label, scores in (
        ("uncalibrated", four_real),
        ("calibrated", four_real - null_bias),
    ):
        predicted = scores.argmax(axis=1)
        accuracy = float((predicted == sample_gold).mean())
        per_class = [
            float((predicted[sample_gold == c] == c).mean()) for c in range(len(VERBALIZERS))
        ]
        print(f"  {label:13s} acc {accuracy:.4f}   per-class recall: ", end="")
        print(", ".join(f"{n} {r:.2f}" for n, r in zip(VERBALIZERS, per_class, strict=True)))

    print()
    print("=" * 78)
    print("Verdict")
    print("=" * 78)
    print(f"  prompt byte-identical to PLAN.md : {prompt_ok}")
    print("  See TECH_VERBALIZER.md for the written diagnosis and the recommendation.")


if __name__ == "__main__":
    main()
