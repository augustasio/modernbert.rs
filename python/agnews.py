"""The AG News zero-shot task: prompt template, verbalizers, dataset.

Shared by every Python rung and by the reference dumper, because CLAUDE.md's first hard
rule is that every rung consumes byte-identical token ids. Two copies of this template
that differ by one space would silently become two different experiments, so there is one
copy and everything imports it.

Everything here is pinned to `docs/research/prompt-template.md`, which traced each string
back to the AnswerDotAI cookbook rather than to the model card. Read that document before
changing anything in this file.
"""

import os

import torch
from datasets import load_dataset

# Paths are anchored to the repo root rather than the working directory, so the scripts
# behave the same whether they are run from the repo root or from python/.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The AG News template from the cookbook's `train.py` (source S4 in the research doc).
#
# Trap: there are four trailing spaces after `** Instructions **`. They are real,
# confirmed with `cat -vet` and by `ast`-extracting the string literal from the original
# source. Do not let an editor strip trailing whitespace in this file.
#
# `{label}` becomes `[MASK]` at inference. `[CLS]` and `[SEP]` are never written here;
# the tokenizer's TemplateProcessing post-processor adds them.
PROMPT_TEMPLATE = (
    "** Instructions **    \n"
    "Given a short news report, determine which news category it belongs to.\n"
    "** Content **\n"
    "What is the category of the following news report?\n"
    "News Report: {txt}\n"
    "** Options **\n"
    "- Business: It is about business or financial news.\n"
    "- World: It's about world news.\n"
    "- Sports: It's about sports.\n"
    "- Tech: It's about Science or Technology, or related companies.\n"
    "** Answer **\n"
    "Answer: [unused0] {label}"
)

# Order matters and is the order the options appear in the template. Every artefact this
# project writes indexes classes this way, so a reordering here is a silent relabelling of
# every committed result.
VERBALIZERS = ["Business", "World", "Sports", "Tech"]

# Trap: `add_prefix_space: true` means encoding "Business" yields `ĠBusiness`, and the
# bare-word vocab entries are unreachable by encoding. These are the ids of the
# space-prefixed forms, which is what the model was trained to raise at the answer
# position. They are asserted, never trusted: see `resolve_verbalizer_ids`.
EXPECTED_VERBALIZER_IDS = [10518, 3645, 15000, 18447]

MASK_TOKEN_ID = 50284

# The dataset's own class names, in its own label order. `Sci/Tech` is remapped to `Tech`
# because `Sci/Tech` tokenizes to three tokens ([22962, 16, 30663]) while the single-mask
# scheme needs a one-token verbalizer. This remap is in the cookbook's
# `agnews_label_map.json`, it is not our invention.
DATASET_LABEL_TO_VERBALIZER = {
    "World": "World",
    "Sports": "Sports",
    "Business": "Business",
    "Sci/Tech": "Tech",
}

# The canonical `ag_news` id no longer resolves: datasets 5.x routes through
# huggingface_hub's URI parser, which requires `namespace/name` and rejects a bare id.
# `fancyzhx/ag_news` is the same dataset under its namespaced name.
AGNEWS_REPO = "fancyzhx/ag_news"

MODEL_DIR = os.path.join(REPO_ROOT, "weights", "ModernBERT-Large-Instruct")
RESULTS_DIR = os.path.join(REPO_ROOT, "results")
REFERENCE_DIR = os.path.join(REPO_ROOT, "python", "reference")


def build_prompt(text: str) -> str:
    """The full prompt for one news report, with the answer slot masked."""
    return PROMPT_TEMPLATE.format(txt=text, label="[MASK]")


def resolve_verbalizer_ids(tokenizer) -> list[int]:
    """Resolve the four verbalizer ids by encoding, then assert them.

    Resolving by encoding rather than by raw vocab lookup is the whole point: a vocab
    lookup of "Business" returns the id of the bare form, which `add_prefix_space: true`
    makes unreachable by encoding and which the model was never trained to raise at this
    position. The classifier would then read a logit that means nothing.

    The assert is here because the failure is silent otherwise. Reading the wrong logits
    produces a plausible-looking accuracy, not a crash.
    """
    ids = []
    for word in VERBALIZERS:
        encoded = tokenizer.encode(word, add_special_tokens=False)
        if len(encoded) != 1:
            raise ValueError(
                f"verbalizer {word!r} encoded to {len(encoded)} tokens {encoded}, "
                f"expected exactly 1; the single-mask scheme requires one token"
            )
        ids.append(encoded[0])

    if ids != EXPECTED_VERBALIZER_IDS:
        pairs = list(zip(VERBALIZERS, ids, EXPECTED_VERBALIZER_IDS, strict=True))
        detail = ", ".join(f"{w}: got {g}, expected {e}" for w, g, e in pairs)
        raise ValueError(
            f"verbalizer ids do not match the values pinned in PLAN.md ({detail}). "
            f"Either the tokenizer's add_prefix_space setting changed or the checkpoint "
            f"is not ModernBERT-Large-Instruct."
        )

    if tokenizer.mask_token_id != MASK_TOKEN_ID:
        raise ValueError(f"[MASK] id is {tokenizer.mask_token_id}, expected {MASK_TOKEN_ID}")

    return ids


def find_mask_position(input_ids: torch.Tensor) -> int:
    """Index of the single `[MASK]` in one encoded prompt.

    Asserts there is exactly one. The template puts `[MASK]` last, but `[MASK]` has
    `lstrip: true` and swallows the space before it, so its index is not simply
    len - 2 for every input; locate it by id.
    """
    positions = (input_ids == MASK_TOKEN_ID).nonzero().flatten().tolist()
    if len(positions) != 1:
        raise ValueError(f"expected exactly one [MASK], found {len(positions)}: {positions}")
    return positions[0]


def load_agnews_test():
    """The 7,600-example AG News test split, with gold labels as verbalizer indices.

    Returns (texts, gold) where gold indexes VERBALIZERS, not the dataset's own labels.
    """
    dataset = load_dataset(AGNEWS_REPO, split="test")
    dataset_names = dataset.features["label"].names

    texts = dataset["text"]
    gold = []
    for label in dataset["label"]:
        verbalizer = DATASET_LABEL_TO_VERBALIZER[dataset_names[label]]
        gold.append(VERBALIZERS.index(verbalizer))

    return texts, gold
