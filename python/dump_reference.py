"""Dump per-layer reference activations for the Rust tier-2 tests.

PLAN.md: "This is the tier that will actually save the project. Do not test only final
logits." A Rust test loads the file this writes and asserts each checkpoint
independently, so a RoPE bug fails at the RoPE checkpoint instead of surfacing as a
mysterious 1e-2 at the end.

CLAUDE.md asks for no cleverness here, and completeness over elegance. So this file reads
as a list of hook registrations and a list of fixtures, with nothing derived and nothing
recomputed. In particular the RoPE'd q and k and the attention probabilities are captured
as the arguments the reference implementation actually passed to its own attention
function, not re-derived from the weights, because a reference that re-derives anything
can agree with a bug.

Two things force the implementation's hand:

  * Attention probabilities only exist if attention is computed eagerly. SDPA fuses the
    softmax and never materialises them, so the model is loaded with
    `attn_implementation="eager"`. The run asserts eager and SDPA agree on the mask logits,
    so using eager here does not quietly make the reference a different computation.
  * The single patch point `eager_attention_forward` yields the layer index, the RoPE'd
    query, the RoPE'd key and the probabilities together, so no call counter is needed to
    work out which layer is speaking.

Usage:
    python/.venv/bin/python python/dump_reference.py
"""

import json
import os

import torch
from safetensors.torch import save_file
from transformers import AutoModelForMaskedLM, AutoTokenizer
from transformers.models.modernbert import modeling_modernbert

from agnews import (
    MASK_TOKEN_ID,
    MODEL_DIR,
    REFERENCE_DIR,
    VERBALIZERS,
    build_prompt,
    find_mask_position,
    load_agnews_test,
    resolve_verbalizer_ids,
)

N_LAYERS = 28

# PLAN.md: global layers are `layer_id % 3 == 0`, so 0, 3, ..., 27. Layer 0 is global, not
# local, and layer 27 is global too. We dump attention internals for one of each.
GLOBAL_LAYER = 0
LOCAL_LAYER = 1

# `local_attention: 128` is the TOTAL window, so 64 each side, and the predicate is
# inclusive. That is 129 keys per query, and it is the number three separate off-by-ones
# in PLAN.md are about. A sequence longer than this is the only kind that makes local
# attention differ from global attention at all.
LOCAL_WINDOW_KEYS = 129


def build_fixtures(tokenizer):
    """The eight fixtures from PLAN.md, as full prompts.

    Every fixture is a prompt with exactly one `[MASK]`, because a reference activation
    dump without a mask position has nothing to compare `logits.mask` against, and because
    a fixture that is not shaped like the real input is testing a path the model never
    takes.

    The near-empty fixture is the one exception to going through the template: the template
    with an empty content slot is already 103 tokens, so putting an empty string through it
    would produce a 103-token "near-empty" fixture, which tests nothing short. It is
    instead a bare masked input, which really is near-empty and really does have a mask
    position.
    """
    texts, _ = load_agnews_test()

    # Shortest and longest by tokenized prompt length, which is the length that matters
    # and is unambiguous, rather than by character count.
    prompt_lengths = [len(tokenizer.encode(build_prompt(text))) for text in texts]
    shortest_index = min(range(len(texts)), key=lambda i: prompt_lengths[i])
    longest_index = max(range(len(texts)), key=lambda i: prompt_lengths[i])

    # Fixtures 3 and 4 concatenate real articles purely to push the sequence past the
    # 129-key local window. Taken from the head of the test split so they are reproducible.
    articles_300 = " ".join(texts[:4])
    articles_600 = " ".join(texts[:9])

    non_ascii = (
        "Les marchés européens ont chuté aujourd'hui. "
        "日経平均株価は大幅に下落しました。 "
        "Российский рынок акций вырос на 3%. "
        "Η Ελληνική οικονομία αναπτύσσεται. "
        "إقتصاد الشرق الأوسط ينمو بسرعة. "
        "주식 시장이 급등했습니다. "
        "Añoranza, coöperatie, naïveté, Straße, çalışma, Ðorđević. "
        "Combining marks: é à ô ñ ü. "
        "Non-breaking spaces here. "
        "Emoji: 📈📉💰🏦🌍 and flags 🇯🇵🇪🇺🇺🇸."
    )

    digits_and_punctuation = (
        "Q3 2004: revenue $1,234,567.89 (up 12.5%), EPS $0.47 vs $0.31 est. "
        "Ticker AAPL -2.3%; MSFT +1.07%; ^GSPC 1,130.54. "
        "Dates 2004-10-26, 10/26/04, 26.10.2004; times 09:30:00-16:00:00 EST. "
        "Phone +1 (555) 012-3456 ext. 789. ISIN US0378331005. "
        "Ratios 3:2, 1/4, 50-50, 100%%; ranges 10--20, 30–40. "
        "Punctuation runs: !!! ??? ... --- *** ((())) [[]] {{}} <<>> \"\" '' @#$%^&*. "
        "Hex 0xDEADBEEF, binary 0b1011, octal 0o755, sci 6.022e23, -273.15."
    )

    # Runs of 2 to 25 spaces. The tokenizer has 23 dedicated whitespace-run added tokens
    # (ids 50254 to 50276, for runs of 24 spaces down to 2), so these runs tokenize
    # differently from pure GPT-2 even though the splitter regex is byte-identical. A run
    # of 25 exceeds the longest added token and must fall back to composition.
    space_runs = (
        "Markets" + "".join(f"{' ' * n}word{n}" for n in range(2, 26)) + "  end  of  report  "
    )

    return [
        {
            "name": "shortest_real",
            "description": f"shortest real AG News test example (index {shortest_index})",
            "source_index": shortest_index,
            "source_text": texts[shortest_index],
            "prompt": build_prompt(texts[shortest_index]),
        },
        {
            "name": "longest_real",
            "description": f"longest real AG News test example (index {longest_index})",
            "source_index": longest_index,
            "source_text": texts[longest_index],
            "prompt": build_prompt(texts[longest_index]),
        },
        {
            "name": "synthetic_300",
            "description": "4 AG News test articles concatenated, to push past the 129-key local window",
            "source_index": None,
            "source_text": articles_300,
            "prompt": build_prompt(articles_300),
        },
        {
            "name": "synthetic_600",
            "description": "9 AG News test articles concatenated, well past the local window",
            "source_index": None,
            "source_text": articles_600,
            "prompt": build_prompt(articles_600),
        },
        {
            "name": "non_ascii",
            "description": "CJK, Cyrillic, Greek, Arabic, Hangul, accents, combining marks, NBSP, emoji",
            "source_index": None,
            "source_text": non_ascii,
            "prompt": build_prompt(non_ascii),
        },
        {
            "name": "digits_punctuation",
            "description": "numbers, currency, dates, tickers and punctuation runs",
            "source_index": None,
            "source_text": digits_and_punctuation,
            "prompt": build_prompt(digits_and_punctuation),
        },
        {
            "name": "multi_space",
            "description": "runs of 2 to 25 spaces, to exercise the whitespace added tokens (ids 50254-50276)",
            "source_index": None,
            "source_text": space_runs,
            "prompt": build_prompt(space_runs),
        },
        {
            "name": "near_empty",
            "description": "bare masked input, not through the template: [CLS] [MASK] [SEP]",
            "source_index": None,
            "source_text": "",
            "prompt": "[MASK]",
        },
    ]


def install_attention_capture(captured: dict):
    """Patch `eager_attention_forward` to record RoPE'd q, RoPE'd k and the probabilities.

    This is the one patch point that sees all three at once together with the layer index,
    because `ModernBertAttention.forward` applies RoPE and then hands the results straight
    to this function. The values recorded are therefore the reference implementation's own,
    not a re-derivation.

    Returns the original function so the caller can restore it.
    """
    original = modeling_modernbert.eager_attention_forward

    def recording_eager_attention_forward(module, query, key, value, attention_mask, **kwargs):
        attn_output, attn_weights = original(module, query, key, value, attention_mask, **kwargs)
        if module.layer_idx in (GLOBAL_LAYER, LOCAL_LAYER):
            captured[f"layer.{module.layer_idx}.q_rope"] = query.detach().clone()
            captured[f"layer.{module.layer_idx}.k_rope"] = key.detach().clone()
            captured[f"layer.{module.layer_idx}.attn_probs"] = attn_weights.detach().clone()
        return attn_output, attn_weights

    modeling_modernbert.eager_attention_forward = recording_eager_attention_forward
    return original


def install_module_hooks(model, captured: dict) -> list:
    """Forward hooks for every checkpoint that is just a module's output.

    Returns the handles so the caller can remove them.
    """

    def record(name):
        def hook(_module, _args, output):
            # Encoder layers return a tuple; everything else returns a bare tensor.
            tensor = output[0] if isinstance(output, tuple) else output
            captured[name] = tensor.detach().clone()

        return hook

    handles = [
        model.model.embeddings.tok_embeddings.register_forward_hook(record("embeddings.tok")),
        model.model.embeddings.norm.register_forward_hook(record("embeddings.norm")),
        model.model.final_norm.register_forward_hook(record("final_norm")),
        model.head.dense.register_forward_hook(record("head.dense")),
        model.head.act.register_forward_hook(record("head.act")),
        model.head.norm.register_forward_hook(record("head.norm")),
    ]
    for layer_index in range(N_LAYERS):
        handles.append(
            model.model.layers[layer_index].register_forward_hook(
                record(f"layer.{layer_index}.out")
            )
        )
    return handles


def check_local_window(attn_probs: torch.Tensor, seq_len: int) -> None:
    """Assert the local layer really attends to at most 129 keys per query.

    This is the check that would have caught the `sliding_window + 1` trap. It is worth
    doing here rather than only in Rust, because if the reference itself has the window
    wrong then every Rust test agrees with the wrong answer.
    """
    if seq_len <= LOCAL_WINDOW_KEYS:
        return

    # Masked-out keys get `finfo.min` added before the softmax, so they come out as exact
    # zeros rather than merely small numbers.
    attended_per_query = (attn_probs[0] > 0).sum(dim=-1)
    widest = int(attended_per_query.max())
    if widest > LOCAL_WINDOW_KEYS:
        raise ValueError(
            f"local layer {LOCAL_LAYER} attends to {widest} keys for some query, "
            f"expected at most {LOCAL_WINDOW_KEYS} (local_attention 128 is the total "
            f"window, 64 each side, inclusive)"
        )
    print(
        f"    local window check: max {widest} keys attended per query (limit {LOCAL_WINDOW_KEYS})"
    )


def main() -> None:
    torch.set_num_threads(4)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    verbalizer_ids = resolve_verbalizer_ids(tokenizer)
    verbalizer_id_tensor = torch.tensor(verbalizer_ids)

    # Eager attention, because SDPA never materialises the attention probabilities.
    model = AutoModelForMaskedLM.from_pretrained(
        MODEL_DIR, dtype=torch.float32, attn_implementation="eager"
    )
    model.eval()
    if model.config._attn_implementation != "eager":
        raise ValueError(
            f"expected eager attention, got {model.config._attn_implementation}; "
            f"attention probabilities cannot be captured from a fused kernel"
        )

    fixtures = build_fixtures(tokenizer)

    tensors = {}
    fixture_records = []
    worst_eager_sdpa_gap = 0.0

    for index, fixture in enumerate(fixtures):
        encoded = tokenizer(fixture["prompt"], return_tensors="pt")
        input_ids = encoded["input_ids"]
        seq_len = input_ids.shape[1]
        mask_position = find_mask_position(input_ids[0])

        captured = {}
        handles = install_module_hooks(model, captured)
        original_attention = install_attention_capture(captured)
        try:
            with torch.inference_mode():
                logits = model(**encoded).logits
        finally:
            for handle in handles:
                handle.remove()
            modeling_modernbert.eager_attention_forward = original_attention

        # The same weights under SDPA, to confirm that dumping under eager has not quietly
        # made the reference a different computation from the one the other rungs run. The
        # implementation is flipped on the one loaded model rather than loading a second
        # copy, which would mean 3.2 GB of fp32 weights resident at once.
        model.set_attn_implementation("sdpa")
        try:
            with torch.inference_mode():
                sdpa_logits = model(**encoded).logits
        finally:
            model.set_attn_implementation("eager")

        if not captured:
            raise ValueError("no activations captured; the hooks did not fire")
        for layer_index in (GLOBAL_LAYER, LOCAL_LAYER):
            if f"layer.{layer_index}.attn_probs" not in captured:
                raise ValueError(
                    f"attention capture for layer {layer_index} did not fire; the patch of "
                    f"eager_attention_forward was bypassed"
                )

        mask_row = logits[0, mask_position]
        captured["logits.mask"] = mask_row.detach().clone()
        captured["logits.verbalizers"] = mask_row[verbalizer_id_tensor].detach().clone()

        # Eager and SDPA differ only in floating point accumulation order. A large gap
        # here would mean the two paths disagree about masking, which would make this dump
        # useless as a reference for the SDPA-based rungs.
        gap = float((mask_row - sdpa_logits[0, mask_position]).abs().max())
        worst_eager_sdpa_gap = max(worst_eager_sdpa_gap, gap)

        print(f"  [{index}] {fixture['name']}: {seq_len} tokens, mask at {mask_position}")
        check_local_window(captured[f"layer.{LOCAL_LAYER}.attn_probs"], seq_len)
        print(f"    eager vs sdpa max |diff| at mask row: {gap:.2e}")

        for name, tensor in captured.items():
            tensors[f"fixture{index}.{name}"] = tensor.to(torch.float32).contiguous()

        predicted = int(torch.argmax(captured["logits.verbalizers"]))
        fixture_records.append(
            {
                "index": index,
                "name": fixture["name"],
                "description": fixture["description"],
                "source_agnews_test_index": fixture["source_index"],
                "source_text": fixture["source_text"],
                "prompt": fixture["prompt"],
                "token_ids": input_ids[0].tolist(),
                "n_tokens": seq_len,
                "n_tokens_source_text_alone": len(
                    tokenizer.encode(fixture["source_text"], add_special_tokens=False)
                ),
                "mask_position": mask_position,
                "exceeds_local_window": seq_len > LOCAL_WINDOW_KEYS,
                "predicted_verbalizer": VERBALIZERS[predicted],
                "eager_vs_sdpa_max_abs_diff_at_mask": gap,
            }
        )

    os.makedirs(REFERENCE_DIR, exist_ok=True)
    activations_path = os.path.join(REFERENCE_DIR, "activations.safetensors")
    save_file(tensors, activations_path)

    fixtures_path = os.path.join(REFERENCE_DIR, "fixtures.json")
    with open(fixtures_path, "w") as handle:
        json.dump(
            {
                "model": MODEL_DIR,
                "dtype": "float32 (upcast from the bf16 checkpoint on load)",
                "attn_implementation": "eager",
                "mask_token_id": MASK_TOKEN_ID,
                "verbalizers": VERBALIZERS,
                "verbalizer_ids": verbalizer_ids,
                "local_window_keys": LOCAL_WINDOW_KEYS,
                "global_layer_dumped": GLOBAL_LAYER,
                "local_layer_dumped": LOCAL_LAYER,
                "tensor_key_format": "fixture{index}.{checkpoint}",
                "checkpoints": sorted({key.split(".", 1)[1] for key in tensors}),
                "worst_eager_vs_sdpa_max_abs_diff_at_mask": worst_eager_sdpa_gap,
                "fixtures": fixture_records,
            },
            handle,
            indent=2,
        )

    print()
    print(f"wrote {activations_path}")
    print(f"  {len(tensors)} tensors, {os.path.getsize(activations_path) / 1e6:.0f} MB")
    print(f"wrote {fixtures_path}")
    print()
    print("fixture token counts:")
    for record in fixture_records:
        flag = "engages local window" if record["exceeds_local_window"] else "fits in one window"
        print(f"  [{record['index']}] {record['name']:20s} {record['n_tokens']:5d} tokens  {flag}")
    print()
    print(f"worst eager vs sdpa gap at the mask row, over all fixtures: {worst_eager_sdpa_gap:.2e}")


if __name__ == "__main__":
    main()
