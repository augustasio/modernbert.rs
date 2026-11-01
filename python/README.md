# Python side

Rungs 0 to 3 of the ladder, plus the reference activation dumps that the Rust tests
verify against.

- Rung 0: HF transformers, stock pipeline. The correctness oracle for everything else.
- Rung 1: MLM head at the mask position only.
- Rung 2: `torch.compile`, inductor, CPU.
- Rung 3: ONNX Runtime.

Also here: the forward hooks that dump per-layer activations to safetensors for tier-2
verification, and the plotting for the p99-against-throughput frontier.

Before anything else, run the BLAS check. If torch is linked against bundled OpenBLAS
rather than Accelerate, rungs 0 to 3 are running on a different BLAS from rungs 5 and 7
and the headline comparison is indefensible.

## Setup

Python 3.13.7 is the system interpreter. The venv lives at `python/.venv` and is
gitignored.

```sh
python3.13 -m venv python/.venv
python/.venv/bin/python -m pip install -r python/requirements.txt
```

Fetch the checkpoint into `weights/` (gitignored, 755 MiB, bf16):

```sh
python/.venv/bin/python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='answerdotai/ModernBERT-Large-Instruct',
    local_dir='weights/ModernBERT-Large-Instruct',
    allow_patterns=['config.json', 'tokenizer.json', 'tokenizer_config.json',
                    'special_tokens_map.json', 'model.safetensors'],
)"
```

The scripts anchor their paths to the repo root, so they can be run from anywhere.

## Running each script

**`blas_check.py`** prints torch's build configuration, the BLAS actually linked into
`libtorch_cpu`, and measured sgemm throughput, then states a verdict. Run it first. The
written-up answer for this machine is in [BLAS.md](BLAS.md).

```sh
python/.venv/bin/python python/blas_check.py
```

**`agnews.py`** is not a script. It holds the prompt template, the four verbalizers and
their asserted token ids, and the dataset loader, so that every rung consumes byte-identical
token ids (CLAUDE.md hard rule 1). Import it, do not copy out of it.

**`rung0_baseline.py`** runs the full 7,600-example zero-shot AG News classification
through the stock HF pipeline in fp32, batch size 1, nothing padded. Takes roughly half an
hour on the M4 Air. Writes `results/rung0.json` (accuracy, per-example predictions, gold
labels, confusion matrix) and `results/rung0_logits.npy` (the four verbalizer logits per
example, shape 7600x4). Fails loudly if accuracy is not clearly above chance.

```sh
python/.venv/bin/python python/rung0_baseline.py                          # full set
python/.venv/bin/python python/rung0_baseline.py --limit 64 --tag first64  # smoke test
python/.venv/bin/python python/rung0_baseline.py --threads 1              # pin threads
```

`--tag` suffixes the output filenames, so a partial run cannot overwrite the full-set
baseline that every other rung is asserted against. `results/rung0_first64.json` is kept
for comparison, but it is not a baseline: the first 64 examples of the test split are 30
of 64 gold Sci/Tech against 1900 of 7600 in the full split, so that subset is badly skewed.

**`diagnose_tech.py`** answers why the Sci/Tech class is never predicted. It checks the
prompt against `PLAN.md` byte for byte, checks the mask position, prints the unrestricted
top 10 at the mask for gold-Tech examples, scores eight alternative verbalizers over a
balanced seeded sample, and probes the class prior with a content-free prompt. It changes
nothing. See [TECH_VERBALIZER.md](TECH_VERBALIZER.md).

```sh
python/.venv/bin/python python/diagnose_tech.py
```

**`calibration.py`** implements contextual calibration (Zhao et al. 2021) as two pure
functions, `fit(content_free_logits)` and `apply(logits, params)`. Run as a script, it loads
the model once, scores three content-free prompts, and writes `results/calibration.json`
(the content-free logits and the fitted params). Calibration is post-processing only. Raw
logits stay the oracle.

**`stats.py`** reads `results/rung0.json`, `results/rung0_logits.npy` and
`results/calibration.json`, and writes `results/rung0_stats.json`. It reports bootstrap and
Wilson confidence intervals, raw and calibrated, and a paired comparison. It does not load
the model. Method and results are in [STATISTICS.md](STATISTICS.md).

```sh
python/.venv/bin/python python/calibration.py
python/.venv/bin/python python/stats.py
```

**`dump_reference.py`** re-runs rung 0 under forward hooks over eight fixtures and writes
the tier-2 reference that the Rust tests assert against: `python/reference/activations.safetensors`
(gitignored) and `python/reference/fixtures.json` (the prompts, their token ids, token
counts and mask positions). Takes about a minute.

```sh
python/.venv/bin/python python/dump_reference.py
```

Tensor keys are `fixture{index}.{checkpoint}`, where checkpoint is one of
`embeddings.tok`, `embeddings.norm`, `layer.{0..27}.out`, `layer.{0,1}.q_rope`,
`layer.{0,1}.k_rope`, `layer.{0,1}.attn_probs`, `final_norm`, `head.dense`, `head.act`,
`head.norm`, `logits.mask`, `logits.verbalizers`. Layer 0 is a global layer and layer 1 is
a local one, so the pair covers both attention types.

The dumper loads the model with `attn_implementation="eager"`, because SDPA fuses the
softmax and never materialises the attention probabilities. It asserts eager and SDPA
agree on the mask logit row, so the reference is not a different computation from the one
the other rungs run.

## Linting

`ruff` config is in `python/ruff.toml`, 100 columns to match rustfmt. Nothing lands red.

```sh
python/.venv/bin/ruff check python/
python/.venv/bin/ruff format --check python/
```

## Two findings worth knowing before reading the results

**The local attention window does engage on the real dataset.** `PLAN.md` reasons that AG
News examples are 40 to 60 tokens, so every query attends to the whole sequence and local
attention is numerically identical to global attention on real data. That holds for the raw
articles (17 to 246 tokens, mean 52.6) but not for what the model is actually fed: the
prompt template with an empty content slot is already 103 tokens, so full prompts run 119
to 348 tokens with a median of 152, and **96.8% of them (7,357 of 7,600) exceed the
129-key window**. The
sliding window is therefore exercised by the tier-3 full-set agreement test, not only by
the synthetic fixtures. The long fixtures are still worth having, because at 152 tokens
only a sliver of the attention matrix is masked, while the 300+ and 600+ token fixtures
engage the window hard.

**Zero-shot, the model never says `Tech`, and no verbalizer fixes it.** All of the error is
concentrated in the Sci/Tech class, which is never predicted. This is not a bug: the
prompt is byte-identical to `PLAN.md`, the mask position is the right slot, and the
verbalizer ids are correct. The cause is a class prior. On a content-free prompt, before
the model sees any article, `Tech` is already 6 logits below `World`, `Business` and
`Sports`, while those three sit within 0.8 of each other. Eight single-token replacements
for `Tech` were measured and every one gives identical accuracy and zero predictions.
Subtracting the content-free bias moves accuracy from 0.72 to 0.87 and Tech recall from
0.00 to 0.66 on a balanced sample, so the model does carry the information; only the
decision rule discards it. Full diagnosis, evidence and recommendation in
[TECH_VERBALIZER.md](TECH_VERBALIZER.md); reproduce with `diagnose_tech.py`. Rung 0 is
unchanged and still reports the uncalibrated argmax.
