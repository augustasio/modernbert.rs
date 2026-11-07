# Fine-tuning ModernBERT on the M4 Air

Research agent report, 2026-11-07. All M4 times are estimates, extrapolated from other
Apple chips. Accuracy figures are measured, but mostly self-reported on model cards. See
`README.md` for the prompt-length correction: the template is about 103 tokens and a
typical prompt about 152, which matches this report's "about 100 tokens of instructions".

## Verdict

- **Feasible:** yes. ModernBERT-base can be fully fine-tuned on this Mac with PyTorch MPS
  (the M4's GPU). ModernBERT-large also fits in 16 GB, but trains slowly.
- **How long (estimate):** about 10 to 20 min for base on a 20k subset, 1 to 2 h for base
  on all 120k, 3 to 6 h for large on 120k, with throttling.
- **Worth it:** yes as a side track, no as a rung. Published fine-tunes of base-sized
  encoders reach 94 to 95% on AG News, against our 84.8% calibrated zero-shot. A base model
  with a classification head should also run several times faster than the large prompt
  model. The speedup comes from three changes at once (model size, input length, head), so
  it cannot be a one-variable rung.
- **Main constraint:** training heats this same fanless machine. Never train while a timing
  benchmark runs, and let the machine cool before the next one. Training on Colab or Kaggle
  also works: where a model was trained does not change its inference speed.

## 1. Feasibility on MPS

Documented:
- The venv has torch 2.15.0 and transformers 5.22.1. The installed `modeling_modernbert.py`
  supports SDPA, flash and flex attention. Without Flash Attention 2 it uses PyTorch's SDPA
  kernel, "which requires padding and unpadding inputs, adding some overhead".
- Flash Attention does not exist on MPS, so ModernBERT runs padded SDPA there: slower, not
  broken. One user measured MPS inference in bf16 at about 120 docs/s for ModernBERT-base
  against 150 for DeBERTa-v3
  ([HF discussion #47](https://huggingface.co/answerdotai/ModernBERT-base/discussions/47)).
- Older Accelerate docs warned that BERT training on MPS gave worse metrics than on CPU
  ([Accelerate MPS guide](https://huggingface.co/docs/accelerate/v0.12.0/usage_guides/mps)).
  That is from torch 1.x: a reason for a smoke check, not a blocker.
- An MLX route exists: [mlx-modernbert](https://pypi.org/project/mlx-modernbert/) fine-tunes
  ModernBERT-large on Apple Silicon with fp16 and gradient checkpointing.

Estimates:
- **Precision:** start in fp32. Try bf16 autocast only after a 50-step run gives the same
  loss curve as fp32.
- **torch.compile:** skip it. MPS support for inductor is immature.
- **Memory, full fine-tune with AdamW in fp32,** about 16 bytes per parameter:
  - Base: 149M x 16 B = about 2.4 GB plus activations. Batch 32 at length 128 should fit
    in under 6 GB.
  - Large: 395M x 16 B = about 6.3 GB plus activations. Batch 16 at length 128 is about 9
    to 11 GB, tight beside macOS. Use gradient checkpointing, or batch 8 with accumulation.
- **LoRA on large:** fits easily, but saves little time, because the forward and backward
  passes still go through the full model.

## 2. Time

Measured on other Apple chips (bert-base-cased, fp32, 100 steps), from
[lucadiliello/pytorch-apple-silicon-benchmarks](https://github.com/lucadiliello/pytorch-apple-silicon-benchmarks):
- Batch 64, length 128: M3 Pro 14-core GPU about 41 ex/s; M3 Max 40-core GPU about
  88 ex/s; T4 about 53 ex/s.
- Batch 16, length 128: M1 Max 32-core GPU about 25 ex/s.
- These used torch 1.13 to 2.2; newer torch is probably faster.

[mrdbourke's M3 test](https://www.mrdbourke.com/apple-m3-machine-learning-test) found that
MPS training speed closely follows GPU core count.

Extrapolated to the M4 Air (10-core GPU): about 25 to 40 ex/s for ModernBERT-base with
dynamic padding on AG News texts of 40 to 60 tokens, and 8 to 13 ex/s for large.

| Run | Base (1 epoch) | Large (1 epoch) |
|---|---|---|
| 5k | 2 to 4 min | 7 to 10 min |
| 20k | 8 to 15 min | 25 to 45 min |
| 120k | 50 to 90 min | 2.5 to 4.5 h |

Throttling: no rigorous measurement for the M4 Air GPU was found. An M3 Air lost 27% in a
sustained 3DMark GPU loop. One blog reports 25 to 40% loss after 12 to 15 min for M4 Air
MLX inference ([CraftRigs](https://craftrigs.pages.dev/guides/macbook-air-m4-thermal-throttling-30min-test/)).
Assume runs over 15 min go 25 to 40% slower than the first minutes; the long rows above
roughly include this.

## 3. Usefulness

Reported fine-tuned AG News test accuracy:
- ModernBERT-base: 94.3% after 3 epochs at lr 2e-5
  ([svenk029/mb-agn](https://huggingface.co/svenk029/mb-agn)). Other community cards
  report 91.5 to 92.4% after 2 epochs
  ([jhszptr](https://huggingface-proxy.pages.dev/jhszptr/ModernBERT-base-finetuned-agnews),
  [cdhartono](https://huggingface.co/cdhartono/modernbert-agnews)). Self-reported.
- BERT-base: about 94.2 to 94.7%
  ([nateraw](https://huggingface.co/nateraw/bert-base-uncased-ag-news/blob/refs%2Fpr%2F1/README.md),
  [ArmanXAI](https://huggingface.co/ArmanXAI/bert-base-agnews-news-classifier)).
- RoBERTa: about 94.9% base ([card](https://huggingface.co/dasoteloro/roberta-base_ag_news))
  and about 95.6% large ([aggregator](https://www.wizwand.com/dataset/ag-news)).
- ModernBERT-Large-Instruct: 95.24% after a 2-epoch full fine-tune through the MLM head
  (PLAN.md, from the Instruct paper).

Few-shot, measured in the SetFit paper ([arXiv 2209.11055](https://arxiv.org/abs/2209.11055),
Table 2, AG News):

| Method | 8 per class (32 total) | 64 per class (256 total) |
|---|---|---|
| Standard fine-tune (RoBERTa-large) | 81.7 | 88.4 |
| SetFit (MPNet) | 82.9 | 88.0 |
| ADAPET (prompt-based) | 85.1 | 88.0 |

Full-data standard fine-tuning reached 93.8. So about 100 to 300 labelled examples should
beat 84.8%, and 5k examples should give about 91 to 93% (interpolated).

Faster and more accurate? Probably both (estimate): base has about 3x fewer non-embedding
FLOPs than large, and the classification input is only the news text, about 50 tokens
against a typical 152-token prompt. Combined, roughly 5 to 9x faster than zero-shot large;
to be measured. Prompt-based fine-tuning of Large-Instruct's MLM head gives the best
accuracy, but keeps the large-model cost.

## 4. How it fits the project

- A fine-tuned model is not a rung. It changes the weights, the task (now supervised), the
  input (no prompt) and the output head at once.
- Report it as a separate track, for example "Track B: task-adapted".
- Keep fixed: the same 7,600 test examples and the same accuracy definition, and run the
  fine-tuned model on both the best rung's implementation and rung 0, so engine speed stays
  separate from model speed.
- Report the labelled examples used, the training data source, and training time and
  hardware. Zero-shot uses 0 labels.
- Present it as accuracy against ms per example: zero-shot raw, zero-shot calibrated, and
  fine-tuned at 256, 5k and 120k labels.
- Optional: to separate model size from input length, also fine-tune base with the same
  prompt through the MLM head.

## 5. A recipe

- **Code:** a plain PyTorch loop, or HF `Trainer`. `AutoModelForSequenceClassification`
  with `answerdotai/ModernBERT-base`, `attn_implementation="sdpa"`, `device="mps"`, fp32.
- **Data:** stratified subsets of AG News train, 5k then 20k. Hold out 2k of train for
  validation; never tune on test. max_length 128, dynamic padding, batches grouped by length.
- **Hyperparameters:** AdamW, lr 5e-5 (also 3e-5 and 8e-5), weight decay 0.01, 6% warmup
  then linear decay, batch 32, 1 to 2 epochs, fixed seed. Few-shot (256 examples): 10 to 20
  epochs, or SetFit.
- **Smoke test first:** 50 steps. Confirm the loss falls and matches a CPU run within noise.
- **Keeping the benchmark machine usable:** never overlap training with timing runs; train
  on AC power with the lid open under `caffeinate -i`; cap memory with
  `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.7`; wait 15 to 20 min after training, or until
  powermetrics shows idle temperatures, before benchmarking. Or train on free Colab or
  Kaggle and download the checkpoint.
