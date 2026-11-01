# Why rung 0 never predicts `Tech`

Rung 0 never picks verbalizer index 3. Every gold Sci/Tech example goes to `World` or
`Business`. The other three classes sit almost perfectly on the diagonal.

**Diagnosis: not a bug. The prompt, the mask position and the verbalizer ids are all
correct. The Sci/Tech class loses to a class prior that exists before the model sees any
news report, and no choice of verbalizer word fixes it.**

**The prompt did not change, so `python/reference/activations.safetensors` and
`fixtures.json` remain valid. Do not regenerate them.**

Evidence collected by `diagnose_tech.py`. Re-run it to reproduce every number below.

## What was ruled out

### The prompt is byte-identical to PLAN.md

Compared against the fenced template block extracted from `PLAN.md` itself, not against a
second copy. 408 characters on both sides, identical. The four trailing spaces after
`** Instructions **` are present: the first line is `'** Instructions **    '`.

### The mask position is the right slot

One `[MASK]` (50284), one `[unused0]` (50285). On the first test example the final tokens
are:

```
['Ġ**', 'Ċ', 'Answer', ':', 'Ġ', '[unused0]', '[MASK]', '[SEP]']
```

`[MASK]` sits immediately after `[unused0]`, which is correct: `[MASK]` has `lstrip: true`
and swallows the space that would otherwise precede the answer token, so it occupies
exactly the position the answer verbalizer occupied in training. With one mask in the
prompt, first and last occurrence are the same index, so reading it agrees with the
reference harness, which takes the last.

### The verbalizer ids are the space-prefixed forms

Resolved by encoding, then asserted: `ĠBusiness` 10518, `ĠWorld` 3645, `ĠSports` 15000,
`ĠTech` 18447. These are the ids `PLAN.md` pins.

### The gold label remap is right

Verified independently by the coordinator: the dataset's order is
`['World', 'Sports', 'Business', 'Sci/Tech']`, the verbalizer order is
`['Business', 'World', 'Sports', 'Tech']`, and the raw first-64 distribution maps exactly
onto the remapped gold.

## What the model actually wants to say

Unrestricted argmax over the full 50,368-wide vocabulary at the mask position, on gold
Sci/Tech examples. The task is clearly understood: every top candidate is a news section
name.

```
example 1  ĠWorld 21.95, ĠBusiness 21.31, ĠSports 19.44, ĠTech 18.05, ĠTechnology 18.04,
           ĠSpace 17.19, ĠScience 15.83, ĠNews 15.22, ĠTechnical 14.41, ĠCompany 14.28
example 3  ĠWorld 23.60, ĠSports 20.55, ĠBusiness 19.19, ĠNews 16.59, ĠHealth 13.94,
           ĠPress 13.92, ĠScience 13.75, ĠWeather 13.73, ĠTechnology 13.55, ĠEnvironment 13.36
example 6  ĠWorld 21.88, ĠBusiness 20.23, ĠTech 16.66, ĠTechnology 16.21, ĠSports 15.57,
           ĠNews 15.15, ĠVirus 14.76, ĠSecurity 14.75, ĠHealth 14.34, ĠScience 13.50
```

`ĠTech` is 4th of the four verbalizers on 5 of the 6 examples inspected, and 3rd on the
sixth. It is not nowhere: on examples 1 and 6 it is within 3 to 5 logits. But it never
wins.

Note example 3: `ĠTech` at 13.21 sits below `ĠNews`, `ĠHealth`, `ĠPress`, `ĠWeather` and
`ĠEnvironment`. The model is not reaching for a technology word and missing. It is
preferring high-frequency section names.

## No verbalizer choice helps

Eight single-token candidates for the Sci/Tech class, scored over a balanced sample of 50
examples per class, seed 0, with `Business`, `World` and `Sports` held fixed. `Sci/Tech`
itself is excluded because it is three tokens (`ĠSci`, `/`, `Tech`), which is why the
cookbook remapped it in the first place.

| 4th verbalizer | overall accuracy | Tech recall | times predicted |
|---|---|---|---|
| `Tech` | 0.7200 | 0.0000 | 0 |
| `Technology` | 0.7200 | 0.0000 | 0 |
| `Science` | 0.7200 | 0.0000 | 0 |
| `Sci` | 0.7200 | 0.0000 | 0 |
| `Computer` | 0.7200 | 0.0000 | 0 |
| `Internet` | 0.7200 | 0.0000 | 0 |
| `Software` | 0.7200 | 0.0000 | 0 |
| `Scientific` | 0.7200 | 0.0000 | 0 |

Every candidate gives the identical result and zero predictions. Mean logits over the same
sample show why:

```
Business    21.873      Tech        13.965
World       22.815      Technology  14.213
Sports      20.449      Science     13.033
                        Sci          6.234
                        Computer    10.043
                        Internet     9.555
                        Software     7.783
                        Scientific   8.698
```

The best candidate, `Technology` at 14.21, is still 6.2 logits below `Sports`, the weakest
of the other three. The gap is far larger than the spread among candidates, so swapping
the word cannot close it.

## The cause: a class prior, measured directly

Score the same prompt with a content-free content slot. The model has no news report to go
on, so whatever it prefers here is prior, not evidence.

```
content 'N/A'   Business 21.01, World 21.15, Sports 20.60, Tech 14.41
content ''      Business 22.28, World 22.50, Sports 21.44, Tech 14.71
content 'none'  Business 17.31, World 18.06, Sports 17.31, Tech 12.17

mean bias       Business 20.20, World 20.57, Sports 19.79, Tech 13.76
```

**With no article at all, `Tech` is already 6.0 to 6.8 logits below the other three, and
the other three are within 0.8 of each other.** That is the entire effect. The Sci/Tech
option is handicapped before the model reads anything, and the margin the content supplies
is smaller than the handicap.

Subtracting that content-free bias, which is contextual calibration and the standard
treatment for exactly this failure, on the same balanced sample:

| | accuracy | Business | World | Sports | Tech |
|---|---|---|---|---|---|
| uncalibrated | 0.7200 | 0.94 | 0.96 | 0.98 | **0.00** |
| calibrated | **0.8700** | 0.90 | 0.94 | 0.98 | **0.66** |

Tech recall goes from 0 to 0.66 and overall accuracy from 0.72 to 0.87, while the other
three classes lose almost nothing.

The full 7,600-example run confirms it. Uncalibrated (`results/rung0.json`): accuracy
0.7158, and `Tech` is predicted 0 times out of 7,600. Of the 1,900 gold Sci/Tech examples,
1,278 go to `Business` and 608 to `World`.

The probe above subtracts a mean of raw logits. The adopted calibration is the exact form
from Zhao et al. (2021): it divides by the mean content-free probability
(`calibration.py`). On the full set it gives **84.78% [83.97, 85.59]**, with Tech recall
56.16% [53.92, 58.42]. The cruder logit-mean probe gives 85.41% on the same logits. The two
forms are close, but only the Zhao form is reported. Full statistics are in
[STATISTICS.md](STATISTICS.md) and `results/rung0_stats.json`. So the model does carry the information needed to
classify these examples. Only the decision rule throws it away.

## Why this happens

The three winning verbalizers are the canonical newspaper section names. `World`,
`Business` and `Sports` are overwhelmingly common in that exact role in pretraining text.
`Tech` as a section heading is much rarer, and the distribution reflects that.

The `Sci/Tech` to `Tech` remap comes from the cookbook's own `agnews_label_map.json`, but
it was built for **fine-tuning**, not zero-shot. `docs/research/prompt-template.md` is
explicit: the published AG News number of 95.24 is a two-epoch full fine-tune, and the
zero-shot table covers only MMLU, MMLU-Pro, ADEv2, NIS and OSE. Two epochs of training
teach the model to emit `ĠTech` at this position, which flattens exactly the prior
measured above. Zero-shot there is nothing to flatten it.

So the verbalizer is not wrong. It was chosen under a setting that removes the problem it
has here.

## What the reference harness does

From `docs/research/prompt-template.md`:

- **It never runs zero-shot AG News.** AG News appears only in `train.py`, as a fine-tune.
  There is no published zero-shot AG News number for this model, so there is nothing this
  figure is failing to reproduce.
- `eval.py` (the MMLU harness, which is where the published zero-shot numbers come from)
  restricts scoring to the candidate ids, applies softmax over the full vocabulary first,
  then sums the probabilities of duplicate surface forms. It takes the **last** mask
  occurrence. It applies **no calibration**.
- The model card and blog take an unrestricted argmax over the whole vocabulary and decode
  the result, with no candidate restriction at all.

Note that MMLU uses `A`/`B`/`C`/`D`, whose priors are far more even than four section
names. The harness never had to deal with this, which is consistent with it carrying no
calibration step.

## Recommendation

This is a method decision, so it is not taken here. Rung 0 as it stands reports the
uncalibrated argmax over the four pinned verbalizers, unchanged.

In preference order:

1. **Report the uncalibrated number as the baseline and keep it.** It is honest, it is
   what the pinned prompt and verbalizers give, and `PLAN.md` already says zero-shot AG
   News accuracy is whatever it is. The project measures agreement between
   implementations, not accuracy, so a 0.72 oracle is exactly as useful as a 0.87 one.
   Every rung still has to match it to four decimal places. **This costs nothing and is
   the only option that needs no new decision.**
2. **Add calibration as a documented, separately reported variant**, not as a change to
   rung 0. One extra forward pass on a content-free prompt, three lines of arithmetic,
   identical cost per example, and it moves accuracy from 0.72 to 0.87. It would make the
   headline accuracy defensible as a zero-shot figure. It must be a variant rather than an
   edit, because changing rung 0 changes the oracle every other rung is asserted against,
   and `CLAUDE.md` hard rule 2 wants one variable at a time.
3. **Do not swap the verbalizer.** It is measured above to make no difference whatsoever,
   and it would invalidate the pinned ids in `PLAN.md` and the asserts that protect them.

Whichever is chosen, two things should be recorded next to the accuracy number: that it is
not comparable to the fine-tuned 95.24, and that roughly a quarter of the test set is a
class the uncalibrated decision rule cannot select.

## Reproducing

```sh
python/.venv/bin/python python/diagnose_tech.py
```

One model instance, fp32, batch 1. Roughly two minutes.
