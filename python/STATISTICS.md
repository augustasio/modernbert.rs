# Statistics for rung 0

`stats.py` computes these numbers from saved files. It does not load the model. Output goes
to `results/rung0_stats.json`.

```sh
python/.venv/bin/python python/calibration.py   # once: fits calibration (one model load)
python/.venv/bin/python python/stats.py          # about 3 seconds, numpy only
```

## Results

n = 7,600. All intervals are 95%.

| | raw | calibrated |
|---|---|---|
| accuracy | 71.58% [70.59, 72.57] | 84.78% [83.97, 85.59] |
| accuracy, Wilson | [70.55, 72.58] | [83.95, 85.57] |
| macro-F1 | 62.22% [61.70, 62.75] | 84.01% [83.18, 84.82] |
| recall Business | 94.05% [92.94, 95.10] | 91.53% [90.24, 92.76] |
| recall World | 93.37% [92.21, 94.46] | 92.26% [91.04, 93.45] |
| recall Sports | 98.89% [98.39, 99.34] | 99.16% [98.73, 99.53] |
| recall Tech | 0.00% [0.00, 0.00], Wilson [0.00, 0.20] | 56.16% [53.92, 58.42] |

Calibrated minus raw: +13.20 pp, paired bootstrap [+12.38, +14.04]. Exact McNemar test:
90 examples are right raw and wrong calibrated, 1,093 are the reverse, p = 1.3e-219.

## What each number is

**Raw** is rung 0's prediction: the argmax of the four verbalizer logits at the mask
position. Raw logits are the oracle. Every later rung must match them, and rung agreement
is always measured on raw logits.

**Calibrated** is the same logits after contextual calibration (`calibration.py`, Zhao et
al. 2021). The transform is post-processing on four numbers per example. It is fitted on
three content-free prompts only, never on test examples or labels, so the result is still
zero-shot. Its parameters are in `results/calibration.json`, so a Rust rung can apply the
same transform to its own logits.

**Accuracy** is the fraction of the 7,600 examples predicted correctly.

**Recall** for a class is the fraction of that class's 1,900 gold examples predicted as
that class.

**Macro-F1** is the unweighted mean of the four per-class F1 scores. If a class is never
predicted, its precision is 0/0. We set its F1 to 0 (the scikit-learn `zero_division=0`
convention). This applies to raw Tech, which is never predicted. It is why raw macro-F1 is
far below raw accuracy.

## How the intervals are computed

**Bootstrap.** We resample the 7,600 examples with replacement, 10,000 times, with seed
20261101. We compute every metric on each resample. The interval is the 2.5th to 97.5th
percentile of the 10,000 values (the percentile method).

**Wilson.** This is the closed-form score interval for a binomial proportion. We report it
for accuracy and for each recall, as a cross-check on the bootstrap. The two agree to within
0.1 pp everywhere except one case, below.

**Raw Tech recall.** The bootstrap interval is [0, 0]. This is a limit of the bootstrap, not
a claim of certainty. Raw Tech has zero correct predictions, and resampling cannot create
successes that do not exist. The Wilson interval [0.00%, 0.20%] is the meaningful bound
here: with 0 of 1,900, the true recall is very probably below 0.2%.

## Why the comparison is paired

Raw and calibrated predictions are made on the same 7,600 examples. Their errors are
strongly correlated: an article that is hard for one decision rule is usually hard for the
other. An unpaired test treats the two accuracies as independent samples. That overstates
the variance of the difference and gives a wrong interval.

We handle the pairing in two ways:

- **Paired bootstrap.** Each resample draws one set of indices and scores raw and
  calibrated on those same indices. The difference is computed per resample.
- **Exact McNemar test.** Examples where both rules agree carry no information about which
  rule is better. Only discordant pairs count: b = raw right and calibrated wrong, c = the
  reverse. Under the null hypothesis of equal accuracy, each discordant pair is a fair
  coin, so min(b, c) follows Binomial(b + c, 1/2). We sum the exact two-sided tail with
  integer arithmetic, so no normal approximation is used.

## Intervals wider than ±1.5 pp

One: **calibrated Tech recall, ±2.25 pp.** This is expected. A per-class recall is measured
on 1,900 examples, not 7,600. The binomial variance p(1 − p) is also at its largest near
p = 0.5, and this recall is 0.56. The standard error is sqrt(0.56 × 0.44 / 1900) = 1.14 pp,
so the 95% half-width is about 2.2 pp. A narrower interval would need more Sci/Tech
examples, and the test split has no more.

All overall numbers (accuracy and macro-F1, raw and calibrated) are within ±1.0 pp.

## What the intervals cover, and what they do not

The intervals cover sampling variation over test examples. They do not cover variation over
prompt templates, verbalizer choices or content-free strings. Each of these is fixed by
design, and a different choice would give a different number. Raw logits are deterministic
on this machine, so there is no run-to-run noise to report.

## Rungs 1 to 3: agreement with rung 0

`ladder.compare_with_rung0` computes these for every later rung, from that rung's own
saved logits (`results/rungN_logits.npy`). Output goes to `results/rungN.json`.

**Prediction agreement** is the number of the 7,600 examples where the rung's raw argmax
equals rung 0's raw argmax. It has a Wilson interval, like accuracy. Agreement is always
measured on raw logits, never on calibrated ones.

**Rule of three.** If a rung disagrees on zero examples, the Wilson interval still has a
width, but a simpler bound is easier to read: with zero events in n trials, 3/n is an
approximate 95% upper bound on the true event rate. For n = 7,600, that is 3.9e-4. It says
the rung's true disagreement rate is very probably below 1 in 2,500. It does not say the
rate is zero.

**Disagreements** are listed one by one, with the gold label and both rungs' four logits.
They are not filtered or explained.

**Absolute logit difference.** For all 7,600 x 4 verbalizer logits we take
|logit_rung − logit_rung0| and report the median, the 99th percentile and the maximum. Two
rungs can agree on every prediction while their logits drift apart; this catches that.
Small differences (around 1e-5) are expected even for a correct rung: computing the head
for one row instead of all rows changes how BLAS blocks the matmul, which changes the
floating-point summation order.

**Accuracy** for each rung uses the same functions as rung 0 (`stats.bootstrap`,
`stats.point_and_interval`, `stats.wilson_interval`), with the same seed and number of
resamples. If a rung's predictions equal rung 0's, its intervals are identical to rung 0's.

**Calibration per rung.** Each rung fits `calibration.fit` from its own logits on the
three content-free prompts, never from rung 0's parameters, and reports the largest
absolute and relative difference of its `p_cf` from rung 0's. The relative difference
matters most for Tech, whose `p_cf` is about 7e-4: a small absolute shift there is a large
relative one. We also report how many calibrated predictions match rung 0's calibrated
predictions. This is extra information, not the correctness criterion.
