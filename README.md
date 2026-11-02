# modernbert.rs

ModernBERT-Large-Instruct inference on a laptop CPU, implemented nine times, measured
carefully, to find out where the time actually goes.

The question is not "is Rust faster". It is how much of a speedup belongs to the language,
how much to Apple's Accelerate matrix units, and how much to work the stock Python
pipeline was simply doing for no reason. So this is a ladder: nine implementations from
stock HF transformers up to a hand-written Rust forward pass, where each rung changes
exactly one variable and every delta can be attributed to something.

Task: AG News topic classification, zero-shot, through ModernBERT's masked-language-model
head. 395M parameters, 7,600 test examples, no fine-tuning.

Measured: batch-1 latency percentiles, a throughput sweep, the p99-against-throughput
frontier, six distinct memory quantities, cold start, and energy per thousand
classifications. On a fanless M4 Air, which throttles, so the protocol interleaves
variants and reports ratios against a concurrently measured baseline.

Written in the spirit of llm.c: small, readable code with no framework abstractions.

See [PLAN.md](PLAN.md) for the full plan, [CLAUDE.md](CLAUDE.md) for conventions, and
[docs/research](docs/research) for the verified architecture and tokenizer findings this
was built against.

## Status

Planning done, scaffold in place. Nothing implemented yet.

## Machine setup for benchmarks

Energy and CPU frequency come from `powermetrics`, which runs only as root. So that
unattended benchmark runs can call it, one sudo rule lets your user run
`/usr/bin/powermetrics` without a password, and nothing else. It was installed on
2026-11-02 at `/etc/sudoers.d/powermetrics`.

**To remove it when the benchmarks are done:**

```bash
sudo rm /etc/sudoers.d/powermetrics
```

To install it again:

```bash
echo "$USER ALL=(root) NOPASSWD: /usr/bin/powermetrics" > /tmp/powermetrics.sudoers
sudo visudo -cf /tmp/powermetrics.sudoers   # must print "parsed OK"
sudo install -m 0440 -o root -g wheel /tmp/powermetrics.sudoers /etc/sudoers.d/powermetrics
```
