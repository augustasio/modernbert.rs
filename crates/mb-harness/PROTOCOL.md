# Worker protocol

Every variant on the ladder, Rust or Python, runs as a **worker**: a separate process that
loads the model once and then answers requests on stdin and stdout. The harness starts the
workers, sends the requests, and records what comes back. One protocol for every variant
means one measurement method for every variant.

Messages are JSON, one object per line, UTF-8. A worker writes nothing else to stdout;
logging goes to stderr.

## Start

The harness starts the worker with these environment variables set, so thread counts are
explicit and equal across variants:

| Variable | Meaning |
|---|---|
| `MB_THREADS` | the thread count for this run |
| `VECLIB_MAXIMUM_THREADS` | caps Accelerate, for every variant that calls it |
| `OMP_NUM_THREADS` | caps OpenMP, which PyTorch's CPU backend uses |
| `MB_WEIGHTS` | the checkpoint directory |

When the model is loaded and ready, the worker prints exactly one line:

```json
{"ready": true, "variant": "rung7", "backend": "accelerate", "threads": 4, "load_seconds": 2.31}
```

`backend` names the matmul library actually in use, so a silently missing Accelerate link
shows up in the results instead of hiding in them.

### The floor, for memory runs only

`python/memory_run.py` also starts each worker once with `MB_STOP_BEFORE_LOAD` set. The
worker then imports what its rung imports and sets its threads, prints one line and exits
with status 0, before it reads any weights:

```json
{"floor": true, "variant": "rung7"}
```

The process's peak RSS is then the runtime alone, PLAN.md's floor RSS. The benchmark
harness never sets this variable.

## Run one example

```json
{"cmd": "run", "example": 1659, "mask": 152, "ids": [50281, 1401, ...]}
```

The harness sends the token ids. Workers never read ids from a file, so every variant
receives byte-identical input from the same process, which is hard rule 1 by construction.

The worker answers:

```json
{"example": 1659, "seconds": 0.4812, "logits": [22.24, 20.30, 16.03, 19.05]}
```

- `seconds` is the worker's own timing of the forward pass alone, from the parsed ids to the
  four logits, measured with a monotonic clock. Parsing the request and writing the answer
  are outside it.
- `logits` are the four verbalizer logits in rung 0's order: Business, World, Sports, Tech.
  The harness checks them against rung 0 during the benchmarked run (hard rule 5), so a
  variant cannot get fast by computing something else.

## Run a batch

Only rung 0 (`python/worker.py --rung 0`) and rung 14 (`--features batch`) accept this.
Any other worker fails on it as an unknown command.

```json
{"cmd": "run_batch", "examples": [1659, 20], "masks": [152, 140], "ids": [[50281, ...], [50281, ...]]}
```

The worker answers with one row of four logits per example, in the request's order, and
one time for the whole batch:

```json
{"examples": [1659, 20], "seconds": 0.9301, "logits": [[22.24, 20.30, 16.03, 19.05], [...]]}
```

`seconds` covers the forward pass of the whole batch, as for `run`. Rung 0 pads the batch on
the right to its longest prompt, with an attention mask, as HF batches. Rung 14 lays the
prompts one after another without padding. The harness sends a slice as batches of the
config's `batch_size`, the same groups to both workers, so each batch is a pair; it checks
every example's logits against rung 0 as for `run`.

## Stop

```json
{"cmd": "quit"}
```

The worker exits with status 0. Any other exit, or a malformed line, fails the run.

## Errors

A worker that cannot answer prints `{"error": "message"}` and exits non-zero. The harness
stops the run and reports the message. It never retries silently, because a retried
sample is a different sample.
