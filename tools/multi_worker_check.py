"""Rough check on a cloud pod (not the paired harness): N workers of T threads side by side, closed loop.

usage: multi.py <worker> <workers> <threads> <batch> <calls_per_worker> <pin:0 none|1 taskset|2 OMP_PLACES> [ENV=V ...]
Prints one JSON line: aggregate examples/s, per-call p50/p90/p99/max (ms), calls over 1 s.
"""
import json, os, subprocess, sys, threading, time, random

MASK = 50284
worker_bin, n, t, b, calls, pin = sys.argv[1], *map(int, sys.argv[2:7])
extra = dict(a.split("=", 1) for a in sys.argv[7:])

rows = []
for line in open("data/agnews_test_prompt_ids.tsv"):
    i, _, ids = line.rstrip("\n").split("\t")
    ids = [int(x) for x in ids.split()]
    rows.append((int(i), ids.index(MASK), ids))

# Physical cores we may use: first CPU of each sibling group, in order.
allowed = sorted(os.sched_getaffinity(0))
cores, seen = [], set()
for c in allowed:
    sib = open(f"/sys/devices/system/cpu/cpu{c}/topology/thread_siblings_list").read().strip()
    if sib not in seen:
        seen.add(sib)
        cores.append(c)

def start(k):
    env = dict(os.environ, MB_WEIGHTS="weights/ModernBERT-Large-Instruct", MB_THREADS=str(t),
               OMP_NUM_THREADS=str(t), MKL_NUM_THREADS=str(t), MKL_THREADING_LAYER="GNU", **extra)
    cmd = worker_bin.split()
    if pin == 2:  # OpenMP binds each thread to one of this worker's cores; no taskset
        mine = cores[k * t:(k + 1) * t]
        env.update(OMP_PLACES=",".join("{%d}" % c for c in mine), OMP_PROC_BIND="close")
    elif pin:
        mine = cores[k * t:(k + 1) * t]
        cmd = ["taskset", "-c", ",".join(map(str, mine))] + cmd
    p = subprocess.Popen(cmd, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
    assert json.loads(p.stdout.readline())["ready"]
    return p

def call(p, batch):
    if b == 1:
        i, m, ids = batch[0]
        p.stdin.write(json.dumps({"cmd": "run", "example": i, "mask": m, "ids": ids}) + "\n")
    else:
        p.stdin.write(json.dumps({"cmd": "run_batch", "examples": [r[0] for r in batch],
                                  "masks": [r[1] for r in batch], "ids": [r[2] for r in batch]}) + "\n")
    return json.loads(p.stdout.readline())["seconds"]

procs = [start(k) for k in range(n)]
times = [[] for _ in range(n)]
barrier = threading.Barrier(n + 1)

def run(k):
    rng = random.Random(1000 + k)
    sched = rng.sample(rows, (calls + 3) * b)
    for w in range(3):  # warm-up
        call(procs[k], sched[w * b:(w + 1) * b])
    barrier.wait()
    for c in range(3, calls + 3):
        times[k].append(call(procs[k], sched[c * b:(c + 1) * b]))

ths = [threading.Thread(target=run, args=(k,)) for k in range(n)]
for th in ths: th.start()
barrier.wait()
t0 = time.perf_counter()
for th in ths: th.join()
wall = time.perf_counter() - t0
for p in procs:
    p.stdin.write('{"cmd": "quit"}\n'); p.stdin.flush(); p.wait()

all_t = sorted(x * 1000 for ts in times for x in ts)
q = lambda f: all_t[min(len(all_t) - 1, int(f * len(all_t)))]
print(json.dumps({"workers": n, "threads": t, "batch": b, "pin": pin, "env": extra,
                  "examples_per_s": round(n * calls * b / wall, 2), "p50": round(q(.5)), "p90": round(q(.9)),
                  "p99": round(q(.99)), "max": round(all_t[-1]), "over_1s": sum(x > 1000 for x in all_t),
                  "calls": len(all_t), "load": open("/proc/loadavg").read().split()[0]}), flush=True)
