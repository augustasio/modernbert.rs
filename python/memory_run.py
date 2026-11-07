"""Memory and cold start for every rung, one fresh process per measurement.

    python/.venv/bin/python python/memory_run.py 4 results/memory/t4            # every rung
    python/.venv/bin/python python/memory_run.py 4 results/memory/t4 rung12     # only these

Memory runs and timing runs cannot be the same runs (PLAN.md, "Three measurement
hazards"): the allocator's high-water mark never shrinks, so a clean peak needs a fresh
process. So this script never reuses a worker. For each rung it starts 5 pairs of fresh
processes, through the same worker protocol as the benchmark
(crates/mb-harness/PROTOCOL.md), with the same environment the harness gives them:

1. A **floor** process, with MB_STOP_BEFORE_LOAD set. The worker imports what its rung
   imports, then exits before reading any weights. Its peak RSS from wait4 is the
   runtime alone: PLAN.md's "floor RSS".
2. A **full** process, which loads, answers a fixed list of examples, and quits. From it:
   - **cold start**: from just before the spawn to the first reply, on the wall clock;
   - **post-load RSS**: the worker's resident size when it says `ready` (it is then idle,
     waiting on stdin, so the reading is exact);
   - **activation delta**: the highest resident size sampled every 10 ms while it
     answers, minus post-load;
   - **peak RSS** and **major page faults**, from wait4 when it exits: the kernel's own
     high-water mark, which sampling cannot miss.

Every answer is checked against rung 0's logits, with the benchmark's gate of 1e-2
(hard rule 5).

Rung 2 compiles with inductor, which caches compiled kernels on disk across processes. Its
"rung2" row gets a fresh, empty TORCHINDUCTOR_CACHE_DIR per process, so its cold start is
truly cold; "rung2-warm" shares one cache, filled by one process that is not recorded
(PLAN.md, changelog).

The file cache is not emptied between processes: that needs `sudo purge`. So every cold
start here reads the weights from a warm file cache, and the figure is process start-up,
load and the first example, not disk speed.

Each rung's records go to <out>/<name>.json as soon as its 5 pairs finish. A rung whose
file exists is skipped, so the script resumes where it stopped.
"""

import ctypes
import json
import os
import random
import subprocess
import sys
import tempfile
import threading
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "tools"))
from make_bench_configs import VARIANTS

WEIGHTS = "weights/ModernBERT-Large-Instruct"
PROMPT_IDS = "data/agnews_test_prompt_ids.tsv"
REFERENCE_LOGITS = "results/rung0_logits.npy"
MASK_ID = 50284
LOGIT_GATE = 1e-2
PROCESSES = 5
SAMPLE_SECONDS = 0.010

# The longest AG News prompt (348 tokens), so the activation peak is the largest a
# batch-1 request can need, plus a seeded sample. Rung 6 takes about 24 s an example, so
# it answers two.
LONGEST_EXAMPLE = 5181
EXAMPLES = 16
RUNG6_EXAMPLES = 2
SEED = 0

ORDER = [
    "rung0",
    "rung1",
    "rung2",
    "rung2-warm",
    "rung3",
    "rung4",
    "rung5",
    "rung6",
    "rung7",
    "rung8",
    "rung9",
    "rung10",
    "rung11",
    "rung12",
    "rung13",
]


def command_for(name: str) -> list[str]:
    """The benchmark's worker command, but with the Rust workers from target/mem.

    The memory builds add MB_STOP_BEFORE_LOAD and nothing else. They live in their own
    folders so the binaries the timing sessions ran stay as they were:

        cargo build --release -p mb-model --features f32-math --bin worker \\
            --target-dir target/mem/rung12      (and so on, as in make_bench_configs.py)
    """
    command = VARIANTS["aa" if name == "rung0" else name.removesuffix("-warm")][1]
    return [part.replace("target/bench/", "target/mem/") for part in command]


# --------------------------------------------------------------------------------------
# Another process's resident size, from libproc. ps would fork a process per sample.
# --------------------------------------------------------------------------------------


class RusageInfoV0(ctypes.Structure):
    """struct rusage_info_v0 from <sys/resource.h>."""

    _fields_ = [
        ("ri_uuid", ctypes.c_uint8 * 16),
        ("ri_user_time", ctypes.c_uint64),
        ("ri_system_time", ctypes.c_uint64),
        ("ri_pkg_idle_wkups", ctypes.c_uint64),
        ("ri_interrupt_wkups", ctypes.c_uint64),
        ("ri_pageins", ctypes.c_uint64),
        ("ri_wired_size", ctypes.c_uint64),
        ("ri_resident_size", ctypes.c_uint64),
        ("ri_phys_footprint", ctypes.c_uint64),
        ("ri_proc_start_abstime", ctypes.c_uint64),
        ("ri_proc_exit_abstime", ctypes.c_uint64),
    ]


LIBPROC = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
RUSAGE_INFO_V0 = 0


def resident(pid: int) -> tuple[int, int] | None:
    """(resident bytes, physical footprint bytes) of pid, or None once it has gone."""
    info = RusageInfoV0()
    if LIBPROC.proc_pid_rusage(pid, RUSAGE_INFO_V0, ctypes.byref(info)) != 0:
        return None
    return info.ri_resident_size, info.ri_phys_footprint


class Sampler:
    """Polls a process's resident size every 10 ms on a thread, keeping the highest."""

    def __init__(self, pid: int):
        self.pid = pid
        self.peak_resident = 0
        self.peak_footprint = 0
        self.samples = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            reading = resident(self.pid)
            if reading is None:
                return
            self.peak_resident = max(self.peak_resident, reading[0])
            self.peak_footprint = max(self.peak_footprint, reading[1])
            self.samples += 1
            time.sleep(SAMPLE_SECONDS)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *_):
        self._stop.set()
        self._thread.join()


# --------------------------------------------------------------------------------------
# One process
# --------------------------------------------------------------------------------------


def system_memory() -> dict:
    """Free pages, compressed pages and swap in use, from vm_stat and sysctl.

    Under memory pressure macOS compresses or swaps a process's pages, and resident size
    and ru_maxrss then read low. This is recorded with every process so a distorted
    reading can be found, rather than averaged in.
    """
    text = subprocess.run(["vm_stat"], capture_output=True, text=True, check=True).stdout
    page = int(text.split("page size of ")[1].split(" ")[0])
    pages = {}
    for line in text.splitlines()[1:]:
        key, _, value = line.partition(":")
        if value.strip().rstrip(".").isdigit():
            pages[key.strip()] = int(value.strip().rstrip("."))
    swap = subprocess.run(
        ["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True, check=True
    ).stdout
    used = swap.split("used = ")[1].split("M")[0]
    return {
        "free_bytes": pages.get("Pages free", 0) * page,
        "compressed_bytes": pages.get("Pages stored in compressor", 0) * page,
        "swap_used_bytes": int(float(used) * 2**20),
    }


def worker_env(threads: int, extra: dict) -> dict:
    """The harness's worker environment (crates/mb-harness/src/bench.rs, worker_env)."""
    env = dict(os.environ)
    env.update(
        {
            "MB_THREADS": str(threads),
            "VECLIB_MAXIMUM_THREADS": str(threads),
            "OMP_NUM_THREADS": str(threads),
            "MB_WEIGHTS": WEIGHTS,
        }
    )
    env.update(extra)
    return env


def reap(proc: subprocess.Popen) -> dict:
    """Wait for the process with wait4 and return its exit code and kernel rusage.

    ru_maxrss is in bytes on Darwin (kilobytes on Linux).
    """
    _, status, usage = os.wait4(proc.pid, 0)
    proc.returncode = os.waitstatus_to_exitcode(status)
    return {
        "exit_code": proc.returncode,
        "peak_rss_bytes": usage.ru_maxrss,
        "major_faults": usage.ru_majflt,
        "minor_faults": usage.ru_minflt,
    }


def read_line(proc: subprocess.Popen, what: str) -> dict:
    line = proc.stdout.readline()
    if not line:
        raise RuntimeError(f"worker closed stdout before {what}")
    message = json.loads(line)
    if "error" in message:
        raise RuntimeError(f"worker error before {what}: {message['error']}")
    return message


def floor_process(command: list[str], threads: int, extra: dict) -> dict:
    env = worker_env(threads, {**extra, "MB_STOP_BEFORE_LOAD": "1"})
    proc = subprocess.Popen(command, cwd=REPO, env=env, stdout=subprocess.PIPE, text=True)
    message = read_line(proc, "floor")
    if not message.get("floor"):
        raise RuntimeError(f"expected a floor line, got {message}")
    proc.stdout.close()
    record = reap(proc)
    if record["exit_code"] != 0:
        raise RuntimeError(f"floor process exited {record['exit_code']}")
    return {"variant": message["variant"], **record}


def full_process(command, threads, extra, examples, reference) -> dict:
    env = worker_env(threads, extra)
    started = time.perf_counter()
    proc = subprocess.Popen(
        command, cwd=REPO, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True
    )
    ready = read_line(proc, "ready")
    ready_seconds = time.perf_counter() - started
    post_load = resident(proc.pid)
    if post_load is None:
        raise RuntimeError("worker gone right after ready")

    worst = 0.0
    first_reply_seconds = None
    with Sampler(proc.pid) as sampler:
        for index, (ids, mask) in examples:
            proc.stdin.write(
                json.dumps({"cmd": "run", "example": index, "mask": mask, "ids": ids}) + "\n"
            )
            proc.stdin.flush()
            reply = read_line(proc, f"the reply to example {index}")
            if first_reply_seconds is None:
                first_reply_seconds = time.perf_counter() - started
            if reply.get("example") != index:
                raise RuntimeError(f"reply for {reply.get('example')}, expected {index}")
            gap = float(np.max(np.abs(np.array(reply["logits"]) - reference[index])))
            worst = max(worst, gap)
    proc.stdin.write(json.dumps({"cmd": "quit"}) + "\n")
    proc.stdin.close()
    proc.stdout.close()
    record = reap(proc)
    if record["exit_code"] != 0:
        raise RuntimeError(f"worker exited {record['exit_code']}")
    if worst > LOGIT_GATE:
        raise RuntimeError(f"worst logit difference {worst} is above the gate {LOGIT_GATE}")
    return {
        "variant": ready.get("variant"),
        "backend": ready.get("backend"),
        "load_seconds": ready.get("load_seconds"),
        "compile_seconds": ready.get("compile_seconds"),
        "ready_seconds": ready_seconds,
        "cold_start_seconds": first_reply_seconds,
        "post_load_rss_bytes": post_load[0],
        "post_load_footprint_bytes": post_load[1],
        "inference_peak_rss_bytes": max(sampler.peak_resident, post_load[0]),
        "inference_peak_footprint_bytes": max(sampler.peak_footprint, post_load[1]),
        "activation_delta_bytes": max(sampler.peak_resident, post_load[0]) - post_load[0],
        "samples": sampler.samples,
        "examples": len(examples),
        "worst_logit_difference": worst,
        **record,
    }


# --------------------------------------------------------------------------------------
# One rung
# --------------------------------------------------------------------------------------


def load_examples() -> list[tuple[list[int], int]]:
    out = []
    with open(os.path.join(REPO, PROMPT_IDS)) as handle:
        for line in handle:
            ids = [int(t) for t in line.rstrip("\n").split("\t")[2].split(" ")]
            mask = len(ids) - 1 - ids[::-1].index(MASK_ID)
            out.append((ids, mask))
    return out


def workload(name: str, all_examples) -> list:
    count = RUNG6_EXAMPLES if name == "rung6" else EXAMPLES
    others = [i for i in range(len(all_examples)) if i != LONGEST_EXAMPLE]
    chosen = [LONGEST_EXAMPLE, *random.Random(SEED).sample(others, count - 1)]
    return [(i, all_examples[i]) for i in chosen]


def run_rung(name, threads, all_examples, reference) -> dict:
    command = command_for(name)
    examples = workload(name, all_examples)
    records = []
    with tempfile.TemporaryDirectory(prefix="mb-inductor-") as shared_cache:
        if name == "rung2-warm":
            # Fill the shared cache once; this process is not recorded.
            full_process(
                command, threads, {"TORCHINDUCTOR_CACHE_DIR": shared_cache}, examples[:1], reference
            )
        for k in range(PROCESSES):
            with tempfile.TemporaryDirectory(prefix="mb-inductor-") as fresh_cache:
                extra = {}
                if name == "rung2":
                    extra = {"TORCHINDUCTOR_CACHE_DIR": fresh_cache}
                elif name == "rung2-warm":
                    extra = {"TORCHINDUCTOR_CACHE_DIR": shared_cache}
                before = system_memory()
                floor = floor_process(command, threads, extra)
                full = full_process(command, threads, extra, examples, reference)
            records.append({"process": k, "system_before": before, "floor": floor, "full": full})
            print(
                f"  {name} process {k + 1} of {PROCESSES}: peak "
                f"{full['peak_rss_bytes'] / 2**20:.0f} MiB, cold start "
                f"{full['cold_start_seconds']:.2f} s",
                flush=True,
            )
    return {
        "name": name,
        "command": command,
        "threads": threads,
        "processes": PROCESSES,
        "example_indices": [i for i, _ in examples],
        "logit_gate": LOGIT_GATE,
        "records": records,
    }


def main() -> None:
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    threads = int(sys.argv[1])
    out_dir = os.path.join(REPO, sys.argv[2])
    names = sys.argv[3:] or ORDER
    unknown = [n for n in names if n not in ORDER]
    if unknown:
        sys.exit(f"unknown rungs {unknown}; known: {ORDER}")
    os.makedirs(out_dir, exist_ok=True)

    all_examples = load_examples()
    reference = np.load(os.path.join(REPO, REFERENCE_LOGITS))
    for name in names:
        path = os.path.join(out_dir, f"{name}.json")
        if os.path.exists(path):
            print(f"== {name}: already done, skipping", flush=True)
            continue
        print(f"== {name}, {time.strftime('%H:%M:%SZ', time.gmtime())}", flush=True)
        try:
            result = run_rung(name, threads, all_examples, reference)
        except Exception as error:  # noqa: BLE001  (one rung's failure stops only itself)
            print(f"== {name} FAILED: {type(error).__name__}: {error}", flush=True)
            continue
        result["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        tmp = path + ".tmp"
        with open(tmp, "w") as handle:
            json.dump(result, handle, indent=1)
        os.replace(tmp, path)
    print(f"== done, {time.strftime('%H:%M:%SZ', time.gmtime())}", flush=True)


if __name__ == "__main__":
    main()
