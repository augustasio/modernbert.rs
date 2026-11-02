#!/bin/sh
# One benchmark session at one thread count: the A/A floor first, then every rung,
# each as 5 replicates. A config that fails stops only itself; the session goes on,
# and the failure is in its folder's manifest and in the log.
#
#     bench/run_session.sh 4
#
# Results go to results/bench/t<threads>/<name>/. Run nothing else on the Mac meanwhile.
threads=$1
[ -n "$threads" ] || { echo "usage: $0 <threads>"; exit 2; }
cd "$(dirname "$0")/.." || exit 1
for name in aa rung7 rung5 rung4 rung1 rung3 rung2 rung6; do
    echo "== $name, $(date -u +%H:%M:%SZ)"
    python/.venv/bin/python python/run_replicates.py \
        "bench/configs/t${threads}_${name}.json" 5 "results/bench/t${threads}/${name}" \
        || echo "== $name FAILED"
done
echo "== session done, $(date -u +%H:%M:%SZ)"
