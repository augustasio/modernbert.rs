#!/bin/sh
# One benchmark session at one thread count: the A/A floor first, then every rung,
# each as 5 replicates. A config that fails stops only itself; the session goes on,
# and the failure is in its folder's manifest and in the log.
#
#     bench/run_session.sh 4
#     bench/run_session.sh 1 rung1 rung3    # only these
#
# To resume a stopped session, run the same command again: run_replicates.py skips every
# replicate that already finished, and reruns one the Mac slept through.
# Results go to results/bench/t<threads>/<name>/. Run nothing else on the Mac meanwhile.
threads=$1
[ -n "$threads" ] || { echo "usage: $0 <threads> [name ...]"; exit 2; }
shift
names=${*:-aa rung13 rung12 rung11 rung10 rung9 rung8 rung7 rung5 rung4 rung1 rung3 rung2 rung6}
cd "$(dirname "$0")/.." || exit 1
for name in $names; do
    echo "== $name, $(date -u +%H:%M:%SZ)"
    python/.venv/bin/python python/run_replicates.py \
        "bench/configs/t${threads}_${name}.json" 5 "results/bench/t${threads}/${name}" \
        || echo "== $name FAILED"
done
echo "== session done, $(date -u +%H:%M:%SZ)"
