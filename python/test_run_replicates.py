"""run_replicates.py resumes, and reruns a replicate the Mac slept through."""

import json
import sys

import pytest

import run_replicates


@pytest.fixture
def fake(tmp_path, monkeypatch):
    """A fake harness that writes a result file and reports how long the Mac slept."""
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"seed": 100, "label": "test"}))
    out = tmp_path / "out"
    calls = []
    sleeps = []

    def run_harness(config_file, result_file):
        calls.append(json.load(open(config_file))["seed"])
        open(result_file, "w").write("{}")
        return 0, sleeps.pop(0) if sleeps else 0.0

    monkeypatch.setattr(run_replicates, "run_harness", run_harness)
    monkeypatch.setattr(run_replicates, "wait_for_nominal", lambda: 0.0)

    def run(n):
        monkeypatch.setattr(sys, "argv", ["run_replicates.py", str(config), str(n), str(out)])
        run_replicates.main()
        return json.load(open(out / "manifest.json"))

    return run, calls, sleeps, out


def test_a_second_run_skips_finished_replicates(fake):
    run, calls, _, _ = fake
    run(2)
    manifest = run(3)
    assert calls == [100, 101, 102]
    assert [r["replicate"] for r in manifest["replicates"]] == [0, 1, 2]


def test_a_gzipped_result_counts_as_finished(fake):
    run, calls, _, out = fake
    run(1)
    (out / "replicate_0.json").rename(out / "replicate_0.json.gz")
    run(1)
    assert calls == [100]


def test_a_cut_off_replicate_runs_again(fake):
    run, calls, _, out = fake
    run(2)
    manifest = json.load(open(out / "manifest.json"))
    manifest["replicates"].pop()
    (out / "manifest.json").write_text(json.dumps(manifest))
    run(2)
    assert calls == [100, 101, 101]


def test_a_replicate_the_mac_slept_through_is_discarded_and_rerun(fake):
    run, calls, sleeps, _ = fake
    sleeps.extend([0.0, 600.0])
    manifest = run(2)
    assert calls == [100, 101, 101]
    assert [d["replicate"] for d in manifest["discarded"]] == [1]
    assert manifest["discarded"][0]["slept_seconds"] == 600
    assert len(manifest["replicates"]) == 2


def test_an_invalid_replicate_is_never_rerun(fake, tmp_path):
    run, calls, _, out = fake
    run(1)
    manifest = json.load(open(out / "manifest.json"))
    manifest["replicates"][0]["exit_status"] = 1
    (out / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(SystemExit, match="look at it"):
        run(1)
    assert calls == [100]
