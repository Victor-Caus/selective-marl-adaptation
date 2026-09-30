"""Authorized recovery: preserve interrupted evaluation, rerun unchanged, summarize."""

import csv
import json
import os
import resource
import shutil
import subprocess
import sys
import time
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    os.environ.update(
        PYTHONPATH="src:.external/reference-sources/marlbenchmark-on-policy",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        MPLBACKEND="Agg",
    )
    out = Path("results/selectivity-controls-20260928-r1")
    recovery = out / "recovery-20260928-r1"
    target = out / "variants-s5"
    assert not (target / "COMPLETE").exists(), "Evaluation already completed"
    assert not recovery.exists(), "Recovery already attempted; inspect before retrying"
    for seed in range(1, 6):
        for control in ("continued", "randomized"):
            assert (
                json.loads((out / f"{control}-s{seed}/metadata.json").read_text())["status"]
                == "complete"
            )
            assert (out / f"{control}-eval-s{seed}/COMPLETE").exists()
        if seed < 5:
            assert (out / f"variants-s{seed}/COMPLETE").exists()
    config = json.loads((target / "config.json").read_text())
    assert config["sessions"] == 20 and config["episodes"] == 60
    assert config["seed"] == 12000001 and config["change_range"] == [12, 35]
    recovery.mkdir()
    shutil.move(str(target), str(recovery / "variants-s5-interrupted"))
    shutil.copy2(out / "variants-s5.log", recovery / "variants-s5-interrupted.log")
    shutil.copy2(out / "status.txt", recovery / "original-status.txt")
    command = [
        sys.executable,
        "-u",
        "scripts/evaluate-online-adapters.py",
        "--actor",
        config["actor"],
        "--memory",
        config["memory"],
        "--output",
        str(target),
        "--sessions",
        "20",
        "--episodes",
        "60",
        "--change-range",
        "12",
        "35",
        "--seed",
        "12000001",
        "--modes",
        *config["modes"],
        "--conditions",
        *config["conditions"],
    ]
    (recovery / "command.json").write_text(json.dumps(command, indent=2))
    try:
        (out / "status.txt").write_text("ablation_evaluation seed=5 recovery=1\n")
        with (out / "variants-s5.log").open("w") as log:
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            with (recovery / "memory.jsonl").open("w") as memory:
                while child.poll() is None:
                    try:
                        status = Path(f"/proc/{child.pid}/status").read_text()
                        values = {
                            line.split(":", 1)[0]: line.split(":", 1)[1].strip()
                            for line in status.splitlines()
                            if line.startswith(("VmRSS:", "VmHWM:", "VmSize:"))
                        }
                        memory.write(
                            json.dumps(dict(time=time.time(), pid=child.pid, **values)) + "\n"
                        )
                        memory.flush()
                    except FileNotFoundError:
                        pass
                    time.sleep(30)
            code = child.wait()
        (recovery / "exit.json").write_text(
            json.dumps(
                dict(
                    returncode=code,
                    peak_child_rss_kib=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
                ),
                indent=2,
            )
        )
        if code:
            raise RuntimeError(f"Evaluation exit code {code}; inspect memory.jsonl and log")

        # Check the repeated completed sessions reproduce the preserved partial run.
        def rows(path):
            with path.open() as f:
                return {
                    (r["mode"], r["condition"], r["session"], r["episode"]): float(r["return"])
                    for r in csv.DictReader(f)
                }

        old = rows(recovery / "variants-s5-interrupted/episodes.csv")
        new = rows(target / "episodes.csv")
        assert old.keys() <= new.keys()
        discrepancy = max((abs(value - new[key]) for key, value in old.items()), default=0)
        (recovery / "replay-check.json").write_text(
            json.dumps(
                dict(compared_episodes=len(old), maximum_absolute_difference=discrepancy), indent=2
            )
        )
        assert discrepancy < 1e-8, "Repeated evaluation differs from preserved partial data"
        (out / "status.txt").write_text("summarizing recovery=1\n")
        subprocess.run(
            [sys.executable, "scripts/summarize-selectivity-controls.py", str(out)], check=True
        )
        (out / "status.txt").write_text("complete\n")
    except BaseException as exc:
        (out / "status.txt").write_text(f"failed recovery=1 error={exc!r}\n")
        raise


if __name__ == "__main__":
    main()
