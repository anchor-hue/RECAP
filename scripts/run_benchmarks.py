#!/usr/bin/env python3
"""Run the seven manuscript benchmark pairs for one or more seeds."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", default="2021", help="Comma-separated random seeds")
    p.add_argument("--devices", default="auto", help="Comma-separated devices, e.g. cuda:0,cuda:1")
    p.add_argument("--outroot", type=Path, default=ROOT / "results/benchmark")
    args = p.parse_args()
    seeds = [int(x) for x in args.seeds.split(",")]
    devices = [x.strip() for x in args.devices.split(",") if x.strip()]
    manifest = json.loads((ROOT / "data/benchmark_manifest.json").read_text())
    jobs = [(row["name"], seed) for row in manifest for seed in seeds]
    args.outroot.mkdir(parents=True, exist_ok=True)

    def worker(device: str, assigned: list[tuple[str, int]]) -> None:
        for dataset, seed in assigned:
            outdir = args.outroot / dataset / f"seed_{seed}"
            outdir.mkdir(parents=True, exist_ok=True)
            command = [
                sys.executable, str(ROOT / "run_recap.py"), "benchmark",
                "--dataset", dataset, "--seed", str(seed), "--device", device,
                "--outdir", str(outdir),
            ]
            environment = os.environ.copy()
            environment.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", PYTHONUNBUFFERED="1")
            with (outdir / "run.log").open("w") as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=environment, check=True)
            print(f"Completed {dataset} seed {seed} on {device}", flush=True)

    with ThreadPoolExecutor(max_workers=len(devices)) as pool:
        futures = [pool.submit(worker, device, jobs[index::len(devices)]) for index, device in enumerate(devices)]
        for future in futures:
            future.result()


if __name__ == "__main__":
    main()
