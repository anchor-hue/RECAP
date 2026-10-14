#!/usr/bin/env python3
"""Train the physical-mixture and/or SCPro models used in the manuscript."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PHYSICAL = ("replicate_holdout_A", "replicate_holdout_B", "all_pure_reference")
SCPRO = ("immune9", "treg8", "main14")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiment", choices=("physical", "scpro", "both"), default="both")
    p.add_argument("--seed", type=int, default=2021)
    p.add_argument("--devices", default="auto", help="Comma-separated devices")
    p.add_argument("--outroot", type=Path, default=ROOT / "results")
    args = p.parse_args()
    devices = [x.strip() for x in args.devices.split(",") if x.strip()]
    experiments = []
    if args.experiment in ("physical", "both"):
        experiments.extend(("physical", name) for name in PHYSICAL)
    if args.experiment in ("scpro", "both"):
        experiments.extend(("scpro", name) for name in SCPRO)

    def worker(device: str, jobs: list[tuple[str, str]]) -> None:
        for experiment, name in jobs:
            outdir = args.outroot / experiment / name / f"seed_{args.seed}"
            outdir.mkdir(parents=True, exist_ok=True)
            flag = "--design" if experiment == "physical" else "--mode"
            command = [
                sys.executable, str(ROOT / "run_recap.py"), experiment,
                flag, name, "--seed", str(args.seed), "--device", device,
                "--outdir", str(outdir),
            ]
            environment = os.environ.copy()
            environment.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", PYTHONUNBUFFERED="1")
            with (outdir / "run.log").open("w") as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=environment, check=True)
            print(f"Completed {experiment}/{name} seed {args.seed} on {device}", flush=True)

    with ThreadPoolExecutor(max_workers=len(devices)) as pool:
        futures = [pool.submit(worker, device, experiments[index::len(devices)]) for index, device in enumerate(devices)]
        for future in futures:
            future.result()


if __name__ == "__main__":
    main()
