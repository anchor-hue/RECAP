#!/usr/bin/env python3
"""Run the three SCPro reference panels; no figures."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODES = ("immune9", "treg8", "main14")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--devices", default="auto", help="Comma-separated devices")
    parser.add_argument("--outroot", type=Path, default=ROOT / "results/scpro")
    args = parser.parse_args()
    devices = [value.strip() for value in args.devices.split(",") if value.strip()]
    if not devices:
        parser.error("--devices must contain at least one device")

    def worker(device: str, modes: tuple[str, ...]) -> None:
        for mode in modes:
            outdir = args.outroot / mode / f"seed_{args.seed}"
            outdir.mkdir(parents=True, exist_ok=True)
            command = [
                sys.executable, str(ROOT / "run_recap.py"), "scpro",
                "--mode", mode, "--seed", str(args.seed),
                "--device", device, "--outdir", str(outdir),
            ]
            environment = os.environ.copy()
            environment.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
            with (outdir / "run.log").open("w") as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                               env=environment, check=True)
            print(f"SCPro result: {outdir}", flush=True)

    with ThreadPoolExecutor(max_workers=len(devices)) as pool:
        futures = [pool.submit(worker, device, MODES[index::len(devices)])
                   for index, device in enumerate(devices)]
        for future in futures:
            future.result()


if __name__ == "__main__":
    main()
