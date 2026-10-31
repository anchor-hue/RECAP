#!/usr/bin/env python3
"""Run the fixed ProteoMixture all-pure-reference experiment; no figures."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SEED = 2021
DESIGN = "all_pure_reference"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto", help="auto, cpu, or cuda:N")
    parser.add_argument("--outroot", type=Path, default=ROOT / "results/physical")
    args = parser.parse_args()

    outdir = args.outroot / DESIGN / f"seed_{SEED}"
    outdir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, str(ROOT / "run_recap.py"), "physical",
        "--design", DESIGN, "--seed", str(SEED),
        "--device", args.device, "--outdir", str(outdir),
    ]
    environment = os.environ.copy()
    environment.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    with (outdir / "run.log").open("w") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                       env=environment, check=True)
    print(f"ProteoMixture result: {outdir}")


if __name__ == "__main__":
    main()
