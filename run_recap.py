#!/usr/bin/env python3
"""Train the manuscript RECAP model on one packaged input pair."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from recap import DAProDeconvConfig, DAProDeconvPipeline


ROOT = Path(__file__).resolve().parent
PHYSICAL_DESIGNS = ("replicate_holdout_A", "replicate_holdout_B", "all_pure_reference")
SCPRO_MODES = ("immune9", "treg8", "main14")


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("experiment", choices=("benchmark", "physical", "scpro", "custom"))
    p.add_argument("--dataset", help="Benchmark name from data/benchmark_manifest.json")
    p.add_argument("--design", choices=PHYSICAL_DESIGNS)
    p.add_argument("--mode", choices=SCPRO_MODES)
    p.add_argument("--ref", type=Path, help="Custom reference expression CSV")
    p.add_argument("--ref-metadata", type=Path, help="Custom reference labels CSV")
    p.add_argument("--target", type=Path, help="Custom target mixture expression CSV")
    p.add_argument("--target-fractions", type=Path, help="Optional fractions for evaluation only")
    p.add_argument("--target-expression", type=Path, help="Optional cell-type expression truth for evaluation only")
    p.add_argument("--celltype-col", default="CellType")
    p.add_argument("--outdir", type=Path)
    p.add_argument("--seed", type=int, default=2021)
    p.add_argument("--device", default="auto", help="auto, cpu, or cuda:N")
    return p


def resolve_inputs(args: argparse.Namespace) -> dict:
    if args.experiment == "benchmark":
        manifest = json.loads((ROOT / "data/benchmark_manifest.json").read_text())
        row = next((item for item in manifest if item["name"] == args.dataset), None)
        if row is None:
            raise ValueError(f"Choose --dataset from: {', '.join(x['name'] for x in manifest)}")
        return {
            "ref": ROOT / row["ref_expr"], "metadata": ROOT / row["ref_meta"],
            "target": ROOT / row["spatial_expr"],
            "fractions": ROOT / row["target_fractions"],
            "expression": ROOT / row["target_expression"],
            "celltype_col": row["celltype_col"], "name": row["name"],
        }
    if args.experiment == "physical":
        if not args.design:
            raise ValueError("physical requires --design")
        directory = ROOT / "data/proteomixture" / args.design
        return {
            "ref": directory / "reference_expression.csv",
            "metadata": directory / "reference_labels.csv",
            "target": directory / "target_expression.csv",
            "fractions": directory / "target_fractions.csv",
            "expression": None,
            "celltype_col": "CellType", "name": args.design,
        }
    if args.experiment == "scpro":
        if not args.mode:
            raise ValueError("scpro requires --mode")
        directory = ROOT / "data/scpro" / args.mode
        return {
            "ref": directory / "reference_expression.csv",
            "metadata": directory / "reference_labels.csv",
            "target": directory / "target_expression.csv",
            "fractions": None, "expression": None,
            "celltype_col": "CellType", "name": args.mode,
        }
    if not args.ref or not args.ref_metadata or not args.target:
        raise ValueError("custom requires --ref, --ref-metadata and --target")
    return {
        "ref": args.ref, "metadata": args.ref_metadata, "target": args.target,
        "fractions": args.target_fractions, "expression": args.target_expression,
        "celltype_col": args.celltype_col, "name": "custom",
    }


def main() -> None:
    args = parser().parse_args()
    inputs = resolve_inputs(args)
    for key in ("ref", "metadata", "target", "fractions", "expression"):
        path = inputs[key]
        if path is not None and not path.is_file():
            raise FileNotFoundError(f"{key}: {path}")
    outdir = args.outdir or ROOT / "results" / args.experiment / inputs["name"] / f"seed_{args.seed}"
    alphas = [0.3] if args.experiment == "physical" else [0.1, 0.2, 0.3]
    cfg = DAProDeconvConfig(
        seed=args.seed, device=args.device, dirichlet_alphas=alphas,
        dirichlet_alpha_weights=[1.0 / len(alphas)] * len(alphas),
        target_type=(
            "simulated" if args.experiment == "benchmark"
            or (args.experiment == "custom" and inputs["expression"] is not None)
            else "real"
        ),
    )
    pipeline = DAProDeconvPipeline(cfg)
    pipeline.fit_from_files(
        ref_path=str(inputs["ref"]), spatial_path=str(inputs["target"]),
        ref_celltype_col=inputs["celltype_col"],
        ref_metadata_path=str(inputs["metadata"]),
        target_fraction_path=str(inputs["fractions"]) if inputs["fractions"] else None,
        target_expression_path=str(inputs["expression"]) if inputs["expression"] else None,
        outdir=str(outdir),
    )
    print(f"RECAP result: {outdir}")


if __name__ == "__main__":
    main()
