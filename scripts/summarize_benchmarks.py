#!/usr/bin/env python3
"""Collect the manuscript metrics from seven completed RECAP benchmark runs."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from recap.metrics import compute_weighted_expression_metrics  # noqa: E402

METRICS = (
    "target_fraction_CCC", "target_fraction_Corr", "target_fraction_RMSE",
    "target_expression_weighted_CCC", "target_expression_weighted_Corr",
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=2021)
    p.add_argument("--results-root", type=Path, default=ROOT / "results/benchmark")
    p.add_argument("--outdir", type=Path, default=ROOT / "analysis/benchmark")
    args = p.parse_args()
    manifest = json.loads((ROOT / "data/benchmark_manifest.json").read_text())
    panels = {
        item["dataset"]: item["proteins"]
        for item in json.loads((ROOT / "data/benchmark_common_protein_panels.json").read_text())
    }
    rows = []
    for item in manifest:
        path = args.results_root / item["name"] / f"seed_{args.seed}" / "metrics.csv"
        values = pd.read_csv(path).iloc[0]
        prediction = path.parent / "target_celltype_expression_input_scale.npz"
        with np.load(prediction, allow_pickle=False) as predicted, np.load(
            ROOT / item["target_expression"], allow_pickle=True
        ) as truth:
            predicted_names = list(map(str, predicted["proteins"]))
            truth_names = list(map(str, truth["proteins"]))
            predicted_proteins = set(predicted_names)
            truth_proteins = set(truth_names)
            panel = panels[item["name"]]
            if not set(panel) <= predicted_proteins & truth_proteins:
                raise ValueError(f"Paper panel contains unavailable proteins: {item['name']}")
            spot_names = list(map(str, predicted["spot_names"]))
            cell_types = list(map(str, predicted["cell_types"]))
            true_spots = list(map(str, truth["spot_names"]))
            true_types = list(map(str, truth["cell_types"]))
            if set(spot_names) != set(true_spots) or set(cell_types) != set(true_types):
                raise ValueError(f"Prediction and truth labels differ: {item['name']}")
            fractions = pd.read_csv(ROOT / item["target_fractions"], index_col=0).loc[
                spot_names, cell_types
            ].to_numpy(float)
            full_panel = [protein for protein in truth_names if protein in predicted_proteins]
            full_estimate = predicted["E"][:, :, [predicted_names.index(protein) for protein in full_panel]]
            full_reference = truth["E"][[true_spots.index(spot) for spot in spot_names]][
                :, [true_types.index(cell_type) for cell_type in cell_types]
            ][:, :, [truth_names.index(protein) for protein in full_panel]]
            full_metrics = compute_weighted_expression_metrics(full_estimate, full_reference, fractions)
            estimate = predicted["E"][:, :, [predicted_names.index(protein) for protein in panel]]
            reference = truth["E"][[true_spots.index(spot) for spot in spot_names]][
                :, [true_types.index(cell_type) for cell_type in cell_types]
            ][:, :, [truth_names.index(protein) for protein in panel]]
            paper_metrics = compute_weighted_expression_metrics(estimate, reference, fractions)
        coverage = len(predicted_proteins & truth_proteins) / len(truth_proteins)
        rows.append({"dataset": item["name"], "seed": args.seed,
                     **{metric: float(values[metric]) for metric in METRICS if metric.startswith("target_fraction_")},
                     "target_expression_weighted_CCC": full_metrics["CCC"],
                     "target_expression_weighted_Corr": full_metrics["Corr"],
                     "paper_panel_proteins": len(panel),
                     "paper_panel_weighted_CCC": paper_metrics["CCC"],
                     "paper_panel_weighted_Pearson_r": paper_metrics["Corr"],
                     "protein_coverage": coverage})
    table = pd.DataFrame(rows)
    args.outdir.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.outdir / f"seed_{args.seed}_per_dataset.csv", index=False)
    table[list(METRICS) + ["paper_panel_weighted_CCC", "paper_panel_weighted_Pearson_r",
                           "protein_coverage"]].mean().to_frame(name="macro_mean").to_csv(
        args.outdir / f"seed_{args.seed}_macro_mean.csv"
    )
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
