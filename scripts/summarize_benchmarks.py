#!/usr/bin/env python3
"""Summarize fractions and both manuscript RECAP expression evaluations."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from recap.metrics import (  # noqa: E402
    compute_standardized_expression_metrics, compute_weighted_expression_metrics,
)

METRICS = (
    "target_fraction_CCC", "target_fraction_Corr", "target_fraction_RMSE",
    "full_input_scale_weighted_CCC", "full_input_scale_weighted_Pearson_r",
    "table2A_standardized_weighted_CCC", "table2A_standardized_weighted_Pearson_r",
    "table2B_input_scale_weighted_CCC", "table2B_input_scale_weighted_Pearson_r",
)


def labels(values, name):
    names = list(map(str, values))
    if len(names) != len(set(names)):
        raise ValueError(f"Duplicate {name} labels.")
    return names


def protein_coverage_union(truth_proteins, predicted_protein_sets):
    """Coverage of GT proteins in the union across the selected seeds."""
    truth = set(map(str, truth_proteins))
    panels = [set(map(str, proteins)) for proteins in predicted_protein_sets]
    if not truth or not panels:
        raise ValueError("Coverage requires GT proteins and at least one seed.")
    return len(truth & set().union(*panels)) / len(truth)


def summarize_union_coverage(coverage_sets, seeds):
    rows = []
    for dataset, data in coverage_sets.items():
        if set(data["predictions"]) != set(seeds):
            raise ValueError(f"Coverage seed set differs: {dataset}")
        fraction = protein_coverage_union(
            data["truth"], [data["predictions"][seed] for seed in seeds],
        )
        rows.append({"dataset": dataset, "method": "RECAP", "n_seeds": len(seeds),
                     "n_gt_proteins": len(data["truth"]),
                     "n_covered_gt_proteins": len(data["truth"] & set().union(
                         *(data["predictions"][seed] for seed in seeds))),
                     "protein_coverage": fraction, "coverage_percent": 100 * fraction})
    return pd.DataFrame(rows)


def summarize_seed(seed, results_root, manifest, panels, coverage_sets=None):
    rows = []
    for item in manifest:
        path = results_root / item["name"] / f"seed_{seed}" / "metrics.csv"
        values = pd.read_csv(path).iloc[0]
        prediction = path.parent / "target_celltype_expression_input_scale.npz"
        with np.load(prediction, allow_pickle=False) as predicted, np.load(
            ROOT / item["target_expression"], allow_pickle=True
        ) as truth:
            predicted_names = labels(predicted["proteins"], "predicted protein")
            truth_names = labels(truth["proteins"], "truth protein")
            predicted_proteins, truth_proteins = set(predicted_names), set(truth_names)
            panel = panels[item["name"]]
            if len(panel) != len(set(panel)) or not set(panel) <= predicted_proteins & truth_proteins:
                raise ValueError(f"Invalid or unavailable paper panel: {item['name']}")
            spot_names = labels(predicted["spot_names"], "predicted sample")
            cell_types = labels(predicted["cell_types"], "predicted cell type")
            true_spots = labels(truth["spot_names"], "truth sample")
            true_types = labels(truth["cell_types"], "truth cell type")
            if set(spot_names) != set(true_spots) or set(cell_types) != set(true_types):
                raise ValueError(f"Prediction and truth labels differ: {item['name']}")
            fractions = pd.read_csv(ROOT / item["target_fractions"], index_col=0).loc[
                spot_names, cell_types
            ].to_numpy(float)
            if not np.isfinite(fractions).all() or (fractions < 0).any() or fractions.sum() <= 0:
                raise ValueError(f"Invalid true fractions: {item['name']}")
            estimate_all = np.asarray(predicted["E"], dtype=float)
            reference_all = np.asarray(truth["E"], dtype=float)[
                [true_spots.index(spot) for spot in spot_names]
            ][:, [true_types.index(cell_type) for cell_type in cell_types]]
            if estimate_all.shape != (len(spot_names), len(cell_types), len(predicted_names)):
                raise ValueError(f"Prediction shape and labels differ: {item['name']}")
            if not np.isfinite(estimate_all).all() or not np.isfinite(reference_all).all():
                raise ValueError(f"Nonfinite expression: {item['name']}")
            full_panel = [protein for protein in truth_names if protein in predicted_proteins]
            full_metrics = compute_weighted_expression_metrics(
                estimate_all[:, :, [predicted_names.index(p) for p in full_panel]],
                reference_all[:, :, [truth_names.index(p) for p in full_panel]], fractions,
            )
            estimate = estimate_all[:, :, [predicted_names.index(p) for p in panel]]
            reference = reference_all[:, :, [truth_names.index(p) for p in panel]]
            direct = compute_weighted_expression_metrics(estimate, reference, fractions)
            standardized, audit = compute_standardized_expression_metrics(estimate, reference, fractions)
        if coverage_sets is not None:
            data = coverage_sets.setdefault(item["name"], {"truth": truth_proteins, "predictions": {}})
            if data["truth"] != truth_proteins or seed in data["predictions"]:
                raise ValueError(f"Inconsistent GT proteins or duplicate seed: {item['name']}")
            data["predictions"][seed] = predicted_proteins
        rows.append({
            "dataset": item["name"], "seed": seed,
            **{m: float(values[m]) for m in METRICS if m.startswith("target_fraction_")},
            "full_input_scale_weighted_CCC": full_metrics["CCC"],
            "full_input_scale_weighted_Pearson_r": full_metrics["Corr"],
            "table2A_standardized_weighted_CCC": standardized["CCC"],
            "table2A_standardized_weighted_Pearson_r": standardized["Corr"],
            "table2B_input_scale_weighted_CCC": direct["CCC"],
            "table2B_input_scale_weighted_Pearson_r": direct["Corr"],
            "common_panel_proteins": len(panel),
            "standardized_panel_proteins": int(audit["retained_proteins"].sum()),
            "constant_prediction_proteins": int(audit["constant_prediction_proteins"].sum()),
            "excluded_constant_truth_proteins": json.dumps([
                p for p, keep in zip(panel, audit["retained_proteins"]) if not keep
            ]),
            "zero_fraction_entries": int((fractions == 0).sum()),
            "run_protein_coverage": len(predicted_proteins & truth_proteins) / len(truth_proteins),
        })
    return pd.DataFrame(rows)


def seed_macro_means(table, dataset_count):
    """Require every dataset for a seed; never silently average a partial panel."""
    grouped = table.groupby("seed")[list(METRICS)]
    return grouped.mean().where(grouped.count() == dataset_count)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    seeds_arg = p.add_mutually_exclusive_group()
    seeds_arg.add_argument("--seed", type=int, help="One seed (default: 2021)")
    seeds_arg.add_argument("--seeds", help="Comma-separated seeds, e.g. 2021,2022,2023")
    p.add_argument("--results-root", type=Path, default=ROOT / "results/benchmark")
    p.add_argument("--outdir", type=Path, default=ROOT / "analysis/benchmark")
    args = p.parse_args()
    try:
        seeds = [int(x.strip()) for x in args.seeds.split(",")] if args.seeds is not None else [
            args.seed if args.seed is not None else 2021
        ]
    except ValueError:
        p.error("--seeds must contain comma-separated integers")
    if len(seeds) != len(set(seeds)):
        p.error("Seeds must be unique")
    manifest = json.loads((ROOT / "data/benchmark_manifest.json").read_text())
    panels = {item["dataset"]: item["proteins"] for item in json.loads(
        (ROOT / "data/benchmark_common_protein_panels.json").read_text()
    )}
    tables, coverage_sets = [], {}
    # Read every requested run before creating outputs; missing runs fail explicitly.
    for seed in seeds:
        tables.append(summarize_seed(seed, args.results_root, manifest, panels, coverage_sets))
        print(f"Evaluated seed {seed} across {len(manifest)} datasets", flush=True)
    table = pd.concat(tables, ignore_index=True)
    macro = seed_macro_means(table, len(manifest))
    summary = macro.agg(["mean", "std", "count"]).T.rename(columns={
        "std": "sample_std", "count": "valid_seed_count",
    })
    coverage = summarize_union_coverage(coverage_sets, seeds)
    coverage_summary = coverage.groupby("method")["coverage_percent"].agg(["mean", "min", "max"])
    # Coverage is one union statistic, not a per-seed metric with a seed SD.
    summary.loc["protein_coverage", "mean"] = coverage["protein_coverage"].mean()
    summary["valid_seed_count"] = summary["valid_seed_count"].astype("Int64")
    args.outdir.mkdir(parents=True, exist_ok=True)
    for seed, seed_table in table.groupby("seed", sort=False):
        seed_table.to_csv(args.outdir / f"seed_{seed}_per_dataset.csv", index=False)
        macro.loc[seed].to_frame(name="macro_mean").to_csv(
            args.outdir / f"seed_{seed}_macro_mean.csv"
        )
    coverage.to_csv(args.outdir / "coverage_by_dataset.csv", index=False)
    coverage_summary.to_csv(args.outdir / "coverage_summary.csv")
    table.to_csv(args.outdir / "per_run.csv", index=False)
    macro.to_csv(args.outdir / "seed_macro.csv")
    summary.to_csv(args.outdir / "cross_dataset_summary.csv", index_label="metric")
    (args.outdir / "evaluation_protocol.json").write_text(json.dumps({
        "method": "RECAP", "seeds": seeds, "datasets": [x["name"] for x in manifest],
        "table2A": {"standardization": "independent per-protein all-entry z-score",
                    "ddof": 1, "standardization_weights": "none",
                    "includes_zero_fraction_entries": True,
                    "constant_truth": "exclude by GT-only mask for every method/seed",
                    "constant_prediction": "retain and standardize to zero",
                    "constant_detection": "exact equality of extrema"},
        "table2B": "input-file scale, no additional standardization, common panel",
        "full_input_scale": "input-file scale, all available GT/output proteins",
        "metric_weights": "true cell fractions; weighted central moments",
        "coverage": {"definition": "GT proteins covered by union of outputs across selected seeds",
                     "seeds": seeds, "dataset_aggregation": "equal dataset mean",
                     "units": "protein_coverage: fraction; coverage_percent and coverage_summary: percent",
                     "per_run_diagnostic": "run_protein_coverage; excluded from seed macro averages",
                     "seed_standard_deviation": "not applicable to the union statistic"},
        "aggregation": "equal dataset means per seed; mean and sample SD (ddof=1) across seeds",
        "undefined_metrics": "NaN; incomplete dataset sets invalidate the seed metric",
        "scope": "RECAP only; competing methods and their input adapters are not packaged",
    }, indent=2) + "\n")
    print(summary.to_string())


if __name__ == "__main__":
    main()
