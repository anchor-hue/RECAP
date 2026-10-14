#!/usr/bin/env python3
"""Compute the manuscript SCPro fraction summaries and T8 GSEA without figures."""
from __future__ import annotations

import argparse
import itertools
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MODES = ("immune9", "treg8", "main14")
IMMUNE = ("B", "DC", "MAC", "MO", "MYE", "NEU", "T4", "T8", "Treg")
MYELOID = ("MYE", "NEU", "MO", "MAC")
TCELL = ("T4", "T8", "Treg")


def read_mode(prepared_root: Path, results_root: Path, mode: str, seed: int) -> pd.DataFrame:
    meta = pd.read_csv(prepared_root / mode / "target_metadata.csv")
    if "Group" not in meta:
        raise ValueError(f"{mode}: target metadata lacks Group")
    fractions = pd.read_csv(
        results_root / mode / f"seed_{seed}" / "target_celltype_proportions.csv", index_col=0
    )
    if not fractions.index.is_unique or not meta.ID.is_unique:
        raise ValueError(f"{mode}: sample IDs are not unique")
    joined = meta.set_index("ID")[["Group"]].join(fractions, how="inner")
    if len(joined) != len(fractions):
        raise ValueError(f"{mode}: target metadata and proportions are misaligned")
    return joined.rename(columns={"Group": "Region"})


def hedges_g(a: np.ndarray, b: np.ndarray) -> float:
    pooled = ((len(a) - 1) * np.var(a, ddof=1) + (len(b) - 1) * np.var(b, ddof=1)) / (len(a) + len(b) - 2)
    d = (a.mean() - b.mean()) / math.sqrt(pooled)
    return float(d * (1 - 3 / (4 * (len(a) + len(b)) - 9)))


def exact_p(a: np.ndarray, b: np.ndarray) -> float:
    values = np.r_[a, b]
    observed = abs(a.mean() - b.mean())
    differences = []
    for indices in itertools.combinations(range(len(values)), len(a)):
        mask = np.zeros(len(values), bool)
        mask[list(indices)] = True
        differences.append(abs(values[mask].mean() - values[~mask].mean()))
    return float(np.mean(np.asarray(differences) >= observed - 1e-12))


def summarize(prepared_root: Path, results_root: Path, out: Path, seed: int) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    immune, treg, main = (read_mode(prepared_root, results_root, mode, seed) for mode in MODES)
    immune_regional = immune.groupby("Region")[list(IMMUNE)].mean()
    immune_regional.to_csv(out / "immune9_regional_means.csv")

    lineage = immune[["Region"]].copy()
    myeloid = immune[list(MYELOID)].sum(axis=1)
    tcell = immune[list(TCELL)].sum(axis=1)
    lineage["Myeloid"] = myeloid / (myeloid + tcell)
    lineage["Tcell"] = tcell / (myeloid + tcell)
    lineage.to_csv(out / "immune9_lineage_by_sample.csv")
    lineage_mean = lineage.groupby("Region")[["Myeloid", "Tcell"]].mean()
    lineage_sd = lineage.groupby("Region")[["Myeloid", "Tcell"]].std(ddof=1)
    lineage_mean.to_csv(out / "immune9_lineage_means.csv")
    lineage_sd.to_csv(out / "immune9_lineage_sample_sd.csv")
    a = lineage.loc[lineage.Region == "IT", "Myeloid"].to_numpy(float)
    b = lineage.loc[lineage.Region == "LN", "Myeloid"].to_numpy(float)

    treg_means = treg.groupby("Region")[list(treg.columns.drop("Region"))].mean()
    treg_means.T.to_csv(out / "treg8_regional_means.csv")

    shared_ids = [sid for sid in immune.index if sid in main.index]
    main_shared = main.loc[shared_ids, list(IMMUNE)]
    main_shared = main_shared.div(main_shared.sum(axis=1), axis=0)
    immune_shared = immune.loc[shared_ids, list(IMMUNE)]
    main_region = main.loc[shared_ids, "Region"]
    main_myeloid = main_shared[list(MYELOID)].sum(axis=1)
    main_tcell = main_shared[list(TCELL)].sum(axis=1)
    main_lineage = pd.DataFrame({
        "Region": main_region,
        "Myeloid": main_myeloid / (main_myeloid + main_tcell),
        "Tcell": main_tcell / (main_myeloid + main_tcell),
    })
    main_lineage.groupby("Region")[["Myeloid", "Tcell"]].mean().to_csv(
        out / "main14_shared_immune_lineage_means.csv"
    )
    rows = []
    for cell_type in IMMUNE:
        x = main_shared[cell_type].to_numpy(float)
        y = immune_shared[cell_type].to_numpy(float)
        rows.append({
            "CellType": cell_type, "N": len(x),
            "Pearson": float(np.corrcoef(x, y)[0, 1]),
            "MAE": float(np.mean(np.abs(x - y))),
        })
    agreement = pd.DataFrame(rows)
    agreement.to_csv(out / "main14_vs_immune9.csv", index=False)
    summary = {
        "seed": seed,
        "n_immune9_IT": len(a), "n_immune9_LN": len(b),
        "myeloid_IT_mean": float(lineage_mean.loc["IT", "Myeloid"]),
        "myeloid_IT_sample_sd": float(lineage_sd.loc["IT", "Myeloid"]),
        "myeloid_LN_mean": float(lineage_mean.loc["LN", "Myeloid"]),
        "myeloid_LN_sample_sd": float(lineage_sd.loc["LN", "Myeloid"]),
        "tcell_IT_mean": float(lineage_mean.loc["IT", "Tcell"]),
        "tcell_IT_sample_sd": float(lineage_sd.loc["IT", "Tcell"]),
        "tcell_LN_mean": float(lineage_mean.loc["LN", "Tcell"]),
        "tcell_LN_sample_sd": float(lineage_sd.loc["LN", "Tcell"]),
        "myeloid_hedges_g": hedges_g(a, b),
        "myeloid_exact_permutation_p": exact_p(a, b),
        "KPC_425KP_IT": float(treg_means.loc["IT", "KPC_425KP"]),
        "KPC_425KP_LN": float(treg_means.loc["LN", "KPC_425KP"]),
        "KPC_8KP_IT": float(treg_means.loc["IT", "KPC_8KP"]),
        "KPC_8KP_LN": float(treg_means.loc["LN", "KPC_8KP"]),
        "KPC_8KN_IT": float(treg_means.loc["IT", "KPC_8KN"]),
        "KPC_8KN_LN": float(treg_means.loc["LN", "KPC_8KN"]),
        "main14_immune9_median_pearson": float(agreement.Pearson.median()),
        "main14_immune9_median_mae": float(agreement.MAE.median()),
    }
    (out / "paper_quantities.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--prepared-root", type=Path, default=ROOT / "data/scpro")
    parser.add_argument("--results-root", type=Path, default=ROOT / "results/scpro")
    parser.add_argument("--outdir", type=Path, default=ROOT / "analysis/scpro")
    parser.add_argument("--skip-gsea", action="store_true", help="Only calculate fraction summaries")
    args = parser.parse_args()
    summary = summarize(args.prepared_root, args.results_root, args.outdir, args.seed)
    if not args.skip_gsea:
        command = [
            sys.executable, str(ROOT / "scripts/scpro_gsea.py"),
            "--prepared-dir", str(args.prepared_root / "immune9"),
            "--result-dir", str(args.results_root / "immune9" / f"seed_{args.seed}"),
            "--output-dir", str(args.outdir / "T8_gsea"),
            "--annotation-file", str(args.prepared_root / "ct_dataset_01_protein_annotation.csv"),
            "--gene-set-dir", str(args.prepared_root),
            "--subtypes", "T8", "--permutations", "2000", "--threads", "2", "--seed", "42",
        ]
        subprocess.run(command, check=True)
        gsea = pd.read_csv(args.outdir / "T8_gsea/gsea_all_results.csv")
        q = pd.to_numeric(gsea["FDR q-val"], errors="coerce")
        nes = pd.to_numeric(gsea["NES"], errors="coerce")
        summary.update({
            "T8_gsea_fdr_lt_0.05": int((q < 0.05).sum()),
            "T8_gsea_IT": int(((q < 0.05) & (nes > 0)).sum()),
            "T8_gsea_LN": int(((q < 0.05) & (nes < 0)).sum()),
        })
        (args.outdir / "paper_quantities.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
