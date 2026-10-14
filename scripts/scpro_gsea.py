#!/usr/bin/env python3
"""
Cell-type-specific residual differential analysis and pre-ranked GSEA for
direct SCPro immune9 or treg8 runs.

Expected files:
    prepared_dir/target_metadata.csv
    prepared_dir/protein_order.csv
    prepared_dir/celltype_order.csv
    result_dir/target_celltype_expression.npz

The analysis:
1. compares IT versus LN for each selected cell type/subtype;
2. computes raw expression differences;
3. removes the mean signal across the other subtypes within each sample;
4. ranks genes by the IT-minus-LN specific residual difference;
5. runs GSEApy prerank for selected subtype(s).

The ranked score is an effect-size-like continuous statistic, not a validated
target-specific expression truth.
"""

from __future__ import annotations

import argparse
import itertools
import hashlib
import json
import math
import re
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from recap.expression_scale import load_target_expression
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats

try:
    import gseapy as gp
except ImportError:
    gp = None


EXPRESSION_CANDIDATES = [
    "target_celltype_expression_input_scale.npz",
]


class EnrichmentError(RuntimeError):
    pass


def decode_strings(values: Iterable[object]) -> list[str]:
    return [
        value.decode("utf-8", errors="replace")
        if isinstance(value, bytes)
        else str(value)
        for value in values
    ]


def find_expression(directory: Path) -> Path:
    for name in EXPRESSION_CANDIDATES:
        direct = directory / name
        if direct.exists():
            return direct
    hits = []
    for name in EXPRESSION_CANDIDATES:
        hits.extend(directory.rglob(name))
    hits = sorted(set(hits))
    if len(hits) != 1:
        raise EnrichmentError(
            f"Expected exactly one expression NPZ under {directory}; found {hits}"
        )
    return hits[0]


def bh_adjust(values: np.ndarray) -> np.ndarray:
    p = np.asarray(values, dtype=float)
    out = np.full_like(p, np.nan)
    finite = np.isfinite(p)
    if not finite.any():
        return out
    pf = p[finite]
    order = np.argsort(pf)
    ranked = pf[order] * len(pf) / np.arange(1, len(pf) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    ranked = np.minimum(ranked, 1.0)
    restored = np.empty_like(ranked)
    restored[order] = ranked
    out[finite] = restored
    return out


def exact_permutation_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    values = np.vstack([a, b])
    n_a = len(a)
    observed = np.abs(np.nanmean(a, axis=0) - np.nanmean(b, axis=0))
    exceed = np.zeros(values.shape[1], dtype=float)
    total = 0
    for indices in itertools.combinations(range(len(values)), n_a):
        mask = np.zeros(len(values), dtype=bool)
        mask[list(indices)] = True
        diff = np.abs(
            np.nanmean(values[mask], axis=0)
            - np.nanmean(values[~mask], axis=0)
        )
        exceed += diff >= observed - 1e-12
        total += 1
    return exceed / max(total, 1)


def orient_expression(
    array: np.ndarray,
    n_spots: int,
    n_celltypes: int,
    n_proteins: int,
) -> np.ndarray:
    expected = (n_spots, n_celltypes, n_proteins)
    if array.shape == expected:
        return array
    matches = []
    for permutation in itertools.permutations(range(3)):
        candidate = np.transpose(array, permutation)
        if candidate.shape == expected:
            matches.append(candidate)
    if len(matches) != 1:
        raise EnrichmentError(
            f"Cannot uniquely orient E shape={array.shape} to {expected}."
        )
    return matches[0]


def load_annotation(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    table = pd.read_csv(path)
    protein_candidates = [
        "ProteinID", "pids", "pid", "Protein", "Protein.ID", "Entry"
    ]
    gene_candidates = [
        "Gene", "genename", "genenames", "GeneNames", "symbol"
    ]
    protein_col = next(
        (column for column in protein_candidates if column in table.columns),
        None,
    )
    gene_col = next(
        (column for column in gene_candidates if column in table.columns),
        None,
    )
    if protein_col is None or gene_col is None:
        raise EnrichmentError(
            f"Cannot identify protein/gene columns in {path}; "
            f"columns={list(table.columns)}"
        )
    table = table[[protein_col, gene_col]].dropna()
    table[protein_col] = table[protein_col].astype(str)
    table[gene_col] = table[gene_col].astype(str)
    return dict(zip(table[protein_col], table[gene_col]))


def clean_gene_symbol(value: str) -> str:
    gene = str(value).strip()
    if ";" in gene:
        gene = gene.split(";")[0]
    if " " in gene:
        gene = gene.split(" ")[0]
    return gene.upper()


def latest_matching_library(
    available: list[str],
    requested: str,
) -> str:
    if requested in available:
        return requested
    prefix = re.sub(r"_20\d\d$", "", requested)
    candidates = [name for name in available if name.startswith(prefix)]
    if not candidates:
        raise EnrichmentError(
            f"No Mouse Enrichr library matching '{requested}'."
        )
    return sorted(candidates)[-1]


def run_analysis(args: argparse.Namespace) -> None:
    prepared_dir = Path(args.prepared_dir)
    result_dir = Path(args.result_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "expression_analysis_scale.json").write_text(json.dumps({
        "scale": args.expression_scale, "expression_key": args.expression_key,
        "rank_score": "IT-minus-LN cell-type-specific residual difference",
        "seed": args.seed, "permutations": args.permutations, "requested_gene_sets": args.gene_sets,
    }, indent=2))

    metadata = pd.read_csv(prepared_dir / "target_metadata.csv")
    if "ID" not in metadata.columns:
        metadata = metadata.rename(columns={metadata.columns[0]: "ID"})
    metadata["ID"] = metadata["ID"].astype(str)
    if "Region" in metadata.columns:
        group_column = "Region"
    elif "CellType" in metadata.columns:
        group_column = "CellType"
    elif "Group" in metadata.columns:
        group_column = "Group"
    else:
        raise EnrichmentError("Target metadata lacks Region/CellType/Group.")

    protein_order = pd.read_csv(
        prepared_dir / "protein_order.csv"
    ).iloc[:, 0].astype(str).tolist()
    celltype_order = pd.read_csv(
        prepared_dir / "celltype_order.csv"
    ).iloc[:, 0].astype(str).tolist()

    expression_path = find_expression(result_dir)
    npz = load_target_expression(expression_path, scale=args.expression_scale,
                                 spatial_path=prepared_dir / "target_expression.csv")
    if args.expression_key not in npz:
        raise EnrichmentError(
            f"NPZ lacks key {args.expression_key}; available={list(npz.keys())}"
        )
    e = np.asarray(npz[args.expression_key], dtype=float)

    spots = (
        decode_strings(npz["spot_names"])
        if "spot_names" in npz
        else metadata["ID"].tolist()
    )
    celltypes = (
        decode_strings(npz["cell_types"])
        if "cell_types" in npz
        else celltype_order
    )
    proteins = (
        decode_strings(npz["proteins"])
        if "proteins" in npz
        else protein_order
    )
    e = orient_expression(e, len(spots), len(celltypes), len(proteins))

    annotation = load_annotation(
        Path(args.annotation_file) if args.annotation_file else None
    )
    gene_map = {
        protein: clean_gene_symbol(annotation.get(protein, protein))
        for protein in proteins
    }

    metadata = metadata.set_index("ID")
    selected_spots = [
        spot for spot in spots
        if spot in metadata.index
        and str(metadata.loc[spot, group_column]) in {args.group_a, args.group_b}
    ]
    indices = [spots.index(spot) for spot in selected_spots]
    groups = [
        str(metadata.loc[spot, group_column]) for spot in selected_spots
    ]
    indices_a = [idx for idx, group in zip(indices, groups) if group == args.group_a]
    indices_b = [idx for idx, group in zip(indices, groups) if group == args.group_b]
    if len(indices_a) < 2 or len(indices_b) < 2:
        raise EnrichmentError(
            f"Need at least two samples per group; got "
            f"{args.group_a}={len(indices_a)}, {args.group_b}={len(indices_b)}"
        )

    residual = np.empty_like(e)
    for k in range(e.shape[1]):
        others = np.delete(e, k, axis=1)
        residual[:, k, :] = e[:, k, :] - np.nanmean(others, axis=1)

    rows = []
    ranked_dir = output_dir / "ranked_lists"
    ranked_dir.mkdir(exist_ok=True)

    for k, celltype in enumerate(celltypes):
        raw_a = e[indices_a, k, :]
        raw_b = e[indices_b, k, :]
        spec_a = residual[indices_a, k, :]
        spec_b = residual[indices_b, k, :]

        raw_diff = np.nanmean(raw_a, axis=0) - np.nanmean(raw_b, axis=0)
        spec_diff = np.nanmean(spec_a, axis=0) - np.nanmean(spec_b, axis=0)

        raw_t, raw_p = stats.ttest_ind(
            raw_a, raw_b, axis=0, equal_var=False, nan_policy="omit"
        )
        spec_t, spec_p = stats.ttest_ind(
            spec_a, spec_b, axis=0, equal_var=False, nan_policy="omit"
        )
        spec_exact = exact_permutation_matrix(spec_a, spec_b)

        raw_fdr = bh_adjust(raw_p)
        spec_fdr = bh_adjust(spec_p)

        for j, protein in enumerate(proteins):
            rows.append({
                "Contrast": f"{args.group_a}_vs_{args.group_b}",
                "GroupA": args.group_a,
                "GroupB": args.group_b,
                "CellType": celltype,
                "Protein": protein,
                "Gene": gene_map[protein],
                "RawDifference": raw_diff[j],
                "RawWelchT": raw_t[j],
                "RawWelchP": raw_p[j],
                "RawWelchFDR": raw_fdr[j],
                "SpecificResidualDifference": spec_diff[j],
                "SpecificWelchT": spec_t[j],
                "SpecificWelchP": spec_p[j],
                "SpecificWelchFDR": spec_fdr[j],
                "SpecificExactPermutationP": spec_exact[j],
            })

        ranking = pd.DataFrame({
            "Gene": [gene_map[p] for p in proteins],
            "Score": spec_diff,
        })
        ranking = (
            ranking.replace([np.inf, -np.inf], np.nan)
            .dropna()
            .query("Gene != ''")
        )
        ranking["AbsScore"] = ranking["Score"].abs()
        ranking = (
            ranking.sort_values("AbsScore", ascending=False)
            .drop_duplicates("Gene")
            .drop(columns="AbsScore")
            .sort_values("Score", ascending=False)
        )
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", celltype)
        ranking.to_csv(
            ranked_dir / f"{args.group_a}_vs_{args.group_b}__{safe}.rnk",
            sep="\t",
            index=False,
            header=False,
        )

    de = pd.DataFrame(rows)
    de.to_csv(
        output_dir / "scpro_direct_specific_differential_proteins.csv",
        index=False,
    )

    selected_subtypes = (
        [x.strip() for x in args.subtypes.split(",") if x.strip()]
        if args.subtypes
        else celltypes
    )
    missing_subtypes = [x for x in selected_subtypes if x not in celltypes]
    if missing_subtypes:
        raise EnrichmentError(
            f"Requested subtypes not found: {missing_subtypes}; available={celltypes}"
        )

    if args.skip_gsea:
        print("Differential tables and ranked lists were generated; GSEA skipped.")
        return

    if gp is None:
        raise EnrichmentError(
            "gseapy is not installed. Run: pip install gseapy"
        )

    requested_libraries = [
        x.strip() for x in args.gene_sets.split(",") if x.strip()
    ]
    if args.gene_set_dir:
        resolved_libraries = requested_libraries
        gene_set_inputs = {library: Path(args.gene_set_dir) / f"{library}.gmt" for library in resolved_libraries}
        for library, path in gene_set_inputs.items():
            if not path.exists():
                raise EnrichmentError(f"Frozen GMT missing for {library}: {path}")
    else:
        available = gp.get_library_name(organism="Mouse")
        resolved_libraries = [latest_matching_library(available, library) for library in requested_libraries]
        gene_set_inputs = {library: library for library in resolved_libraries}

    all_results = []
    for subtype in selected_subtypes:
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", subtype)
        ranking_path = (
            ranked_dir / f"{args.group_a}_vs_{args.group_b}__{safe}.rnk"
        )
        subtype_out = output_dir / "gsea" / safe
        subtype_out.mkdir(parents=True, exist_ok=True)

        for library in resolved_libraries:
            library_out = subtype_out / re.sub(
                r"[^A-Za-z0-9_.-]+", "_", library
            )
            result = gp.prerank(
                rnk=str(ranking_path),
                gene_sets=str(gene_set_inputs[library]),
                organism="Mouse",
                outdir=str(library_out),
                permutation_num=args.permutations,
                min_size=args.min_size,
                max_size=args.max_size,
                seed=args.seed,
                threads=args.threads,
                no_plot=True,
                verbose=True,
            )
            table = result.res2d.copy()
            table["CellType"] = subtype
            table["GeneSetLibrary"] = library
            all_results.append(table)

    combined = pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()
    combined.to_csv(output_dir / "gsea_all_results.csv", index=False)

    if not combined.empty:
        fdr_column = next(
            (
                column for column in [
                    "FDR q-val", "FDR qval", "fdr", "FDR"
                ]
                if column in combined.columns
            ),
            None,
        )
        nes_column = next(
            (
                column for column in ["NES", "nes"]
                if column in combined.columns
            ),
            None,
        )
        if fdr_column is not None and nes_column is not None:
            significant = combined[
                pd.to_numeric(combined[fdr_column], errors="coerce")
                <= args.fdr_cutoff
            ].copy()
            significant.to_csv(
                output_dir / "gsea_significant_results.csv", index=False
            )

    manifest = {
        "contrast": f"{args.group_a}_vs_{args.group_b}",
        "subtypes": selected_subtypes,
        "gene_sets_requested": requested_libraries,
        "gene_sets_resolved": resolved_libraries,
        "permutations": args.permutations,
        "ranking_score": "SpecificResidualDifference",
        "expression_scale": args.expression_scale,
        "expression_key": args.expression_key,
        "expression_file": str((result_dir / "target_celltype_expression_input_scale.npz").resolve()) if args.expression_scale == "input_file" and (result_dir / "target_celltype_expression_input_scale.npz").exists() else str(expression_path.resolve()),
        "seed": args.seed, "min_size": args.min_size, "max_size": args.max_size,
        "fdr_cutoff": args.fdr_cutoff,
        "gene_set_files": {library: {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for library, path in gene_set_inputs.items()} if args.gene_set_dir else None,
        "gene_symbols_uppercase": True,
        "interpretation": (
            "Positive NES indicates enrichment toward GroupA (IT by default); "
            "negative NES indicates enrichment toward GroupB (LN by default)."
        ),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Input-file-scale T8 residual expression and GSEA for the manuscript SCPro analysis."
    )
    parser.add_argument("--prepared-dir", required=True)
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--annotation-file")
    parser.set_defaults(expression_scale="input_file", expression_key="E", no_gsea_plots=True)
    parser.add_argument("--group-a", default="IT")
    parser.add_argument("--group-b", default="LN")
    parser.add_argument(
        "--subtypes", default="T8",
        help=(
            "Comma-separated subtype names. Omit to analyze all. "
            "Example: KPC_425KP,KPC_8KP"
        ),
    )
    parser.add_argument(
        "--gene-sets",
        default="GO_Biological_Process_2025,Reactome_Pathways_2024",
    )
    parser.add_argument("--gene-set-dir", default=None, help="Directory with frozen <library>.gmt files; avoids online library changes.")
    parser.add_argument("--permutations", type=int, default=2000)
    parser.add_argument("--min-size", type=int, default=10)
    parser.add_argument("--max-size", type=int, default=500)
    parser.add_argument("--fdr-cutoff", type=float, default=0.25)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-gsea", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        run_analysis(args)
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
