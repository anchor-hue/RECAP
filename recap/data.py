"""Data loading, normalization, protein-space alignment, and simulated-spatial synthesis."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp


@dataclass
class ProteinAlignment:
    target_proteins: List[str]
    shared_proteins: List[str]
    sp_only_proteins: List[str]
    ref_only_proteins: List[str]
    shared_mask: np.ndarray
    sp_only_mask: np.ndarray
    reliability: np.ndarray


def _read_h5ad(path: str):
    try:
        import anndata as ad  # type: ignore
    except ImportError as exc:
        raise ImportError("Reading .h5ad requires `anndata`. Install it or use CSV input.") from exc
    return ad.read_h5ad(path)


def read_expression_table(path: str, celltype_col: Optional[str] = None) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    """Read expression data from .csv/.tsv/.h5ad.

    CSV/TSV format: rows=samples/cells/spots, columns=proteins, first column=index.
    H5AD format: `X` is expression, `obs[celltype_col]` is optional labels.
    """
    lower = path.lower()
    labels = None
    if lower.endswith(".h5ad"):
        adata = _read_h5ad(path)
        X = adata.X.toarray() if sp.issparse(adata.X) else np.asarray(adata.X)
        df = pd.DataFrame(X, index=adata.obs_names.astype(str), columns=adata.var_names.astype(str))
        if celltype_col is not None:
            if celltype_col not in adata.obs.columns:
                raise KeyError(f"celltype_col='{celltype_col}' not found in h5ad obs.")
            labels = pd.Series(adata.obs[celltype_col].astype(str).values, index=df.index, name=celltype_col)
        return df.fillna(0.0).replace([np.inf, -np.inf], 0.0), labels
    sep = "\t" if lower.endswith(".tsv") or lower.endswith(".txt") else ","
    df = pd.read_csv(path, sep=sep, index_col=0)
    # Coerce all columns to numeric (non-numeric text like "minimum completeness"
    # become NaN), then fill NaN and Inf with 0.0.  Without this, a single text
    # value in a column makes the whole column object-dtype and breaks normalize_frame.
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.fillna(0.0).replace([np.inf, -np.inf], 0.0)
    return df, labels


def read_metadata(path: str, celltype_col: str, index_like: Optional[pd.Index] = None) -> pd.Series:
    lower = path.lower()
    sep = "\t" if lower.endswith(".tsv") or lower.endswith(".txt") else ","
    meta = pd.read_csv(path, sep=sep, index_col=0)
    if celltype_col not in meta.columns:
        raise KeyError(f"celltype_col='{celltype_col}' not found in metadata.")
    labels = meta[celltype_col].astype(str)
    if index_like is not None:
        labels = labels.reindex(index_like)
        if labels.isna().any():
            missing = labels.index[labels.isna()].tolist()[:5]
            raise ValueError(f"Metadata is missing labels for samples such as {missing}.")
    return labels


def normalize_frame(df: pd.DataFrame, method: str = "none") -> pd.DataFrame:
    """Normalize expression table.

    `none` is recommended when data are already preprocessed.
    `z_score` standardizes each protein across samples.
    `min_max_sample` scales each sample row to [0, 1].
    """
    if method is None or method == "none":
        return df.astype(np.float32)
    if method == "z_score":
        std = df.std(axis=0)
        # ddof=1 produces NaN when a column has fewer than 2 observations
        # (e.g. a single spatial spot).  Fall back to ddof=0 in that case.
        nan_mask = std.isna()
        if nan_mask.any():
            std[nan_mask] = df.loc[:, nan_mask].std(axis=0, ddof=0)
        result = (df - df.mean(axis=0)) / (std + 1e-10)
        return result.fillna(0.0).replace([np.inf, -np.inf], 0.0).astype(np.float32)
    if method == "min_max_sample":
        arr = df.to_numpy(dtype=np.float32)
        mn = arr.min(axis=1, keepdims=True)
        mx = arr.max(axis=1, keepdims=True)
        arr = (arr - mn) / (mx - mn + 1e-10)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        return pd.DataFrame(arr, index=df.index, columns=df.columns)
    raise ValueError(f"Unsupported normalization method: {method}")


def normalize_target_expression_truth(raw_spatial: pd.DataFrame, E: Optional[np.ndarray], method: str) -> Optional[np.ndarray]:
    """Put ground-truth target E into the same scale as normalized spatial X.

    For `none`, E is returned unchanged. For protein-wise z-score, the same protein means/stds
    computed from raw spatial spots are applied to every cell-type-specific expression. For
    sample-wise min-max, the spot-level min/max of the raw mixture is applied to all cell-type
    profiles in that spot, which preserves sum_k pi_k E_k = normalized mixture when pi sums to 1.
    """
    if E is None:
        return None
    E = np.asarray(E, dtype=np.float32)
    if method is None or method == "none":
        return E
    X = raw_spatial.to_numpy(dtype=np.float32)
    if E.shape[0] != X.shape[0] or E.shape[2] != X.shape[1]:
        raise ValueError(
            f"target_expression shape {E.shape} is incompatible with spatial table {X.shape}; "
            "expected N_spots x K_celltypes x P_spatial_proteins."
        )
    if method == "z_score":
        mean = X.mean(axis=0, keepdims=True)
        std = X.std(axis=0, ddof=1, keepdims=True) + 1e-10
        return ((E - mean[None, :, :]) / std[None, :, :]).astype(np.float32)
    if method == "min_max_sample":
        mn = X.min(axis=1, keepdims=True)
        mx = X.max(axis=1, keepdims=True)
        return ((E - mn[:, None, :]) / (mx[:, None, :] - mn[:, None, :] + 1e-10)).astype(np.float32)
    raise ValueError(f"Unsupported normalization method: {method}")


def align_reference_to_spatial(
    ref_df: pd.DataFrame,
    spatial_df: pd.DataFrame,
    sp_only_reliability: float = 0.3,
) -> Tuple[pd.DataFrame, pd.DataFrame, ProteinAlignment]:
    """Align reference cells to the spatial protein universe.

    Reference-only proteins are removed. Spatial-only proteins are appended to reference and set to zero.
    """
    target_proteins = list(map(str, spatial_df.columns))
    ref_proteins = list(map(str, ref_df.columns))
    ref_df = ref_df.copy()
    spatial_df = spatial_df.copy()
    ref_df.columns = ref_proteins
    spatial_df.columns = target_proteins

    ref_set = set(ref_proteins)
    target_set = set(target_proteins)
    shared = [p for p in target_proteins if p in ref_set]
    sp_only = [p for p in target_proteins if p not in ref_set]
    ref_only = [p for p in ref_proteins if p not in target_set]
    if len(shared) == 0:
        raise ValueError("No shared proteins between reference and spatial data.")

    aligned = pd.DataFrame(0.0, index=ref_df.index, columns=target_proteins, dtype=np.float32)
    aligned.loc[:, shared] = ref_df.loc[:, shared].to_numpy(dtype=np.float32)
    spatial = spatial_df.loc[:, target_proteins].astype(np.float32)

    shared_mask = np.array([p in shared for p in target_proteins], dtype=bool)
    sp_only_mask = np.array([p in sp_only for p in target_proteins], dtype=bool)
    reliability = np.ones(len(target_proteins), dtype=np.float32)
    reliability[sp_only_mask] = float(sp_only_reliability)
    alignment = ProteinAlignment(target_proteins, shared, sp_only, ref_only, shared_mask, sp_only_mask, reliability)
    return aligned, spatial, alignment


def _resolve_cell_types(labels: pd.Series, cell_types: Optional[Sequence[str]]) -> List[str]:
    if cell_types is None:
        return sorted(labels.astype(str).unique().tolist())
    return [str(x) for x in cell_types]


def _draw_one_pseudospot(
    rng: np.random.Generator,
    x_by_type: Dict[str, np.ndarray],
    cell_types: List[str],
    K: int,
    P: int,
    alpha: float,
    sample_size: int,
    noise_std: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Draw one pseudospot from a symmetric Dirichlet(alpha).

    This helper contains the exact sampling sequence of the original
    single-alpha `make_pseudospots` loop, so the scalar path and the
    multi-alpha path consume the RNG stream identically per pseudospot.

    Returns (X_row (P,), pi (K,), E_row (K,P), pi_draw (K,)).
    """
    # Draw latent proportions, then sample an actual finite cell count.  The
    # saved pi is the realized count proportion, so labels exactly match
    # the mixed expression and low sample_size settings are evaluated fairly.
    pi_draw = rng.dirichlet(np.full(K, alpha, dtype=np.float64)).astype(np.float64)
    # Numerical safeguard for numpy.random.Generator.multinomial.
    # Dirichlet samples theoretically sum to 1, but tiny floating-point
    # differences can make numpy's strict internal check fail.
    pi_draw = np.clip(pi_draw, 0.0, None)
    pi_sum = float(pi_draw.sum())
    if pi_sum <= 0.0:
        pi_draw = np.ones(K, dtype=np.float64) / K
    else:
        pi_draw = pi_draw / pi_sum
    if K > 1:
        prefix_sum = float(pi_draw[:-1].sum())
        if prefix_sum >= 1.0:
            pi_draw[:-1] = pi_draw[:-1] / prefix_sum * (1.0 - 1e-12)
        pi_draw[-1] = 1.0 - pi_draw[:-1].sum()
    pi_draw = np.clip(pi_draw, 0.0, 1.0)
    pi_draw = pi_draw / (pi_draw.sum() + 1e-12)
    counts = rng.multinomial(int(sample_size), pi_draw).astype(np.int64)
    if counts.sum() == 0:
        counts[rng.integers(0, K)] = 1
    pi = (counts / counts.sum()).astype(np.float32)
    X_row = np.zeros(P, dtype=np.float32)
    E_row = np.zeros((K, P), dtype=np.float32)
    for k, ct in enumerate(cell_types):
        n_ct = int(counts[k])
        if n_ct <= 0:
            # Undefined in a biological sense because this cell type is absent
            # from the simulated spot.  We keep zeros here; losses/metrics use
            # Pi_syn to mask or down-weight absent cell types.
            E_row[k, :] = 0.0
            continue
        cells = x_by_type[ct]
        idx = rng.integers(0, cells.shape[0], size=n_ct)
        E_row[k, :] = cells[idx].mean(axis=0)
    X_row[:] = np.einsum("k,kp->p", pi, E_row)
    if noise_std > 0:
        X_row[:] += rng.normal(0.0, noise_std, size=P).astype(np.float32)
    return X_row, pi, E_row, pi_draw


def build_alpha_allocation(
    alphas: Sequence[float],
    weights: Optional[Sequence[float]],
    n_pseudospots: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """Deterministic stratified allocation of `n_pseudospots` over alpha levels.

    Counts are computed with the largest-remainder method on the normalized
    weights (ties broken by the alpha list order), so the rounding remainder
    is assigned deterministically.  The per-pseudospot alpha assignment vector
    is then shuffled with the run RNG to randomize pseudospot order while
    staying reproducible.

    When all alphas are equal (multi-alpha interface with a single value) the
    shuffle is skipped, which keeps the resulting RNG stream identical to the
    legacy scalar `make_pseudospots` path.

    Returns (alpha_per_spot (N,) float64, allocation_df).
    """
    alphas = [float(a) for a in alphas]
    for a in alphas:
        if not (a > 0):
            raise ValueError(f"Dirichlet concentration must be > 0, got {a}")
    if len(alphas) == 0:
        raise ValueError("alphas must be a non-empty list.")
    w = np.ones(len(alphas), dtype=np.float64) if weights is None else np.asarray(weights, dtype=np.float64)
    if w.shape != (len(alphas),):
        raise ValueError(f"alpha weights length {w.shape[0]} != alphas length {len(alphas)}")
    if np.any(w < 0) or float(w.sum()) <= 0:
        raise ValueError("alpha weights must be non-negative with a positive sum.")
    w = w / w.sum()

    exact = w * n_pseudospots
    counts = np.floor(exact).astype(np.int64)
    remainder = int(n_pseudospots) - int(counts.sum())
    # Deterministic remainder assignment: largest fractional part first,
    # ties broken by earlier index in the alpha list.
    order = sorted(range(len(alphas)), key=lambda i: (-(exact[i] - counts[i]), i))
    for i in range(remainder):
        counts[order[i % len(alphas)]] += 1

    alpha_vec = np.concatenate([np.full(int(c), a, dtype=np.float64) for a, c in zip(alphas, counts)])
    if len(set(alphas)) > 1:
        rng.shuffle(alpha_vec)

    alloc_df = pd.DataFrame({
        "alpha": alphas,
        "requested_weight": [float(x) for x in (weights if weights is not None else np.ones(len(alphas)))],
        "normalized_weight": w,
        "realized_count": counts,
        "realized_fraction": counts / float(n_pseudospots),
    })
    return alpha_vec, alloc_df


def compute_composition_diagnostics(
    alpha_assigned: np.ndarray,
    pi_draw: np.ndarray,
    pi_realized: np.ndarray,
) -> pd.DataFrame:
    """Composition statistics for the pseudospot training set (README section 11).

    Reports, for the combined set and for each alpha level separately:
      - mean/median maximum cell-type fraction;
      - mean/median effective number of cell types (1 / sum_k p_k^2);
      - P(max fraction > 0.8) and P(max fraction > 0.9);
      - mean/median number of non-zero cell types.
    Each statistic is computed before discretization (Dirichlet draws) and
    after discretization/sampling to sample_size cells.
    """
    def _stats(pi: np.ndarray) -> Dict[str, float]:
        pi = np.asarray(pi, dtype=np.float64)
        max_frac = pi.max(axis=1)
        eff_k = 1.0 / np.maximum((pi ** 2).sum(axis=1), 1e-12)
        nonzero = (pi > 1e-9).sum(axis=1).astype(np.float64)
        return {
            "mean_max_fraction": float(np.mean(max_frac)),
            "median_max_fraction": float(np.median(max_frac)),
            "mean_effective_K": float(np.mean(eff_k)),
            "median_effective_K": float(np.median(eff_k)),
            "p_max_fraction_gt_0_8": float(np.mean(max_frac > 0.8)),
            "p_max_fraction_gt_0_9": float(np.mean(max_frac > 0.9)),
            "mean_nonzero_celltypes": float(np.mean(nonzero)),
            "median_nonzero_celltypes": float(np.median(nonzero)),
        }

    groups: List[Tuple[str, np.ndarray]] = [("combined", np.ones(len(alpha_assigned), dtype=bool))]
    for a in sorted(set(alpha_assigned.tolist())):
        groups.append((f"alpha={a:g}", np.isclose(alpha_assigned, a)))

    rows = []
    for group_name, mask in groups:
        for stage, pi in (("before_discretization", pi_draw), ("after_discretization", pi_realized)):
            row = {"group": group_name, "stage": stage, "n_pseudospots": int(mask.sum())}
            row.update(_stats(pi[mask]))
            rows.append(row)
    return pd.DataFrame(rows)


def make_pseudospots(
    ref_completed: pd.DataFrame,
    labels: pd.Series,
    cell_types: Optional[Sequence[str]] = None,
    n_pseudospots: int = 5000,
    sample_size: int = 50,
    alpha: float = 0.3,
    noise_std: float = 0.0,
    seed: int = 2021,
    collect_diagnostics: bool = False,
) -> Tuple:
    """Generate pseudo-spots and both supervision labels (legacy scalar mode).

    Returns X_syn (N,P), Pi_syn (N,K), E_syn (N,K,P), cell_types.

    When `collect_diagnostics=True`, also returns the alpha allocation table
    and the composition diagnostics table (6-tuple).  The generated
    pseudospots themselves are unchanged: the sampling sequence is identical
    to the original single-alpha implementation.
    """
    rng = np.random.default_rng(seed)
    labels = labels.astype(str).reindex(ref_completed.index)
    if labels.isna().any():
        raise ValueError("labels contain NaN after aligning to reference index.")
    cell_types = _resolve_cell_types(labels, cell_types)
    K, P = len(cell_types), ref_completed.shape[1]
    x_by_type = {}
    for ct in cell_types:
        arr = ref_completed.loc[labels == ct].to_numpy(dtype=np.float32)
        if arr.shape[0] == 0:
            raise ValueError(f"No reference cells found for cell type '{ct}'.")
        x_by_type[ct] = arr

    X_syn = np.zeros((n_pseudospots, P), dtype=np.float32)
    Pi_syn = np.zeros((n_pseudospots, K), dtype=np.float32)
    E_syn = np.zeros((n_pseudospots, K, P), dtype=np.float32)
    pi_draw_all = np.zeros((n_pseudospots, K), dtype=np.float64)
    for m in range(n_pseudospots):
        X_row, pi, E_row, pi_draw = _draw_one_pseudospot(
            rng, x_by_type, cell_types, K, P, alpha, sample_size, noise_std
        )
        X_syn[m] = X_row
        Pi_syn[m] = pi
        E_syn[m] = E_row
        pi_draw_all[m] = pi_draw

    if not collect_diagnostics:
        return X_syn, Pi_syn, E_syn, cell_types

    alpha_assigned = np.full(n_pseudospots, float(alpha), dtype=np.float64)
    alloc_df = pd.DataFrame({
        "alpha": [float(alpha)],
        "requested_weight": [1.0],
        "normalized_weight": [1.0],
        "realized_count": [int(n_pseudospots)],
        "realized_fraction": [1.0],
    })
    diag_df = compute_composition_diagnostics(alpha_assigned, pi_draw_all, Pi_syn)
    return X_syn, Pi_syn, E_syn, cell_types, alloc_df, diag_df


def make_pseudospots_multi(
    ref_completed: pd.DataFrame,
    labels: pd.Series,
    alphas: Sequence[float],
    alpha_weights: Optional[Sequence[float]] = None,
    cell_types: Optional[Sequence[str]] = None,
    n_pseudospots: int = 5000,
    sample_size: int = 50,
    noise_std: float = 0.0,
    seed: int = 2021,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], pd.DataFrame, pd.DataFrame]:
    """Generate pseudo-spots with a fixed multi-alpha Dirichlet strategy (v8.3).

    Each pseudospot is assigned one symmetric Dirichlet concentration from
    `alphas`; the total pseudospot budget `n_pseudospots` is allocated across
    alpha levels with a deterministic stratified scheme (see
    `build_alpha_allocation`) and the per-spot alpha order is shuffled with
    the run seed.  Everything else (cell sampling, expression aggregation,
    labels) follows the exact scalar-mode procedure.

    Returns (X_syn, Pi_syn, E_syn, cell_types, alpha_allocation_df,
             composition_diagnostics_df).
    """
    rng = np.random.default_rng(seed)
    labels = labels.astype(str).reindex(ref_completed.index)
    if labels.isna().any():
        raise ValueError("labels contain NaN after aligning to reference index.")
    cell_types = _resolve_cell_types(labels, cell_types)
    K, P = len(cell_types), ref_completed.shape[1]
    x_by_type = {}
    for ct in cell_types:
        arr = ref_completed.loc[labels == ct].to_numpy(dtype=np.float32)
        if arr.shape[0] == 0:
            raise ValueError(f"No reference cells found for cell type '{ct}'.")
        x_by_type[ct] = arr

    alpha_vec, alloc_df = build_alpha_allocation(alphas, alpha_weights, n_pseudospots, rng)

    X_syn = np.zeros((n_pseudospots, P), dtype=np.float32)
    Pi_syn = np.zeros((n_pseudospots, K), dtype=np.float32)
    E_syn = np.zeros((n_pseudospots, K, P), dtype=np.float32)
    pi_draw_all = np.zeros((n_pseudospots, K), dtype=np.float64)
    for m in range(n_pseudospots):
        X_row, pi, E_row, pi_draw = _draw_one_pseudospot(
            rng, x_by_type, cell_types, K, P, float(alpha_vec[m]), sample_size, noise_std
        )
        X_syn[m] = X_row
        Pi_syn[m] = pi
        E_syn[m] = E_row
        pi_draw_all[m] = pi_draw

    diag_df = compute_composition_diagnostics(alpha_vec, pi_draw_all, Pi_syn)
    return X_syn, Pi_syn, E_syn, cell_types, alloc_df, diag_df


def make_simulated_spatial_from_single_cells(
    cell_df: pd.DataFrame,
    labels: pd.Series,
    cell_types: Optional[Sequence[str]] = None,
    n_spots: int = 1000,
    sample_size: int = 50,
    alpha: float = 0.3,
    noise_std: float = 0.0,
    seed: int = 2021,
    spot_prefix: str = "spot",
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, List[str]]:
    """Mix single-cell protein profiles into simulated spatial spots with full ground truth.

    Returns:
      spatial_df: N_spots x P expression matrix.
      fraction_df: N_spots x K true cell-type proportions.
      E_true: N_spots x K x P true cell-type-specific expression tensor.
      cell_types: order of K cell types used in both fraction_df and E_true.
    """
    X, Pi, E, cell_types = make_pseudospots(
        ref_completed=cell_df,
        labels=labels,
        cell_types=cell_types,
        n_pseudospots=n_spots,
        sample_size=sample_size,
        alpha=alpha,
        noise_std=noise_std,
        seed=seed,
    )
    index = [f"{spot_prefix}_{i:05d}" for i in range(n_spots)]
    spatial_df = pd.DataFrame(X, index=index, columns=cell_df.columns)
    fraction_df = pd.DataFrame(Pi, index=index, columns=cell_types)
    return spatial_df, fraction_df, E.astype(np.float32), cell_types


def save_simulated_spatial(
    outdir: str,
    spatial_df: pd.DataFrame,
    fraction_df: pd.DataFrame,
    E_true: np.ndarray,
    cell_types: Sequence[str],
    prefix: str = "spatial",
) -> Dict[str, str]:
    import os

    os.makedirs(outdir, exist_ok=True)
    paths = {
        "spatial": os.path.join(outdir, f"{prefix}.csv"),
        "fractions": os.path.join(outdir, f"{prefix}_fractions.csv"),
        "expression": os.path.join(outdir, f"{prefix}_expression_gt.npz"),
    }
    spatial_df.to_csv(paths["spatial"])
    fraction_df.to_csv(paths["fractions"])
    np.savez_compressed(
        paths["expression"],
        E=E_true,
        spot_names=np.array(spatial_df.index),
        cell_types=np.array(cell_types),
        proteins=np.array(spatial_df.columns),
    )
    return paths


def compute_prototypes(ref_completed: pd.DataFrame, labels: pd.Series, cell_types: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
    labels = labels.astype(str).reindex(ref_completed.index)
    X = ref_completed.to_numpy(dtype=np.float32)
    mu, var = [], []
    for ct in cell_types:
        idx = np.where(labels.values == str(ct))[0]
        if idx.size == 0:
            raise ValueError(f"No cells for cell type '{ct}'.")
        sub = X[idx]
        mu.append(sub.mean(axis=0))
        var.append(sub.var(axis=0) + 1e-4)
    return np.stack(mu).astype(np.float32), np.stack(var).astype(np.float32)
