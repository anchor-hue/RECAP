"""RECAP evaluation utilities."""
from __future__ import annotations

import os
from math import sqrt
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def _finite_pair(pred: np.ndarray, gt: np.ndarray):
    pred = np.asarray(pred, dtype=float).reshape(-1)
    gt = np.asarray(gt, dtype=float).reshape(-1)
    mask = np.isfinite(pred) & np.isfinite(gt)
    return pred[mask], gt[mask]


def concordance_ccc(pred: np.ndarray, gt: np.ndarray) -> float:
    pred, gt = _finite_pair(pred, gt)
    if pred.size < 2:
        return float("nan")
    vx, vy = np.var(pred), np.var(gt)
    mx, my = np.mean(pred), np.mean(gt)
    if np.std(pred) == 0 or np.std(gt) == 0:
        return float("nan")
    corr = np.corrcoef(pred, gt)[0, 1]
    denom = vx + vy + (mx - my) ** 2
    return float(2.0 * corr * np.std(pred) * np.std(gt) / denom) if denom > 0 else float("nan")


def compute_array_metrics(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    x, y = _finite_pair(pred, gt)
    if x.size == 0:
        return {"MSE": float("nan"), "MAE": float("nan"), "RMSE": float("nan"), "Corr": float("nan"), "CCC": float("nan"), "PredMean": float("nan"), "TrueMean": float("nan"), "PredStd": float("nan"), "TrueStd": float("nan"), "StdRatio": float("nan")}
    mse = float(np.mean((x - y) ** 2))
    mae = float(np.mean(np.abs(x - y)))
    rmse = sqrt(mse)
    corr = float(np.corrcoef(x, y)[0, 1]) if x.size > 1 and np.std(x) > 0 and np.std(y) > 0 else float("nan")
    pred_std = float(np.std(x))
    true_std = float(np.std(y))
    return {
        "MSE": mse,
        "MAE": mae,
        "RMSE": rmse,
        "Corr": corr,
        "CCC": concordance_ccc(x, y),
        "PredMean": float(np.mean(x)),
        "TrueMean": float(np.mean(y)),
        "PredStd": pred_std,
        "TrueStd": true_std,
        "StdRatio": float(pred_std / (true_std + 1e-12)),
    }


def compute_weighted_basic_metrics(pred: np.ndarray, gt: np.ndarray, weight: np.ndarray) -> Dict[str, float]:
    """Weighted MSE/MAE/RMSE for arrays. Weight is broadcastable to pred/gt."""
    pred = np.asarray(pred, dtype=float)
    gt = np.asarray(gt, dtype=float)
    weight = np.asarray(weight, dtype=float)
    weight = np.broadcast_to(weight, pred.shape)
    mask = np.isfinite(pred) & np.isfinite(gt) & np.isfinite(weight) & (weight > 0)
    if not np.any(mask):
        return {"MSE": float("nan"), "MAE": float("nan"), "RMSE": float("nan")}
    x, y, w = pred[mask], gt[mask], weight[mask]
    wsum = np.sum(w) + 1e-12
    mse = float(np.sum(w * (x - y) ** 2) / wsum)
    mae = float(np.sum(w * np.abs(x - y)) / wsum)
    return {"MSE": mse, "MAE": mae, "RMSE": sqrt(mse)}


def compute_fraction_metrics(preds: pd.DataFrame, gt: pd.DataFrame) -> Dict[str, float]:
    common = [c for c in preds.columns if c in gt.columns]
    if not common:
        return {"CCC": float("nan"), "RMSE": float("nan"), "Corr": float("nan")}
    metrics = compute_array_metrics(preds[common].to_numpy(), gt[common].to_numpy())
    return {"CCC": metrics["CCC"], "RMSE": metrics["RMSE"], "Corr": metrics["Corr"]}


def compute_expression_metrics(pred_E: np.ndarray, gt_E: np.ndarray, cell_types: Optional[Sequence[str]] = None) -> Tuple[Dict[str, float], pd.DataFrame]:
    pred_E = np.asarray(pred_E, dtype=float)
    gt_E = np.asarray(gt_E, dtype=float)
    if pred_E.shape != gt_E.shape:
        raise ValueError(f"Expression metric shape mismatch: pred {pred_E.shape} vs gt {gt_E.shape}.")
    overall = compute_array_metrics(pred_E, gt_E)
    K = pred_E.shape[1]
    names = list(cell_types) if cell_types is not None else [f"celltype_{i}" for i in range(K)]
    rows = []
    for k in range(K):
        m = compute_array_metrics(pred_E[:, k, :], gt_E[:, k, :])
        m["cell_type"] = names[k] if k < len(names) else f"celltype_{k}"
        rows.append(m)
    by_ct = pd.DataFrame(rows)
    cols = ["cell_type", "MSE", "MAE", "RMSE", "Corr", "CCC", "PredMean", "TrueMean", "PredStd", "TrueStd", "StdRatio"]
    return overall, by_ct[cols]


def compute_masked_expression_metrics(pred_E: np.ndarray, gt_E: np.ndarray, fractions: np.ndarray, threshold: float = 0.01, cell_types: Optional[Sequence[str]] = None) -> Tuple[Dict[str, float], pd.DataFrame]:
    """Metrics for E only where the true cell type is present: pi_true > threshold."""
    pred_E = np.asarray(pred_E, dtype=float)
    gt_E = np.asarray(gt_E, dtype=float)
    fractions = np.asarray(fractions, dtype=float)
    if pred_E.shape != gt_E.shape:
        raise ValueError(f"Expression metric shape mismatch: pred {pred_E.shape} vs gt {gt_E.shape}.")
    if fractions.shape != pred_E.shape[:2]:
        raise ValueError(f"fractions shape {fractions.shape} must match E first two dims {pred_E.shape[:2]}.")
    mask = fractions > threshold
    mask3 = np.broadcast_to(mask[:, :, None], pred_E.shape)
    overall = compute_array_metrics(pred_E[mask3], gt_E[mask3])
    overall["PresentThreshold"] = float(threshold)
    overall["PresentPairs"] = int(mask.sum())
    K = pred_E.shape[1]
    names = list(cell_types) if cell_types is not None else [f"celltype_{i}" for i in range(K)]
    rows = []
    for k in range(K):
        mk = mask[:, k]
        m = compute_array_metrics(pred_E[mk, k, :], gt_E[mk, k, :]) if np.any(mk) else compute_array_metrics(np.array([]), np.array([]))
        m["cell_type"] = names[k] if k < len(names) else f"celltype_{k}"
        m["present_spots"] = int(mk.sum())
        rows.append(m)
    by_ct = pd.DataFrame(rows)
    cols = ["cell_type", "present_spots", "MSE", "MAE", "RMSE", "Corr", "CCC", "PredMean", "TrueMean", "PredStd", "TrueStd", "StdRatio"]
    return overall, by_ct[cols]


def compute_weighted_expression_metrics(pred_E: np.ndarray, gt_E: np.ndarray, fractions: np.ndarray) -> Dict[str, float]:
    """Weighted moments using true cell fractions, as defined in the manuscript."""
    pred_E = np.asarray(pred_E, dtype=float)
    gt_E = np.asarray(gt_E, dtype=float)
    fractions = np.asarray(fractions, dtype=float)
    if pred_E.shape != gt_E.shape:
        raise ValueError(f"Expression shapes differ: {pred_E.shape} versus {gt_E.shape}.")
    if fractions.shape != pred_E.shape[:2]:
        raise ValueError(f"fractions shape {fractions.shape} must match E first two dims {pred_E.shape[:2]}.")
    w = fractions[:, :, None]
    basic = compute_weighted_basic_metrics(pred_E, gt_E, w)
    weights = np.broadcast_to(w, pred_E.shape)
    valid = np.isfinite(pred_E) & np.isfinite(gt_E) & np.isfinite(weights) & (weights > 0)
    if not valid.any():
        return {**basic, "Corr": float("nan"), "CCC": float("nan"),
                "PredMean": float("nan"), "TrueMean": float("nan"),
                "PredStd": float("nan"), "TrueStd": float("nan"),
                "StdRatio": float("nan")}
    x, y, weight = pred_E[valid], gt_E[valid], weights[valid]
    total = weight.sum()
    mx, my = np.sum(weight * x) / total, np.sum(weight * y) / total
    vx = np.sum(weight * (x - mx) ** 2) / total
    vy = np.sum(weight * (y - my) ** 2) / total
    cov = np.sum(weight * (x - mx) * (y - my)) / total
    corr = cov / np.sqrt(vx * vy) if vx > 0 and vy > 0 else float("nan")
    denom = vx + vy + (mx - my) ** 2
    ccc = 2 * cov / denom if denom > 0 else float("nan")
    return {
        "MSE": basic["MSE"],
        "MAE": basic["MAE"],
        "RMSE": basic["RMSE"],
        "Corr": float(corr),
        "CCC": float(ccc),
        "PredMean": float(mx),
        "TrueMean": float(my),
        "PredStd": float(np.sqrt(vx)),
        "TrueStd": float(np.sqrt(vy)),
        "StdRatio": float(np.sqrt(vx) / (np.sqrt(vy) + 1e-12)),
    }


def load_expression_truth_npz(path: str) -> np.ndarray:
    if path.lower().endswith(".npy"):
        return np.load(path).astype(np.float32)
    if path.lower().endswith(".csv"):
        df = pd.read_csv(path, index_col=0)
        arr = df.values.astype(np.float32)
        if arr.ndim != 2:
            raise ValueError(f"CSV target expression must be 2D (K_celltypes, P_proteins), got {arr.shape}.")
        # Rows = cell types, columns = proteins → (1, K, P) for a single spot.
        return arr[np.newaxis, :, :]  # (1, K, P)
    data = np.load(path, allow_pickle=True)
    arr = data["E"] if "E" in data.files else data[data.files[0]]
    if arr.ndim != 3:
        raise ValueError(f"Target expression truth must have shape (N,K,P), got {arr.shape}.")
    return arr.astype(np.float32)


def save_loss_curves(outdir: str, history: Dict[str, Iterable[float]], prefix: str = "training") -> None:
    """Save training history as a table; publication figures are external."""
    os.makedirs(outdir, exist_ok=True)
    hist_df = pd.DataFrame({k: list(v) for k, v in history.items()})
    hist_df.to_csv(os.path.join(outdir, f"{prefix}_history.csv"), index=False)
