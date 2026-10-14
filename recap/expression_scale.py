"""Restore RECAP expression predictions to the target input-file scale.

The conversion uses target-mixture normalization statistics and never uses
the held-out cell-type expression truth.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

MODEL_FILE = "target_celltype_expression.npz"
INPUT_FILE = "target_celltype_expression_input_scale.npz"
PARAM_FILE = "target_expression_normalization.npz"


def normalization_parameters(frame: pd.DataFrame, method: str) -> dict:
    """Match normalize_target_expression_truth, including float32/ddof/epsilon."""
    x = frame.to_numpy(dtype=np.float32)
    method = method or "none"
    if method == "z_score":
        if len(x) < 2:
            # Same single-sample fallback as normalize_frame.
            offset, scale = x.mean(axis=0), x.std(axis=0, ddof=0) + 1e-10
        else:
            offset, scale = x.mean(axis=0), x.std(axis=0, ddof=1) + 1e-10
    elif method == "min_max_sample":
        offset = x.min(axis=1, keepdims=True)
        scale = x.max(axis=1, keepdims=True) - offset + 1e-10
    elif method == "none":
        offset, scale = np.zeros(x.shape[1]), np.ones(x.shape[1])
    else:
        raise ValueError(f"Unsupported normalization: {method}")
    if not (np.isfinite(offset).all() and np.isfinite(scale).all() and (scale > 0).all()):
        raise ValueError("Invalid target normalization parameters")
    return {"normalize": np.array(method), "offset": offset, "scale": scale,
            "proteins": np.asarray(frame.columns, dtype=str),
            "spot_names": np.asarray(frame.index, dtype=str)}


def inverse_target_scale(values: np.ndarray, parameters: dict) -> np.ndarray:
    """Restore N×P reconstruction or N×K×P expression to input-file scale."""
    x = np.asarray(values, dtype=np.float64)
    if x.ndim not in (2, 3):
        raise ValueError(f"Expected N×P or N×K×P array, got {x.shape}")
    if x.shape[0] != len(parameters["spot_names"]) or x.shape[-1] != len(parameters["proteins"]):
        raise ValueError("Normalization metadata does not match array shape")
    offset = np.asarray(parameters["offset"], dtype=np.float64)
    scale = np.asarray(parameters["scale"], dtype=np.float64)
    if offset.ndim == 2 and x.ndim == 3:
        offset, scale = offset[:, None, :], scale[:, None, :]
    return x * scale + offset


def export_input_scale(run_dir: str | Path, source: dict, parameters: dict) -> Path:
    """Write a self-describing input-file-scale expression export."""
    run_dir = Path(run_dir)
    for key in ["proteins", "spot_names"]:
        if list(np.asarray(source[key]).astype(str)) != list(parameters[key].astype(str)):
            raise ValueError(f"Expression and normalization {key} differ")
    if "expression_scale" in source and str(np.asarray(source["expression_scale"]).item()) != "model_normalized":
        raise ValueError("Source must be in model-normalized space; refusing a second inverse transform")
    param_path = run_dir / PARAM_FILE
    param_temporary = param_path.with_name(param_path.name + ".tmp.npz")
    np.savez_compressed(param_temporary, **parameters)
    param_temporary.replace(param_path)
    payload = {k: inverse_target_scale(source[k], parameters) for k in ["E", "E0"] if k in source}
    payload.update({k: source[k] for k in ["spot_names", "cell_types", "proteins"]})
    payload.update(expression_scale=np.array("input_file"),
                   normalization_method=parameters["normalize"],
                   normalization_offset=parameters["offset"],
                   normalization_scale=parameters["scale"],
                   scale_schema_version=np.array(1))
    path = run_dir / INPUT_FILE
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(temporary, **payload)
    temporary.replace(path)
    return path


def load_target_expression(path: str | Path, *, scale: str = "input_file",
                           spatial_path: str | Path | None = None) -> dict:
    """Read restored expression; legacy model-space files need known parameters."""
    path = Path(path)
    if path.is_dir():
        path = path / INPUT_FILE
    if scale not in {"input_file", "model_normalized"}:
        raise ValueError(f"Unknown requested scale: {scale}")
    if scale == "input_file" and path.name == MODEL_FILE and (path.parent / INPUT_FILE).exists():
        path = path.parent / INPUT_FILE
    if scale == "model_normalized" and path.name == INPUT_FILE:
        path = path.parent / MODEL_FILE
    with np.load(path, allow_pickle=True) as z:
        result = {key: z[key] for key in z.files}
    stored = str(np.asarray(result.get("expression_scale", "model_normalized")).item())
    if stored == scale:
        return result
    if stored != "model_normalized" or scale != "input_file":
        raise ValueError(f"Cannot read {stored} data as {scale}: {path}")
    param_path = path.parent / PARAM_FILE
    if param_path.exists():
        with np.load(param_path, allow_pickle=False) as z:
            parameters = {key: z[key] for key in z.files}
    else:
        config_path = path.parent / "config.json"
        if not config_path.exists():
            raise ValueError(f"Missing normalization configuration for legacy file: {path}")
        method = json.loads(config_path.read_text())["normalize"]
        if spatial_path is None:
            raise ValueError(f"Provide original spatial input for legacy inverse transform: {path}")
        frame = pd.read_csv(spatial_path, index_col=0)
        parameters = normalization_parameters(frame, method)
    for key in ["proteins", "spot_names"]:
        if list(np.asarray(result[key]).astype(str)) != list(parameters[key].astype(str)):
            raise ValueError(f"Legacy expression and normalization {key} differ: {path}")
    for key in ["E", "E0"]:
        if key in result:
            result[key] = inverse_target_scale(result[key], parameters)
    result["expression_scale"] = np.array("input_file")
    return result


def save_input_scale_target_metrics(
    run_dir: str | Path, E: np.ndarray, reconstruction: np.ndarray,
    baseline_reconstruction: np.ndarray, parameters: dict, *,
    mixture_truth_input: np.ndarray, expression_truth_input: np.ndarray | None = None,
    fractions: np.ndarray | None = None, cell_types=None, present_threshold: float = 0.01,
) -> dict:
    """Evaluate restored predictions against input-file-scale truth.

    Prediction arguments are model-normalized; truth arguments are on the
    input-file scale. Weighted CCC/Pearson use true-fraction-weighted moments.
    """
    from .metrics import (compute_array_metrics, compute_expression_metrics,
                          compute_masked_expression_metrics, compute_weighted_expression_metrics)

    run_dir = Path(run_dir)
    metrics = {}
    mixture_truth_input = np.asarray(mixture_truth_input, dtype=np.float64)
    for name, values in [("target_reconstruction", reconstruction),
                         ("target_baseline_reconstruction", baseline_reconstruction)]:
        restored = inverse_target_scale(values, parameters)
        if restored.shape != mixture_truth_input.shape:
            raise ValueError("Input-scale mixture truth must match target reconstruction shape")
        metrics.update({f"{name}_{key}": value
                        for key, value in compute_array_metrics(restored, mixture_truth_input).items()})
    if expression_truth_input is not None:
        restored = inverse_target_scale(E, parameters)
        overall, by_celltype = compute_expression_metrics(restored, expression_truth_input, cell_types)
        metrics.update({f"target_expression_{key}": value for key, value in overall.items()})
        by_celltype.to_csv(run_dir / "target_expression_by_celltype_metrics_input_scale.csv", index=False)
        if fractions is not None:
            present, present_by_celltype = compute_masked_expression_metrics(
                restored, expression_truth_input, fractions, threshold=present_threshold, cell_types=cell_types)
            metrics.update({f"target_expression_present_{key}": value for key, value in present.items()})
            present_by_celltype.to_csv(
                run_dir / "target_expression_present_by_celltype_metrics_input_scale.csv", index=False)
            weighted = compute_weighted_expression_metrics(restored, expression_truth_input, fractions)
            metrics.update({f"target_expression_weighted_{key}": value for key, value in weighted.items()})
    pd.DataFrame([metrics]).to_csv(run_dir / "metrics_input_scale.csv", index=False)
    metadata_path = run_dir / "metrics_scale.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    metadata.update({
        "input_scale_target_metrics": "input_file",
        "input_scale_metrics_file": "metrics_input_scale.csv",
        "input_scale_expression_ground_truth_available": expression_truth_input is not None,
        "input_scale_target_expression_weighted_definition":
            "weighted means, variances, and covariance using true cell fractions",
    })
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    return metrics
