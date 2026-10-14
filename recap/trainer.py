"""RECAP training pipeline with the configuration used in the manuscript."""
from __future__ import annotations

import copy
import json
import os
import random
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from itertools import cycle
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .data import (
    align_reference_to_spatial,
    compute_prototypes,
    make_pseudospots,
    make_pseudospots_multi,
    normalize_frame,
    normalize_target_expression_truth,
    read_expression_table,
    read_metadata,
)
from .expression_scale import (normalization_parameters, inverse_target_scale,
                               export_input_scale, save_input_scale_target_metrics)
from .metrics import (
    compute_fraction_metrics,
    load_expression_truth_npz,
    save_loss_curves,
)
from .models import PSAProDeconvNet, MaskedAutoencoder, grad_reverse


@dataclass
class DAProDeconvConfig:
    seed: int = 2021
    device: str = "auto"
    normalize: str = "z_score"
    batch_size: int = 64
    ae_epochs: int = 100
    main_epochs: int = 120
    backbone_epochs: int = 70
    psa_epochs: int = 50
    lr_ae: float = 1e-3
    lr_main: float = 2e-4
    lr_domain: float = 3e-4
    z_dim: int = 128
    ae_latent_dim: int = 128
    dropout: float = 0.2
    n_pseudospots: int = 5000
    sample_size: int = 50
    dirichlet_alpha: float = 0.3
    # v8.3 multi-alpha training scheme.  When set, each pseudospot is assigned
    # one concentration from this list (with `dirichlet_alpha_weights`) instead
    # of the scalar `dirichlet_alpha`.  The list/weights are a model-design
    # choice, not target-adaptive: training never reads target-generation
    # metadata.
    dirichlet_alphas: Optional[List[float]] = field(default_factory=lambda: [0.1, 0.2, 0.3])
    dirichlet_alpha_weights: Optional[List[float]] = field(default_factory=lambda: [1 / 3, 1 / 3, 1 / 3])
    pseudo_noise_std: float = 0.0
    sp_only_reliability: float = 0.3
    ae_sp_only_weight: float = 2.0
    lambda_prop: float = 1.0
    lambda_expr0: float = 0.5
    lambda_expr: float = 1.0
    lambda_rec_syn: float = 1.0
    lambda_rec_sp: float = 1.0
    lambda_proto: float = 0.1
    lambda_corr: float = 0.001
    lambda_resp_entropy: float = 0.0
    lambda_adv: float = 0.3
    adv_warmup_epochs: int = 1
    lambda_coral: float = 0.0
    lambda_adv_psa: float = 0.0        # adversarial weight in PSA stage (default 0 = disabled)
    adv_d_steps: int = 2               # number of discriminator updates per encoder update
    grad_penalty_weight: float = 0.1   # gradient penalty weight for discriminator regularization
    best_score_expr_weight: float = 0.2
    protein_chunk_size: int = 256
    pi_floor: float = 1e-3
    freeze_backbone_in_psa: bool = True
    detach_pi_in_psa: bool = True
    # v8.3 ablation switch: when True, the PSA refinement stage is skipped and
    # the final expression estimate is the reference-anchored baseline E0
    # (final E == E0, no residual/correction branch).  Used only for the
    # NoPSA / NoRefinement ablation; all other model logic is unchanged.
    disable_psa_refinement: bool = False
    expression_present_threshold: float = 0.01
    num_workers: int = 0
    target_type: str = "real"


def set_seed(seed: int) -> None:
    torch.set_num_threads(1)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def set_requires_grad(module: nn.Module, flag: bool) -> None:
    for p in module.parameters():
        p.requires_grad_(flag)


def _validate_training_data(**arrays: np.ndarray) -> None:
    """Check training data for NaN / Inf before entering the model.

    Raises ValueError with the name of the first offending array if any non-finite
    values are found.
    """
    for name, arr in arrays.items():
        if arr is None:
            continue
        arr = np.asarray(arr)
        if not np.isfinite(arr).all():
            nan_count = int(np.isnan(arr).sum())
            inf_count = int(np.isinf(arr).sum())
            raise ValueError(
                f"Training data '{name}' contains non-finite values: "
                f"{nan_count} NaN, {inf_count} Inf out of {arr.size} elements."
            )
        if arr.size == 0:
            raise ValueError(f"Training data '{name}' is empty.")


def weighted_mse(pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return ((pred - target) ** 2 * weight).sum() / (weight.sum() + 1e-8)


def _compute_coral_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Deep CORAL loss: Frobenius norm of covariance difference.

    Aligns second-order statistics of source and target feature distributions.
    More stable than adversarial training and provides complementary signal.

    Reference: Sun & Saenko (2016), "Deep CORAL: Correlation Alignment for
    Deep Domain Adaptation."
    """
    if x.shape[0] <= 1 or y.shape[0] <= 1:
        return torch.tensor(0.0, device=x.device)
    d = x.shape[1]
    # Center the features
    x_centered = x - x.mean(dim=0, keepdim=True)
    y_centered = y - y.mean(dim=0, keepdim=True)
    # Covariance matrices
    cov_x = (x_centered.t() @ x_centered) / (x.shape[0] - 1)
    cov_y = (y_centered.t() @ y_centered) / (y.shape[0] - 1)
    # Frobenius norm of difference, normalized by feature dimension
    return ((cov_x - cov_y) ** 2).sum() / (4 * d * d)


class DAProDeconvPipeline:
    def __init__(self, config: Optional[DAProDeconvConfig] = None):
        self.config = config or DAProDeconvConfig()
        set_seed(self.config.seed)
        self.device = resolve_device(self.config.device)
        self.alignment = None
        self.cell_types = None
        self.model: Optional[PSAProDeconvNet] = None
        self.ae: Optional[MaskedAutoencoder] = None
        self.ae_history: Dict[str, list] = {}
        self.main_history: Dict[str, list] = {}

    def _lambda_adv_now(self, epoch: int, total_epochs: int) -> float:
        """Compute current adversarial loss weight with sigmoid-like warmup.

        Uses a smooth sigmoid schedule that ramps up more quickly than linear:
        - Starts near 0 at epoch 0
        - Reaches 0.5 * lambda_adv around adv_warmup_epochs
        - Approaches lambda_adv asymptotically
        """
        cfg = self.config
        if cfg.lambda_adv <= 0:
            return 0.0
        if cfg.adv_warmup_epochs <= 0:
            return cfg.lambda_adv
        # Sigmoid schedule: center at warmup_epochs, scale controls steepness
        center = float(cfg.adv_warmup_epochs)
        scale = max(1.0, center / 4.0)  # controls smoothness
        # Map epoch to [-center/scale, (total-1-center)/scale]
        progress = (epoch - center) / scale
        # Sigmoid: 0→1 as progress goes from -inf to +inf
        factor = 1.0 / (1.0 + np.exp(-progress))
        return float(cfg.lambda_adv * factor)

    def _compute_gradient_penalty(self, z_src: torch.Tensor, z_tgt: torch.Tensor) -> torch.Tensor:
        """WGAN-style gradient penalty on the discriminator.

        Penalizes deviation of gradient norm from 1 at random interpolations
        between source and target latent codes. This enforces the 1-Lipschitz
        constraint softly, complementing spectral normalization.
        """
        if z_src.shape[0] == 0 or z_tgt.shape[0] == 0:
            return torch.tensor(0.0, device=z_src.device)
        # Match batch sizes for interpolation
        n = min(z_src.shape[0], z_tgt.shape[0])
        z_s = z_src[:n]
        z_t = z_tgt[:n]
        alpha = torch.rand(n, 1, device=z_src.device)
        interpolates = alpha * z_s + (1 - alpha) * z_t
        interpolates.requires_grad_(True)
        d_interpolates = self.model.domain(interpolates)
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(d_interpolates),
            create_graph=True,
            retain_graph=True,
        )[0]
        gradient_norm = gradients.view(n, -1).norm(2, dim=1)
        return ((gradient_norm - 1.0) ** 2).mean()

    def _train_ae(self, target_x: np.ndarray, shared_mask: np.ndarray, sp_only_mask: np.ndarray) -> MaskedAutoencoder:
        cfg = self.config
        P = target_x.shape[1]
        ae = MaskedAutoencoder(P, latent_dim=cfg.ae_latent_dim, dropout=cfg.dropout).to(self.device)
        opt = torch.optim.Adam(ae.parameters(), lr=cfg.lr_ae)
        x = torch.tensor(target_x, dtype=torch.float32)
        loader = DataLoader(TensorDataset(x), batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
        mask = torch.tensor(shared_mask.astype(np.float32), device=self.device)[None, :]
        w = torch.ones(P, device=self.device)
        w[torch.tensor(sp_only_mask, device=self.device)] = cfg.ae_sp_only_weight
        history = defaultdict(list)
        for epoch in range(cfg.ae_epochs):
            ae.train()
            total = 0.0
            for (xb,) in loader:
                xb = xb.to(self.device)
                pred = ae(xb * mask)
                loss = ((pred - xb) ** 2 * w[None, :]).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += loss.item()
            history["ae_loss"].append(total / max(1, len(loader)))
        self.ae_history = history
        return ae

    @torch.no_grad()
    def _impute_reference(self, ref_tilde: np.ndarray, shared_mask: np.ndarray, ae: MaskedAutoencoder) -> np.ndarray:
        ae.eval()
        ref = torch.tensor(ref_tilde, dtype=torch.float32, device=self.device)
        loader = DataLoader(TensorDataset(ref), batch_size=self.config.batch_size, shuffle=False)
        chunks = []
        for (xb,) in loader:
            chunks.append(ae(xb).cpu().numpy())
        recon = np.vstack(chunks).astype(np.float32)
        completed = ref_tilde.astype(np.float32).copy()
        completed[:, ~shared_mask] = recon[:, ~shared_mask]
        return completed

    def fit_from_tables(
        self,
        ref_df: pd.DataFrame,
        ref_labels: pd.Series,
        spatial_df: pd.DataFrame,
        target_labels: Optional[pd.DataFrame] = None,
        target_expression: Optional[np.ndarray] = None,
        outdir: Optional[str] = None,
        cell_types: Optional[Sequence[str]] = None,
    ) -> Dict[str, object]:
        cfg = self.config
        outdir = outdir or "daprodeconv_results"
        os.makedirs(outdir, exist_ok=True)
        with open(os.path.join(outdir, "config.json"), "w") as f:
            json.dump(asdict(cfg), f, indent=2)

        raw_spatial_df = spatial_df.copy()
        # Output metadata only: training normalization and optimization are unchanged.
        self.target_normalization = normalization_parameters(raw_spatial_df, cfg.normalize)
        target_expression = normalize_target_expression_truth(raw_spatial_df, target_expression, cfg.normalize)
        ref_df = normalize_frame(ref_df, cfg.normalize)
        spatial_df = normalize_frame(spatial_df, cfg.normalize)
        ref_tilde_df, spatial_aligned_df, alignment = align_reference_to_spatial(ref_df, spatial_df, cfg.sp_only_reliability)
        self.alignment = alignment

        ae = self._train_ae(spatial_aligned_df.to_numpy(np.float32), alignment.shared_mask, alignment.sp_only_mask)
        self.ae = ae
        ref_completed_np = self._impute_reference(ref_tilde_df.to_numpy(np.float32), alignment.shared_mask, ae)
        ref_completed = pd.DataFrame(ref_completed_np, index=ref_tilde_df.index, columns=alignment.target_proteins)
        ref_completed.to_csv(os.path.join(outdir, "reference_completed.csv"))
        pd.DataFrame({
            "protein": alignment.target_proteins,
            "shared": alignment.shared_mask,
            "sp_only": alignment.sp_only_mask,
            "reliability": alignment.reliability,
        }).to_csv(os.path.join(outdir, "protein_alignment.csv"), index=False)

        if cfg.dirichlet_alphas:
            # v8.3 fixed multi-alpha training scheme: one shared pseudospot
            # budget allocated deterministically across the alpha levels.
            X_syn, Pi_syn, E_syn, cell_types, alloc_df, comp_diag_df = make_pseudospots_multi(
                ref_completed,
                ref_labels,
                alphas=cfg.dirichlet_alphas,
                alpha_weights=cfg.dirichlet_alpha_weights,
                cell_types=cell_types,
                n_pseudospots=cfg.n_pseudospots,
                sample_size=cfg.sample_size,
                noise_std=cfg.pseudo_noise_std,
                seed=cfg.seed,
            )
        else:
            # Legacy scalar mode (backward compatible, bit-identical pseudospots).
            X_syn, Pi_syn, E_syn, cell_types, alloc_df, comp_diag_df = make_pseudospots(
                ref_completed,
                ref_labels,
                cell_types=cell_types,
                n_pseudospots=cfg.n_pseudospots,
                sample_size=cfg.sample_size,
                alpha=cfg.dirichlet_alpha,
                noise_std=cfg.pseudo_noise_std,
                seed=cfg.seed,
                collect_diagnostics=True,
            )
        alloc_df.to_csv(os.path.join(outdir, "alpha_allocation.csv"), index=False)
        comp_diag_df.to_csv(os.path.join(outdir, "pseudospot_composition_diagnostics.csv"), index=False)
        self.cell_types = cell_types
        mu, sigma2 = compute_prototypes(ref_completed, ref_labels, cell_types)
        X_sp = spatial_aligned_df.to_numpy(np.float32)
        _validate_training_data(X_syn=X_syn, Pi_syn=Pi_syn, E_syn=E_syn, X_sp=X_sp, mu=mu, sigma2=sigma2, reliability=alignment.reliability)

        outputs = self._train_main(
            X_syn=X_syn,
            Pi_syn=Pi_syn,
            E_syn=E_syn,
            X_sp=X_sp,
            mu=mu,
            sigma2=sigma2,
            reliability=alignment.reliability,
            target_labels=target_labels,
            target_expression=target_expression,
            outdir=outdir,
            spot_names=list(spatial_aligned_df.index),
            proteins=alignment.target_proteins,
            cell_types=cell_types,
        )
        save_loss_curves(outdir, self.ae_history, prefix="stage1_ae")
        return outputs

    def fit_from_files(
        self,
        ref_path: str,
        spatial_path: str,
        ref_celltype_col: str,
        ref_metadata_path: Optional[str] = None,
        target_fraction_path: Optional[str] = None,
        target_expression_path: Optional[str] = None,
        outdir: str = "daprodeconv_results",
        cell_types: Optional[Sequence[str]] = None,
    ) -> Dict[str, object]:
        ref_df, labels_from_h5ad = read_expression_table(ref_path, celltype_col=ref_celltype_col if ref_path.endswith(".h5ad") else None)
        if labels_from_h5ad is not None:
            ref_labels = labels_from_h5ad
        elif ref_metadata_path is not None:
            ref_labels = read_metadata(ref_metadata_path, ref_celltype_col, index_like=ref_df.index)
        else:
            raise ValueError("Provide ref_metadata_path for CSV/TSV reference input, or use h5ad with ref_celltype_col.")
        spatial_df, _ = read_expression_table(spatial_path)
        target_labels = pd.read_csv(target_fraction_path, index_col=0) if target_fraction_path else None
        target_expression = load_expression_truth_npz(target_expression_path) if target_expression_path else None
        return self.fit_from_tables(ref_df, ref_labels, spatial_df, target_labels, target_expression, outdir, cell_types)

    def _train_main(
        self,
        X_syn: np.ndarray,
        Pi_syn: np.ndarray,
        E_syn: np.ndarray,
        X_sp: np.ndarray,
        mu: np.ndarray,
        sigma2: np.ndarray,
        reliability: np.ndarray,
        target_labels: Optional[pd.DataFrame],
        target_expression: Optional[np.ndarray],
        outdir: str,
        spot_names: Sequence[str],
        proteins: Sequence[str],
        cell_types: Sequence[str],
    ) -> Dict[str, object]:
        """Train v8.1 in two stages.

        Stage A, Backbone training:
            train encoder + proportion decoder + baseline expression decoder using E0/x0_hat.
            The PSA residual allocator is structurally inactive in the loss.
            Adversarial domain adaptation via Gradient Reversal Layer (GRL) and
            optional CORAL loss aligns source (pseudo-spot) and target (spatial) latent
            distributions.

        Stage B, PSA refinement:
            freeze encoder/proportion by default and train the expression/PSA module with
            stop-gradient proportions. This preserves v7-style proportion performance while
            allowing PSA to improve E and reconstruction.
            Optionally applies domain-adversarial training to the expression module.
        """
        cfg = self.config
        source_ds = TensorDataset(
            torch.tensor(X_syn, dtype=torch.float32),
            torch.tensor(Pi_syn, dtype=torch.float32),
            torch.tensor(E_syn, dtype=torch.float32),
        )
        target_ds = TensorDataset(torch.tensor(X_sp, dtype=torch.float32))
        source_loader = DataLoader(source_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=False, num_workers=cfg.num_workers)
        target_loader = DataLoader(target_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=False, num_workers=cfg.num_workers)

        model = PSAProDeconvNet(
            n_proteins=X_sp.shape[1],
            mu=torch.tensor(mu, dtype=torch.float32),
            sigma2=torch.tensor(sigma2, dtype=torch.float32),
            z_dim=cfg.z_dim,
            dropout=cfg.dropout,
            protein_chunk_size=cfg.protein_chunk_size,
            pi_floor=cfg.pi_floor,
        ).to(self.device)
        self.model = model

        bce_logits = nn.BCEWithLogitsLoss(reduction="mean")
        rel = torch.tensor(reliability, dtype=torch.float32, device=self.device)

        # ------------------------------------------------------------------
        # Stage A: v7-style proportion backbone with baseline E0 only.
        # ------------------------------------------------------------------
        opt_backbone = torch.optim.Adam(
            list(model.encoder.parameters()) + list(model.proportion.parameters()) + list(model.expression.parameters()),
            lr=cfg.lr_main,
        )
        opt_d = torch.optim.Adam(model.domain.parameters(), lr=cfg.lr_domain)
        backbone_history = defaultdict(list)
        best_backbone_state = None
        best_backbone_score = float("inf")
        best_backbone_epoch = -1
        use_adv = cfg.lambda_adv > 0
        use_coral = cfg.lambda_coral > 0

        for epoch in range(cfg.backbone_epochs):
            model.train()
            target_iter = cycle(target_loader)
            epoch_sums = defaultdict(float)
            n_batches = 0
            lambda_adv_now = self._lambda_adv_now(epoch, cfg.backbone_epochs)
            for xb_syn, pi_syn, e_syn in source_loader:
                (xb_sp,) = next(target_iter)
                xb_syn = xb_syn.to(self.device)
                pi_syn = pi_syn.to(self.device)
                e_syn = e_syn.to(self.device)
                xb_sp = xb_sp.to(self.device)
                N_syn = xb_syn.shape[0]
                N_sp = xb_sp.shape[0]

                # ----------------------------------------------------------
                # Step A1: Train discriminator on detached latents.
                # Multiple D steps per G step keep the discriminator strong.
                # ----------------------------------------------------------
                loss_d = torch.tensor(0.0, device=self.device)
                loss_gp = torch.tensor(0.0, device=self.device)
                if use_adv and lambda_adv_now > 0:
                    for _ in range(cfg.adv_d_steps):
                        set_requires_grad(model.domain, True)
                        set_requires_grad(model.encoder, False)
                        with torch.no_grad():
                            z_syn_det = model.encoder(xb_syn)
                            z_sp_det = model.encoder(xb_sp)
                        logits_syn_d = model.domain(z_syn_det)
                        logits_sp_d = model.domain(z_sp_det)
                        loss_d_step = bce_logits(logits_syn_d, torch.zeros(N_syn, 1, device=self.device)) + \
                                      bce_logits(logits_sp_d, torch.ones(N_sp, 1, device=self.device))
                        # Gradient penalty for Lipschitz regularization
                        if cfg.grad_penalty_weight > 0:
                            loss_gp = self._compute_gradient_penalty(z_syn_det, z_sp_det)
                            loss_d_total = loss_d_step + cfg.grad_penalty_weight * loss_gp
                        else:
                            loss_d_total = loss_d_step
                        opt_d.zero_grad()
                        loss_d_total.backward()
                        opt_d.step()
                        loss_d = loss_d_step.detach()

                # ----------------------------------------------------------
                # Step A2: Train encoder + task heads.
                # Uses GRL for adversarial loss (joint training, no label flipping).
                # ----------------------------------------------------------
                set_requires_grad(model.domain, False)
                set_requires_grad(model.encoder, True)
                out_syn = model.deconvolve(xb_syn, baseline_only=True)
                out_sp = model.deconvolve(xb_sp, baseline_only=True)

                # --- Task losses ---
                loss_prop = torch.mean((out_syn["pi"] - pi_syn) ** 2)
                weight_expr = pi_syn[:, :, None] * rel[None, None, :]
                loss_expr0 = weighted_mse(out_syn["E0"], e_syn, weight_expr)
                loss_rec0_syn = torch.mean((out_syn["x0_hat"] - xb_syn) ** 2)
                loss_rec0_sp = torch.mean((out_sp["x0_hat"] - xb_sp) ** 2)

                proto_weight = out_sp["pi"][:, :, None] * rel[None, None, :]
                proto_num = ((out_sp["E0"] - model.expression.mu[None, :, :]) ** 2) / (model.expression.sigma2[None, :, :] + 1e-6)
                loss_proto = (proto_num * proto_weight).sum() / (proto_weight.sum() + 1e-8)

                # --- Adversarial loss via GRL ---
                # With GRL, we use TRUE labels (src→0, tgt→1) and the GRL
                # automatically trains encoder to maximize D's loss.
                loss_adv = torch.tensor(0.0, device=self.device)
                if use_adv and lambda_adv_now > 0:
                    z_syn_grl = grad_reverse(out_syn["z"], lambda_adv_now)
                    z_sp_grl = grad_reverse(out_sp["z"], lambda_adv_now)
                    logits_syn_adv = model.domain(z_syn_grl)
                    logits_sp_adv = model.domain(z_sp_grl)
                    # True labels: source=0, target=1
                    # GRL reverses encoder gradient → encoder maximizes this loss → domain confusion
                    loss_adv = bce_logits(logits_syn_adv, torch.zeros(N_syn, 1, device=self.device)) + \
                               bce_logits(logits_sp_adv, torch.ones(N_sp, 1, device=self.device))

                # --- CORAL loss (second-order statistics alignment) ---
                loss_coral = torch.tensor(0.0, device=self.device)
                if use_coral and cfg.lambda_coral > 0:
                    loss_coral = _compute_coral_loss(out_syn["z"], out_sp["z"])

                # --- Total loss ---
                loss_task = (
                    cfg.lambda_prop * loss_prop
                    + cfg.lambda_expr0 * loss_expr0
                    + cfg.lambda_rec_syn * loss_rec0_syn
                    + cfg.lambda_rec_sp * loss_rec0_sp
                    + cfg.lambda_proto * loss_proto
                )
                loss_g = loss_task + loss_adv + cfg.lambda_coral * loss_coral
                opt_backbone.zero_grad()
                loss_g.backward()
                opt_backbone.step()

                vals = {
                    "loss_D": loss_d.item(),
                    "loss_G": loss_g.item(),
                    "loss_task": loss_task.item(),
                    "loss_prop": loss_prop.item(),
                    "loss_expr0": loss_expr0.item(),
                    "loss_rec0_syn": loss_rec0_syn.item(),
                    "loss_rec0_sp": loss_rec0_sp.item(),
                    "loss_proto": loss_proto.item(),
                    "loss_adv_grl": loss_adv.item(),
                    "loss_coral": loss_coral.item(),
                    "loss_gp": loss_gp.item() if isinstance(loss_gp, torch.Tensor) else float(loss_gp),
                    "lambda_adv_now": lambda_adv_now,
                }
                for k, v in vals.items():
                    epoch_sums[k] += float(v)
                n_batches += 1

            for k, v in epoch_sums.items():
                backbone_history[k].append(v / max(1, n_batches))
            score = backbone_history["loss_rec0_sp"][-1] + backbone_history["loss_prop"][-1] + cfg.best_score_expr_weight * backbone_history["loss_expr0"][-1]
            backbone_history["best_score"].append(score)
            if score < best_backbone_score:
                best_backbone_score = score
                best_backbone_epoch = epoch + 1
                best_backbone_state = copy.deepcopy(model.state_dict())
            if (epoch + 1) % max(1, cfg.backbone_epochs // 10) == 0 or epoch == 0:
                keys = ["loss_G", "loss_D", "loss_task", "loss_prop", "loss_expr0", "loss_rec0_syn", "loss_rec0_sp",
                        "loss_proto", "loss_adv_grl", "loss_coral", "lambda_adv_now", "best_score"]
                existing = [k for k in keys if k in backbone_history]
                print(f"[Backbone] Epoch {epoch+1:03d}/{cfg.backbone_epochs}: " + ", ".join(f"{k}={backbone_history[k][-1]:.4f}" for k in existing))

        if best_backbone_state is not None:
            model.load_state_dict(best_backbone_state)
        torch.save(model.state_dict(), os.path.join(outdir, "backbone_model_best.pt"))
        with open(os.path.join(outdir, "backbone_best_checkpoint.json"), "w") as f:
            json.dump({"best_epoch": best_backbone_epoch, "best_score": best_backbone_score}, f, indent=2)
        save_loss_curves(outdir, backbone_history, prefix="stage4_backbone")

        # ------------------------------------------------------------------
        # Stage B: proportion-preserving PSA refinement.
        # ------------------------------------------------------------------
        if cfg.disable_psa_refinement:
            # NoPSA / NoRefinement ablation: keep the backbone-only model as the
            # final model, so the predicted expression is the baseline E0 (the
            # correction/responsibility/gate branch is never trained or used).
            self.main_history = {"backbone": backbone_history, "psa": {}}
            torch.save(model.state_dict(), os.path.join(outdir, "daprodeconv_model_best.pt"))
            with open(os.path.join(outdir, "best_checkpoint.json"), "w") as f:
                json.dump({
                    "backbone_best_epoch": best_backbone_epoch,
                    "backbone_best_score": best_backbone_score,
                    "psa_best_epoch": 0,
                    "psa_best_score": None,
                    "disable_psa_refinement": True,
                }, f, indent=2)
            self.model = model
            return self.predict_and_save(
                X_sp=X_sp,
                spot_names=spot_names,
                proteins=proteins,
                cell_types=cell_types,
                outdir=outdir,
                target_labels=target_labels,
                target_expression=target_expression,
            )

        if cfg.freeze_backbone_in_psa:
            set_requires_grad(model.encoder, False)
            set_requires_grad(model.proportion, False)
        set_requires_grad(model.domain, False)
        set_requires_grad(model.expression, True)
        opt_psa = torch.optim.Adam(model.expression.parameters(), lr=cfg.lr_main)

        # Optionally re-enable discriminator for PSA-stage adversarial training
        psa_use_adv = cfg.lambda_adv_psa > 0
        if psa_use_adv:
            set_requires_grad(model.domain, True)
            opt_d_psa = torch.optim.Adam(model.domain.parameters(), lr=cfg.lr_domain)

        psa_history = defaultdict(list)
        best_psa_state = None
        best_psa_score = float("inf")
        best_psa_epoch = -1

        for epoch in range(cfg.psa_epochs):
            model.expression.train()
            if cfg.freeze_backbone_in_psa:
                model.encoder.eval()
                model.proportion.eval()
            else:
                model.encoder.train()
                model.proportion.train()
            target_iter = cycle(target_loader)
            epoch_sums = defaultdict(float)
            n_batches = 0
            lambda_adv_psa_now = self._lambda_adv_now(epoch, cfg.psa_epochs) if psa_use_adv else 0.0

            for xb_syn, pi_syn, e_syn in source_loader:
                (xb_sp,) = next(target_iter)
                xb_syn = xb_syn.to(self.device)
                pi_syn = pi_syn.to(self.device)
                e_syn = e_syn.to(self.device)
                xb_sp = xb_sp.to(self.device)

                # Compute fixed/stopped proportions. With freeze_backbone_in_psa=True, this is the v7 backbone output.
                if cfg.freeze_backbone_in_psa:
                    with torch.no_grad():
                        z_syn = model.encoder(xb_syn)
                        pi_hat_syn = model.proportion(z_syn)
                        z_sp = model.encoder(xb_sp)
                        pi_hat_sp = model.proportion(z_sp)
                    z_syn_for_expr = z_syn.detach()
                    z_sp_for_expr = z_sp.detach()
                else:
                    z_syn_for_expr = model.encoder(xb_syn)
                    pi_hat_syn = model.proportion(z_syn_for_expr)
                    z_sp_for_expr = model.encoder(xb_sp)
                    pi_hat_sp = model.proportion(z_sp_for_expr)

                pi_expr_syn = pi_hat_syn.detach() if cfg.detach_pi_in_psa else pi_hat_syn
                pi_expr_sp = pi_hat_sp.detach() if cfg.detach_pi_in_psa else pi_hat_sp
                out_expr_syn = model.expression(z_syn_for_expr, pi_expr_syn, xb_syn, protein_chunk_size=cfg.protein_chunk_size)
                out_expr_sp = model.expression(z_sp_for_expr, pi_expr_sp, xb_sp, protein_chunk_size=cfg.protein_chunk_size)

                # Optional: PSA-stage adversarial training
                loss_d_psa = torch.tensor(0.0, device=self.device)
                if psa_use_adv and lambda_adv_psa_now > 0:
                    set_requires_grad(model.domain, True)
                    with torch.no_grad():
                        z_syn_d = z_syn_for_expr.detach() if not cfg.freeze_backbone_in_psa else z_syn_for_expr
                        z_sp_d = z_sp_for_expr.detach() if not cfg.freeze_backbone_in_psa else z_sp_for_expr
                    logits_syn_d = model.domain(z_syn_d)
                    logits_sp_d = model.domain(z_sp_d)
                    loss_d_psa = bce_logits(logits_syn_d, torch.zeros(z_syn_d.shape[0], 1, device=self.device)) + \
                                 bce_logits(logits_sp_d, torch.ones(z_sp_d.shape[0], 1, device=self.device))
                    opt_d_psa.zero_grad()
                    loss_d_psa.backward()
                    opt_d_psa.step()
                    set_requires_grad(model.domain, False)

                # Proportion loss is monitored only in the PSA stage when proportions are frozen/detached.
                loss_prop_monitor = torch.mean((pi_hat_syn.detach() - pi_syn) ** 2)
                weight_expr = pi_syn[:, :, None] * rel[None, None, :]
                loss_expr0 = weighted_mse(out_expr_syn["E0"], e_syn, weight_expr)
                loss_expr = weighted_mse(out_expr_syn["E"], e_syn, weight_expr)
                loss_rec_syn = torch.mean((out_expr_syn["x_hat"] - xb_syn) ** 2)
                loss_rec_sp = torch.mean((out_expr_sp["x_hat"] - xb_sp) ** 2)
                # Also keep baseline target reconstruction as a monitor; it should preserve v7-style behavior.
                loss_rec0_sp = torch.mean((out_expr_sp["x0_hat"] - xb_sp) ** 2)

                proto_weight = pi_expr_sp[:, :, None] * rel[None, None, :]
                proto_num = ((out_expr_sp["E0"] - model.expression.mu[None, :, :]) ** 2) / (model.expression.sigma2[None, :, :] + 1e-6)
                loss_proto = (proto_num * proto_weight).sum() / (proto_weight.sum() + 1e-8)
                corr_num = (out_expr_sp["correction"] ** 2) / (model.expression.sigma2[None, :, :] + 1e-6)
                loss_corr = (corr_num * proto_weight).sum() / (proto_weight.sum() + 1e-8)
                loss_resp_entropy = out_expr_sp["rho_entropy"]

                loss_psa = (
                    cfg.lambda_expr0 * loss_expr0
                    + cfg.lambda_expr * loss_expr
                    + cfg.lambda_rec_syn * loss_rec_syn
                    + cfg.lambda_rec_sp * loss_rec_sp
                    + cfg.lambda_proto * loss_proto
                    + cfg.lambda_corr * loss_corr
                    + cfg.lambda_resp_entropy * loss_resp_entropy
                )
                opt_psa.zero_grad()
                loss_psa.backward()
                opt_psa.step()

                vals = {
                    "loss_PSA": loss_psa.item(),
                    "loss_prop_monitor": loss_prop_monitor.item(),
                    "loss_expr0": loss_expr0.item(),
                    "loss_expr": loss_expr.item(),
                    "loss_rec_syn": loss_rec_syn.item(),
                    "loss_rec_sp": loss_rec_sp.item(),
                    "loss_rec0_sp": loss_rec0_sp.item(),
                    "loss_proto": loss_proto.item(),
                    "loss_corr": loss_corr.item(),
                    "loss_resp_entropy": loss_resp_entropy.item(),
                }
                if psa_use_adv:
                    vals["loss_D_psa"] = loss_d_psa.item()
                for k, v in vals.items():
                    epoch_sums[k] += float(v)
                n_batches += 1

            for k, v in epoch_sums.items():
                psa_history[k].append(v / max(1, n_batches))
            score = psa_history["loss_rec_sp"][-1] + cfg.best_score_expr_weight * (psa_history["loss_expr"][-1] + psa_history["loss_expr0"][-1]) + cfg.lambda_corr * psa_history["loss_corr"][-1]
            psa_history["best_score"].append(score)
            if score < best_psa_score:
                best_psa_score = score
                best_psa_epoch = epoch + 1
                best_psa_state = copy.deepcopy(model.state_dict())
            if (epoch + 1) % max(1, cfg.psa_epochs // 10) == 0 or epoch == 0:
                keys = ["loss_PSA", "loss_prop_monitor", "loss_expr0", "loss_expr", "loss_rec_syn", "loss_rec_sp",
                        "loss_rec0_sp", "loss_proto", "loss_corr", "loss_resp_entropy", "best_score"]
                if psa_use_adv:
                    keys.insert(1, "loss_D_psa")
                existing = [k for k in keys if k in psa_history]
                print(f"[PSA] Epoch {epoch+1:03d}/{cfg.psa_epochs}: " + ", ".join(f"{k}={psa_history[k][-1]:.4f}" for k in existing))

        if best_psa_state is not None:
            model.load_state_dict(best_psa_state)
        self.main_history = {"backbone": backbone_history, "psa": psa_history}
        save_loss_curves(outdir, psa_history, prefix="stage5_psa")
        torch.save(model.state_dict(), os.path.join(outdir, "daprodeconv_model_best.pt"))
        with open(os.path.join(outdir, "best_checkpoint.json"), "w") as f:
            json.dump({
                "backbone_best_epoch": best_backbone_epoch,
                "backbone_best_score": best_backbone_score,
                "psa_best_epoch": best_psa_epoch,
                "psa_best_score": best_psa_score,
            }, f, indent=2)

        self.model = model
        return self.predict_and_save(
            X_sp=X_sp,
            spot_names=spot_names,
            proteins=proteins,
            cell_types=cell_types,
            outdir=outdir,
            target_labels=target_labels,
            target_expression=target_expression,
        )

    @torch.no_grad()
    def _predict_arrays(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if self.model is None:
            raise RuntimeError("Model is not trained.")
        self.model.eval()
        ds = TensorDataset(torch.tensor(X, dtype=torch.float32))
        loader = DataLoader(ds, batch_size=self.config.batch_size, shuffle=False)
        pi_list, xhat_list, e_list, x0_list, e0_list = [], [], [], [], []
        for (xb,) in loader:
            out = self.model.deconvolve(xb.to(self.device), baseline_only=self.config.disable_psa_refinement)
            pi_list.append(out["pi"].cpu().numpy())
            xhat_list.append(out["x_hat"].cpu().numpy())
            e_list.append(out["E"].cpu().numpy())
            x0_list.append(out["x0_hat"].cpu().numpy())
            e0_list.append(out["E0"].cpu().numpy())
        return (
            np.vstack(pi_list),
            np.vstack(xhat_list),
            np.concatenate(e_list, axis=0),
            np.vstack(x0_list),
            np.concatenate(e0_list, axis=0),
        )

    @torch.no_grad()
    def predict_and_save(
        self,
        X_sp: np.ndarray,
        spot_names: Sequence[str],
        proteins: Sequence[str],
        cell_types: Sequence[str],
        outdir: str,
        target_labels: Optional[pd.DataFrame] = None,
        target_expression: Optional[np.ndarray] = None,
    ) -> Dict[str, object]:
        """Save final predictions and evaluate target expression on input-file scale."""
        if self.model is None:
            raise RuntimeError("Model is not trained.")
        if not hasattr(self, "target_normalization"):
            raise RuntimeError("Target normalization metadata is required for scale restoration.")

        pi, xhat, E, x0hat, E0 = self._predict_arrays(X_sp)
        pi_df = pd.DataFrame(pi, index=spot_names, columns=cell_types)
        pi_df.to_csv(os.path.join(outdir, "target_celltype_proportions.csv"))

        expression_payload = dict(
            E=E, E0=E0, spot_names=np.array(spot_names),
            cell_types=np.array(cell_types), proteins=np.array(proteins),
            expression_scale=np.array("model_normalized"),
        )
        export_input_scale(outdir, expression_payload, self.target_normalization)
        xhat_input_df = pd.DataFrame(
            inverse_target_scale(xhat, self.target_normalization),
            index=spot_names, columns=proteins,
        )
        xhat_input_df.to_csv(os.path.join(outdir, "target_reconstruction_input_scale.csv"))
        pd.DataFrame(
            inverse_target_scale(x0hat, self.target_normalization),
            index=spot_names, columns=proteins,
        ).to_csv(os.path.join(outdir, "target_baseline_reconstruction_input_scale.csv"))

        metrics: Dict[str, float] = {}
        target_fraction_array = None
        if target_labels is not None:
            missing = [c for c in cell_types if c not in target_labels.columns]
            if missing:
                raise ValueError(f"target_fractions missing cell types: {missing}")
            if set(spot_names).issubset(set(target_labels.index)):
                target_fraction_array = target_labels.loc[spot_names, cell_types].to_numpy(dtype=np.float32)
            else:
                target_fraction_array = target_labels[cell_types].to_numpy(dtype=np.float32)
            metrics.update({
                f"target_fraction_{key}": value
                for key, value in compute_fraction_metrics(pi_df, target_labels).items()
            })

        if target_expression is not None and target_expression.shape != E.shape:
            raise ValueError(
                f"target_expression shape must match predicted E: {target_expression.shape} versus {E.shape}"
            )
        restored_metrics = save_input_scale_target_metrics(
            outdir, E, xhat, x0hat, self.target_normalization,
            mixture_truth_input=inverse_target_scale(X_sp, self.target_normalization),
            expression_truth_input=(
                inverse_target_scale(target_expression, self.target_normalization)
                if target_expression is not None else None
            ),
            fractions=target_fraction_array,
            cell_types=cell_types,
            present_threshold=self.config.expression_present_threshold,
        )
        metrics.update(restored_metrics)
        pd.DataFrame([metrics]).to_csv(os.path.join(outdir, "metrics.csv"), index=False)
        return {
            "proportions": pi_df,
            "expression_input_scale": inverse_target_scale(E, self.target_normalization),
            "reconstruction_input_scale": xhat_input_df,
            "metrics": metrics,
        }
