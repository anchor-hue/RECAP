"""Neural modules for PSA-ProDeconv v8.1.

v8.1 keeps the v7 joint proportion/expression backbone and adds a proportion-preserving PSA refinement stage. It replaces the final
expression decoder with a Protein Signal Allocation (PSA) decoder:

1) predict a reference-anchored baseline expression E0;
2) compute spot-level unexplained protein residual R = X - sum_k pi_k E0_k;
3) learn a cell-type responsibility rho_{s,k,p};
4) allocate R back to spot-celltype expression with a mass-conserving correction.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def mlp(dims, activation=nn.LeakyReLU, dropout: float = 0.0, final_activation: Optional[nn.Module] = None):
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        is_last = i == len(dims) - 2
        if not is_last:
            layers.append(activation(0.2) if activation is nn.LeakyReLU else activation())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        elif final_activation is not None:
            layers.append(final_activation)
    return nn.Sequential(*layers)


class MaskedAutoencoder(nn.Module):
    """Masked MLP autoencoder for spatial-protein-space completion."""

    def __init__(self, n_proteins: int, latent_dim: int = 128, hidden1: int = 512, hidden2: int = 256, dropout: float = 0.1):
        super().__init__()
        self.encoder = mlp([n_proteins, hidden1, hidden2, latent_dim], dropout=dropout)
        self.decoder = mlp([latent_dim, hidden2, hidden1, n_proteins], dropout=dropout)

    def forward(self, x_masked: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x_masked))


class SharedEncoder(nn.Module):
    def __init__(self, n_proteins: int, z_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        self.net = mlp([n_proteins, 512, 256, z_dim], dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ProportionDecoder(nn.Module):
    def __init__(self, z_dim: int, n_celltypes: int, dropout: float = 0.1):
        super().__init__()
        self.net = mlp([z_dim, 64, n_celltypes], dropout=dropout)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.net(z), dim=-1)


class ProteinSignalAllocationDecoder(nn.Module):
    """Protein Signal Allocation expression decoder.

    Baseline expression:
        E0[i,k,p] = mu[k,p] + sqrt(sigma2[k,p]) * B(z_i, c_k, q_p)

    Residual allocation:
        R[i,p] = x[i,p] - sum_k pi[i,k] * E0[i,k,p]
        rho[i,k,p] = softmax_k(A(z_i, c_k, q_p) + log(pi[i,k]+eps))
        gate[i,p] = sigmoid(L(z_i, q_p))
        E[i,k,p] = E0[i,k,p] + gate[i,p] * rho[i,k,p] * R[i,p] / max(pi[i,k], pi_floor)

    When gate -> 1 and eps is small, the correction is approximately mass-conserving:
        sum_k pi_k E_kp ~= x_p.
    """

    def __init__(
        self,
        mu: torch.Tensor,
        sigma2: torch.Tensor,
        z_dim: int = 128,
        cell_emb_dim: int = 32,
        protein_emb_dim: int = 32,
        scorer_hidden: int = 128,
        dropout: float = 0.1,
        pi_floor: float = 1e-3,
        init_gate_bias: float = -1.0,
    ):
        super().__init__()
        if mu.ndim != 2:
            raise ValueError("mu must have shape (K, P).")
        self.register_buffer("mu", mu.float())
        self.register_buffer("sigma2", sigma2.float())
        K, P = mu.shape
        self.n_celltypes = K
        self.n_proteins = P
        self.pi_floor = float(pi_floor)

        self.cell_encoder = mlp([P, 128, cell_emb_dim], dropout=dropout)
        self.protein_encoder = mlp([K, 64, protein_emb_dim], dropout=dropout)
        self.cell_residual = nn.Parameter(torch.zeros(K, cell_emb_dim))
        self.protein_residual = nn.Parameter(torch.zeros(P, protein_emb_dim))

        input_dim = z_dim + cell_emb_dim + protein_emb_dim
        self.baseline_scorer = mlp([input_dim, scorer_hidden, 64, 1], dropout=dropout)
        self.responsibility_scorer = mlp([input_dim, scorer_hidden, 64, 1], dropout=dropout)
        self.gate_scorer = mlp([z_dim + protein_emb_dim, 64, 1], dropout=dropout)
        # Initialize the correction gate conservatively so training starts close to baseline E0.
        if isinstance(self.gate_scorer[-1], nn.Linear):
            nn.init.constant_(self.gate_scorer[-1].bias, init_gate_bias)

    def cell_embeddings(self) -> torch.Tensor:
        return self.cell_encoder(self.mu) + self.cell_residual

    def protein_embeddings(self) -> torch.Tensor:
        return self.protein_encoder(self.mu.t()) + self.protein_residual

    def forward_baseline(
        self,
        z: torch.Tensor,
        pi: torch.Tensor,
        protein_chunk_size: int = 256,
    ) -> dict[str, torch.Tensor]:
        """Return baseline expression E0 and baseline reconstruction x0_hat only.

        This is used in v8.1 Backbone Stage, where the model learns stable
        proportions and reference-anchored expression without allowing PSA
        residual correction to explain away proportion errors.
        """
        N = z.shape[0]
        K, P = self.n_celltypes, self.n_proteins
        c = self.cell_embeddings()
        q = self.protein_embeddings()
        sigma = torch.sqrt(torch.clamp(self.sigma2, min=1e-6))

        E0_chunks = []
        x0_chunks = []
        for start in range(0, P, protein_chunk_size):
            end = min(P, start + protein_chunk_size)
            q_chunk = q[start:end]
            pc = end - start
            z_exp = z[:, None, None, :].expand(N, K, pc, z.shape[-1])
            c_exp = c[None, :, None, :].expand(N, K, pc, c.shape[-1])
            q_exp = q_chunk[None, None, :, :].expand(N, K, pc, q.shape[-1])
            u = torch.cat([z_exp, c_exp, q_exp], dim=-1)
            u_flat = u.reshape(N * K * pc, -1)
            base_resid = self.baseline_scorer(u_flat).reshape(N, K, pc)
            E0 = self.mu[None, :, start:end] + sigma[None, :, start:end] * base_resid
            x0 = torch.einsum("nk,nkp->np", pi, E0)
            E0_chunks.append(E0)
            x0_chunks.append(x0)
        E0_all = torch.cat(E0_chunks, dim=-1)
        x0_all = torch.cat(x0_chunks, dim=-1)
        return {"E0": E0_all, "x0_hat": x0_all}

    def forward(
        self,
        z: torch.Tensor,
        pi: torch.Tensor,
        x: torch.Tensor,
        protein_chunk_size: int = 256,
    ) -> dict[str, torch.Tensor]:
        """Return final E, baseline E0, reconstruction x_hat, baseline reconstruction x0_hat,
        residual correction, gate, and responsibility entropy.
        """
        N = z.shape[0]
        K, P = self.n_celltypes, self.n_proteins
        c = self.cell_embeddings()  # (K, dc)
        q = self.protein_embeddings()  # (P, dp)
        sigma = torch.sqrt(torch.clamp(self.sigma2, min=1e-6))

        E0_chunks = []
        E_chunks = []
        x0_chunks = []
        xhat_chunks = []
        corr_chunks = []
        gate_chunks = []
        # Store rho only when feasible enough; it has the same size as E.
        rho_chunks = []
        entropy_terms = []

        pi_safe = torch.clamp(pi, min=self.pi_floor)
        log_pi = torch.log(torch.clamp(pi, min=1e-8))

        for start in range(0, P, protein_chunk_size):
            end = min(P, start + protein_chunk_size)
            q_chunk = q[start:end]
            pc = end - start

            z_exp = z[:, None, None, :].expand(N, K, pc, z.shape[-1])
            c_exp = c[None, :, None, :].expand(N, K, pc, c.shape[-1])
            q_exp = q_chunk[None, None, :, :].expand(N, K, pc, q.shape[-1])
            u = torch.cat([z_exp, c_exp, q_exp], dim=-1)
            u_flat = u.reshape(N * K * pc, -1)

            base_resid = self.baseline_scorer(u_flat).reshape(N, K, pc)
            E0 = self.mu[None, :, start:end] + sigma[None, :, start:end] * base_resid
            x0 = torch.einsum("nk,nkp->np", pi, E0)
            R = x[:, start:end] - x0

            alloc_logits = self.responsibility_scorer(u_flat).reshape(N, K, pc)
            alloc_logits = alloc_logits + log_pi[:, :, None]
            rho = F.softmax(alloc_logits, dim=1)

            z_gate = z[:, None, :].expand(N, pc, z.shape[-1])
            q_gate = q_chunk[None, :, :].expand(N, pc, q.shape[-1])
            gate_in = torch.cat([z_gate, q_gate], dim=-1)
            gate = torch.sigmoid(self.gate_scorer(gate_in.reshape(N * pc, -1)).reshape(N, pc))

            correction = gate[:, None, :] * rho * R[:, None, :] / pi_safe[:, :, None]
            E = E0 + correction
            xhat = torch.einsum("nk,nkp->np", pi, E)

            E0_chunks.append(E0)
            E_chunks.append(E)
            x0_chunks.append(x0)
            xhat_chunks.append(xhat)
            corr_chunks.append(correction)
            gate_chunks.append(gate)
            rho_chunks.append(rho)
            entropy = -(rho * torch.log(torch.clamp(rho, min=1e-8))).sum(dim=1).mean()
            entropy_terms.append(entropy)

        E0_all = torch.cat(E0_chunks, dim=-1)
        E_all = torch.cat(E_chunks, dim=-1)
        x0_all = torch.cat(x0_chunks, dim=-1)
        xhat_all = torch.cat(xhat_chunks, dim=-1)
        corr_all = torch.cat(corr_chunks, dim=-1)
        gate_all = torch.cat(gate_chunks, dim=-1)
        rho_all = torch.cat(rho_chunks, dim=-1)
        entropy_mean = torch.stack(entropy_terms).mean() if entropy_terms else torch.tensor(0.0, device=z.device)
        return {
            "E0": E0_all,
            "E": E_all,
            "x0_hat": x0_all,
            "x_hat": xhat_all,
            "correction": corr_all,
            "gate": gate_all,
            "rho": rho_all,
            "rho_entropy": entropy_mean,
        }


# ---------------------------------------------------------------------------
# Adversarial Domain Adaptation components
# ---------------------------------------------------------------------------

class GradientReversalLayer(torch.autograd.Function):
    """Gradient Reversal Layer (GRL) for adversarial domain adaptation.

    In the forward pass, this acts as identity. In the backward pass,
    it multiplies the gradient by -lambda, effectively reversing the
    gradient direction. This allows joint (single-step) training of
    encoder and discriminator — the encoder maximizes the discriminator
    loss while the discriminator minimizes it.

    Compared to label flipping with alternating optimization:
    - Single backward pass (no separate D/G steps needed)
    - More stable gradients
    - Standard in domain-adversarial training

    Reference:
        Ganin & Lempitsky (2015), "Unsupervised Domain Adaptation
        by Backpropagation." JMLR 17.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_: float = 1.0) -> torch.Tensor:
        ctx.lambda_ = lambda_
        return x

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # Reversed gradient scaled by lambda_
        return grad_output.neg() * ctx.lambda_, None


def grad_reverse(x: torch.Tensor, lambda_: float = 1.0) -> torch.Tensor:
    """Apply gradient reversal to x with scaling factor lambda_."""
    return GradientReversalLayer.apply(x, lambda_)


class DomainDiscriminator(nn.Module):
    """Binary domain discriminator with spectral normalization.

    Architecture: deeper MLP [z_dim → 128 → 64 → 32 → 1] with spectral
    normalization on hidden layers to enforce 1-Lipschitz constraint.
    This stabilizes GAN-style training and produces more informative
    gradients for the encoder.

    Uses BCEWithLogitsLoss. Source pseudo-spots labeled 0, target spatial
    spots labeled 1.
    """

    def __init__(
        self,
        z_dim: int = 128,
        dropout: float = 0.2,
        hidden_dims: tuple = (128, 64, 32),
    ):
        super().__init__()
        layers = []
        in_dim = z_dim
        for h in hidden_dims:
            linear = nn.Linear(in_dim, h)
            layers.append(nn.utils.spectral_norm(linear))
            layers.append(nn.LeakyReLU(0.2))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = h
        # Final layer: no spectral norm → allows unbounded logits for BCE
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class PSAProDeconvNet(nn.Module):
    def __init__(
        self,
        n_proteins: int,
        mu: torch.Tensor,
        sigma2: torch.Tensor,
        z_dim: int = 128,
        dropout: float = 0.2,
        cell_emb_dim: int = 32,
        protein_emb_dim: int = 32,
        protein_chunk_size: int = 256,
        pi_floor: float = 1e-3,
    ):
        super().__init__()
        self.encoder = SharedEncoder(n_proteins, z_dim=z_dim, dropout=dropout)
        self.proportion = ProportionDecoder(z_dim, mu.shape[0], dropout=dropout)
        self.expression = ProteinSignalAllocationDecoder(
            mu=mu,
            sigma2=sigma2,
            z_dim=z_dim,
            cell_emb_dim=cell_emb_dim,
            protein_emb_dim=protein_emb_dim,
            dropout=dropout,
            pi_floor=pi_floor,
        )
        self.domain = DomainDiscriminator(z_dim=z_dim, dropout=dropout)
        self.protein_chunk_size = protein_chunk_size

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def deconvolve(self, x: torch.Tensor, baseline_only: bool = False, detach_pi_for_psa: bool = False):
        z = self.encoder(x)
        pi = self.proportion(z)
        if baseline_only:
            expr_out = self.expression.forward_baseline(z, pi, protein_chunk_size=self.protein_chunk_size)
            # For a baseline-only pass, use E0/x0_hat as the active reconstruction.
            return {"z": z, "pi": pi, "E0": expr_out["E0"], "E": expr_out["E0"], "x0_hat": expr_out["x0_hat"], "x_hat": expr_out["x0_hat"]}
        pi_for_expr = pi.detach() if detach_pi_for_psa else pi
        expr_out = self.expression(z, pi_for_expr, x, protein_chunk_size=self.protein_chunk_size)
        return {"z": z, "pi": pi, **expr_out}

# Backward-compatible alias for existing trainer imports if needed.
DAProDeconvNet = PSAProDeconvNet
