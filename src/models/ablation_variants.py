"""
SleepMamba reproduction — Table 7: Architecture ablation
================================================================================
Implements the 6 variants + full-model baseline for Table 7 (Section 5.5).

PROTOCOL (per our earlier discussion, since the paper's "three independent
runs" doesn't specify a data split): FIXED single train/test split (fold 9
as held-out test, folds 0-8 as train -- the exact split already validated
in our very first sanity check), 3 different random seeds per variant.
This is our interpretation of "stability" as training-seed stochasticity,
not fold-to-fold data variance -- flagged clearly, not stated by the paper.

VARIANTS:
  1. Full SleepMamba        -- baseline for this table (re-run under THIS
                                protocol, not reused from Table 3/6, since
                                those used full 10-fold CV, a different
                                protocol -- not a fair like-for-like baseline)
  2. w/o SDAM                -- MLE without channel/temporal attention
  3. w/o DAM-Intra           -- DAM using only the inter-modal (modality-axis) branch
  4. w/o DAM-Inter           -- DAM using only the intra-modal (temporal-axis) branch
  5. w/o SBM                 -- no inter-epoch modeling; DAM output goes straight to head
  6. DAM w/ independent gates -- two separate gates instead of one shared gate (Eq. 9)
  7. DAM replaced by Cross-Attention -- ASSUMPTION FLAGGED: paper gives zero
     implementation detail for this variant. We implement it as standard
     multi-head cross-attention where the temporal-axis view attends to
     the modality-axis view, replacing DAM's two SSM branches entirely
     while keeping everything else (MLE, SBM, head) identical. This choice
     is not verifiable against the paper and is our best-effort
     interpretation, consistent with how the paper's Related Work section
     frames cross-attention as an alternative to structured SSM fusion.
"""

import torch
import torch.nn as nn

from mle import MultimodalLocalEncoder
from mamba_backend import MambaBlock
from sbm import SBMStack


class DAMSingleBranch(nn.Module):
    """
    Shape-preserving (batch,D,E)->(batch,D,E), like DualAxisMambaBlock, but
    uses ONLY ONE of the two axis scans (for w/o DAM-Intra / w/o DAM-Inter).
    Same pre-norm + residual stabilization as the full DualAxisMambaBlock.
    """

    def __init__(self, D=128, E=20, axis="temporal", d_state=16, d_conv=4, expand=2):
        super().__init__()
        assert axis in ("temporal", "modal")
        self.axis = axis
        self.norm = nn.LayerNorm(D)
        self.gate_proj = nn.Conv1d(D, D, kernel_size=1)
        if axis == "temporal":
            self.ssm = MambaBlock(d_model=D, d_state=d_state, d_conv=d_conv, expand=expand)
        else:
            self.ssm = MambaBlock(d_model=E, d_state=d_state, d_conv=d_conv, expand=expand)

    def forward(self, F_in):
        F_normed = self.norm(F_in.transpose(1, 2)).transpose(1, 2)
        V = torch.sigmoid(self.gate_proj(F_normed))

        if self.axis == "temporal":
            x = F_normed.transpose(1, 2)      # (batch, E, D)
            z = self.ssm(x).transpose(1, 2)     # (batch, D, E)
        else:
            z = self.ssm(F_normed)                # (batch, D, E)

        z = z * V
        return F_in + z


class DAMIndependentGates(nn.Module):
    """
    Same as DualAxisMambaBlock, but with TWO separate gates (one per axis)
    instead of Eq. 9's single shared gate. Table 7's "DAM w/ independent
    gates" variant.
    """

    def __init__(self, D=128, E=20, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.norm = nn.LayerNorm(D)
        self.gate_proj_temp = nn.Conv1d(D, D, kernel_size=1)
        self.gate_proj_mod = nn.Conv1d(D, D, kernel_size=1)
        self.ssm_temp = MambaBlock(d_model=D, d_state=d_state, d_conv=d_conv, expand=expand)
        self.ssm_mod = MambaBlock(d_model=E, d_state=d_state, d_conv=d_conv, expand=expand)

    def forward(self, F_in):
        F_normed = self.norm(F_in.transpose(1, 2)).transpose(1, 2)

        V_temp = torch.sigmoid(self.gate_proj_temp(F_normed))
        V_mod = torch.sigmoid(self.gate_proj_mod(F_normed))

        x_temp = F_normed.transpose(1, 2)
        z_intra = self.ssm_temp(x_temp).transpose(1, 2) * V_temp

        z_inter = self.ssm_mod(F_normed) * V_mod

        Z = z_intra + z_inter
        return F_in + Z


class CrossAttentionFusionBlock(nn.Module):
    """
    Table 7's "DAM replaced by Cross-Attention" variant. ASSUMPTION
    FLAGGED (see module docstring): standard multi-head attention, temporal
    view (query) attends to modality view (key/value), replacing DAM's dual
    SSM scans. Shape-preserving (batch,D,E)->(batch,D,E) via residual.
    """

    def __init__(self, D=128, E=20, n_heads=4, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(D)
        # query: temporal view (batch, E, D); key/value: modality view (batch, D, D)
        # projecting modality axis (E-dim per position) to D first for compatible attention dims
        self.modal_proj = nn.Linear(E, D)
        self.cross_attn = nn.MultiheadAttention(embed_dim=D, num_heads=n_heads,
                                                  dropout=dropout, batch_first=True)
        self.out_proj = nn.Linear(D, D)

    def forward(self, F_in):
        # F_in: (batch, D, E)
        F_normed = self.norm(F_in.transpose(1, 2)).transpose(1, 2)  # (batch, D, E)

        query = F_normed.transpose(1, 2)          # (batch, E, D) -- temporal view
        kv_source = self.modal_proj(F_normed)        # (batch, D, D) -- modality view projected to D
        attn_out, _ = self.cross_attn(query, kv_source, kv_source)  # (batch, E, D)
        attn_out = self.out_proj(attn_out)
        attn_out = attn_out.transpose(1, 2)          # (batch, D, E)

        return F_in + attn_out


class AblationDAMStack(nn.Module):
    """Generic stack wrapper: chains `num_layers` of whichever block class is
    passed in, then applies the same Eq. 13 AvgPool at the end."""

    def __init__(self, block_factory, num_layers=2):
        super().__init__()
        self.layers = nn.ModuleList([block_factory() for _ in range(num_layers)])

    def forward(self, F):
        Z = F
        for layer in self.layers:
            Z = layer(Z)
        g = Z.mean(dim=2)
        return g


class SleepMambaAblation(nn.Module):
    """
    Flexible SleepMamba variant builder for Table 7.

    variant: one of
      'full', 'no_sdam', 'no_dam_intra', 'no_dam_inter',
      'no_sbm', 'dam_independent_gates', 'dam_cross_attention'
    """

    def __init__(self, variant="full", n_modalities=2, D=128, E=20, n_classes=5,
                 dam_layers=2, sbm_layers=2, d_state=16, d_conv=4, expand=2, dropout=0.5):
        super().__init__()
        assert variant in (
            "full", "no_sdam", "no_dam_intra", "no_dam_inter",
            "no_sbm", "dam_independent_gates", "dam_cross_attention",
        )
        self.variant = variant
        self.D = D

        use_sdam = (variant != "no_sdam")
        self.mle = MultimodalLocalEncoder(n_modalities=n_modalities, D=D, E=E,
                                           dropout=dropout, use_sdam=use_sdam)

        if variant == "no_dam_intra":
            # only the INTER-modal (modality-axis) branch remains
            factory = lambda: DAMSingleBranch(D=D, E=E, axis="modal",
                                               d_state=d_state, d_conv=d_conv, expand=expand)
            self.dam_stack = AblationDAMStack(factory, num_layers=dam_layers)
        elif variant == "no_dam_inter":
            # only the INTRA-modal (temporal-axis) branch remains
            factory = lambda: DAMSingleBranch(D=D, E=E, axis="temporal",
                                               d_state=d_state, d_conv=d_conv, expand=expand)
            self.dam_stack = AblationDAMStack(factory, num_layers=dam_layers)
        elif variant == "dam_independent_gates":
            factory = lambda: DAMIndependentGates(D=D, E=E, d_state=d_state, d_conv=d_conv, expand=expand)
            self.dam_stack = AblationDAMStack(factory, num_layers=dam_layers)
        elif variant == "dam_cross_attention":
            factory = lambda: CrossAttentionFusionBlock(D=D, E=E)
            self.dam_stack = AblationDAMStack(factory, num_layers=dam_layers)
        else:
            # 'full', 'no_sdam', 'no_sbm' all use the standard, already-verified DAMStack
            from dam import DAMStack
            self.dam_stack = DAMStack(D=D, E=E, num_layers=dam_layers,
                                       d_state=d_state, d_conv=d_conv, expand=expand)

        self.use_sbm = (variant != "no_sbm")
        if self.use_sbm:
            self.sbm_stack = SBMStack(D=D, num_layers=sbm_layers,
                                       d_state=d_state, d_conv=d_conv, expand=expand)

        self.head = nn.Linear(D, n_classes)

    def forward(self, X):
        batch, T, C, L = X.shape
        x_flat = X.reshape(batch * T, C, L)
        F_pp = self.mle(x_flat)
        g_flat = self.dam_stack(F_pp)
        G = g_flat.reshape(batch, T, self.D)

        if self.use_sbm:
            O = self.sbm_stack(G)
        else:
            O = G  # w/o SBM: skip inter-epoch modeling entirely

        logits = self.head(O)
        y_hat = torch.softmax(logits, dim=-1)
        return y_hat


def _quick_selfcheck():
    print("=== Table 7 ablation variants self-check ===\n")
    batch, T, C, L = 2, 5, 2, 3000
    dummy_X = torch.randn(batch, T, C, L)

    variants = ["full", "no_sdam", "no_dam_intra", "no_dam_inter",
                "no_sbm", "dam_independent_gates", "dam_cross_attention"]

    for v in variants:
        model = SleepMambaAblation(variant=v, n_modalities=2, D=128, E=20, n_classes=5)
        y_hat = model(dummy_X)
        n_params = sum(p.numel() for p in model.parameters())
        prob_sums = y_hat.sum(dim=-1)
        max_dev = (prob_sums - 1.0).abs().max().item()
        assert y_hat.shape == (batch, T, 5)
        assert max_dev < 1e-4
        print(f"{v:24s}  output={tuple(y_hat.shape)}  params={n_params:,}  "
              f"softmax_check={'OK' if max_dev < 1e-4 else 'FAIL'}")

    print("\nAll 7 variants (full + 6 ablations) pass shape/softmax checks.")


if __name__ == "__main__":
    _quick_selfcheck()