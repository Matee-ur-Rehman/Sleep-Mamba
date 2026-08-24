"""
SleepMamba reproduction — Step 4c: Sequence Bi-Mamba (SBM) layer
=====================================================================
Implements Section 3.5 / Eq. 14-16 / Fig. 1(c).

Unlike DAM, this operates ACROSS epochs, not within one. Input is the
stacked sequence of epoch descriptors from DAM:
    G = [g_1, ..., g_T] in R^(T x D)
Output is the bidirectionally-contextualized sequence:
    O = [o_1, ..., o_T] in R^(T x D)

This module is naturally shape-preserving ((batch,T,D) -> (batch,T,D)),
so "stacked twice" (Section 4.2) is unambiguous here, unlike DAM.

======================================================================
ASSUMPTIONS FLAGGED (same reasoning as mamba_ref.py / dam.py):
======================================================================
  - Internal MambaBlock hyperparameters (d_state=16, d_conv=4, expand=2)
    reused from mamba_ref.py for consistency, with the same expand=2
    justification (paper's 1.76M total parameter budget).
======================================================================
"""

import torch
import torch.nn as nn

from mamba_ref import MambaBlock


class SequenceBiMambaLayer(nn.Module):
    """Single SBM layer: forward Mamba + backward Mamba (on flipped sequence,
    flipped back), summed. Eq. 14-16. Shape-preserving: (batch,T,D)->(batch,T,D)."""

    def __init__(self, D=128, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.mamba_fwd = MambaBlock(d_model=D, d_state=d_state, d_conv=d_conv, expand=expand)
        self.mamba_bwd = MambaBlock(d_model=D, d_state=d_state, d_conv=d_conv, expand=expand)

    def forward(self, G):
        # G: (batch, T, D)
        O_fwd = self.mamba_fwd(G)  # Eq. 14

        G_flipped = torch.flip(G, dims=[1])
        O_bwd_flipped = self.mamba_bwd(G_flipped)
        O_bwd = torch.flip(O_bwd_flipped, dims=[1])  # Eq. 15: flip back to original order

        O = O_fwd + O_bwd  # Eq. 16
        return O


class SBMStack(nn.Module):
    """Chains `num_layers` SequenceBiMambaLayer instances. Section 4.2: stacked twice."""

    def __init__(self, D=128, num_layers=2, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.layers = nn.ModuleList([
            SequenceBiMambaLayer(D=D, d_state=d_state, d_conv=d_conv, expand=expand)
            for _ in range(num_layers)
        ])

    def forward(self, G):
        O = G
        for layer in self.layers:
            O = layer(O)
        return O  # (batch, T, D)


def _quick_selfcheck():
    print("=== Step 4c self-check: SBM (single layer + stacked) ===")
    batch, T, D = 4, 5, 128

    single = SequenceBiMambaLayer(D=D)
    dummy_G = torch.randn(batch, T, D)
    o = single(dummy_G)
    print(f"Single SBM layer: input {tuple(dummy_G.shape)} -> output {tuple(o.shape)}"
          f"   (expected: ({batch}, {T}, {D}), shape-preserving)")
    assert o.shape == (batch, T, D)

    n_params_single = sum(p.numel() for p in single.parameters())
    print(f"Single SBM layer parameter count: {n_params_single:,}")

    stack = SBMStack(D=D, num_layers=2)
    o_stack = stack(dummy_G)
    print(f"\nSBMStack (2 layers): input {tuple(dummy_G.shape)} -> output {tuple(o_stack.shape)}"
          f"   (expected: ({batch}, {T}, {D}))")
    assert o_stack.shape == (batch, T, D)

    n_params_stack = sum(p.numel() for p in stack.parameters())
    print(f"SBMStack (2 layers) parameter count: {n_params_stack:,}")

    # sanity: different T values should all work, since Mamba has no fixed-length assumption
    for T_test in (5, 15, 21, 30):
        dummy_G_t = torch.randn(2, T_test, D)
        o_t = stack(dummy_G_t)
        assert o_t.shape == (2, T_test, D)
    print("\nConfirmed SBMStack works for all paper-specified T values: {5, 15, 21, 30}")

    print("\nSelf-check passed.")


if __name__ == "__main__":
    _quick_selfcheck()