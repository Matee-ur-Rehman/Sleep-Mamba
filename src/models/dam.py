"""
SleepMamba reproduction — Step 4b (part 2): Dual-Axis Mamba (DAM) block
===========================================================================
Implements Section 3.4 / Eq. 9-13 / Fig. 1(b).

Input:  F'' (batch, D, E)   -- output of the MLE
Output: g_i (batch, D)      -- compact epoch-level descriptor (Eq. 13)

======================================================================
ASSUMPTIONS / STRUCTURAL DECISIONS FLAGGED:
======================================================================
1. STACKING STRUCTURE (important): Section 4.2 states "both the DAM and
   the SBM are stacked twice." But DAM's own Eq. 13 ends in an average
   pool that collapses (D,E) -> D, leaving no temporal axis for a
   second DAM layer to operate on. The only structurally consistent
   reading: the internal DualAxisMambaBlock stays shape-preserving
   (D,E) -> (D,E), two of these are chained, and the AvgPool -> g_i
   step happens ONCE, only after the last stacked layer. This is our
   resolution of an internal inconsistency in the paper's stated
   stacking depth vs. its own equations, not a literal instruction
   from the text.

2. Tensor axis convention: our MLE outputs F'' as (batch, D, E) in
   PyTorch's channels-first conv1d convention (D=128 channels, E=20
   time steps). To implement Eq. 10-11:
     - SSM_temp (intra-modal, "scans the temporal axis E"): we
       transpose to (batch, E, D) so the MambaBlock treats E as the
       sequence length and D as the per-step feature dim.
     - SSM_mod (inter-modal, "treats D as a latent modality sequence"):
       used directly as (batch, D, E) -- D is already dim 1 in our
       native layout, so no transpose needed here; MambaBlock treats D
       as sequence length and E as the per-step feature dim.
   Both MambaBlocks therefore have DIFFERENT d_model values
   (D=128 for the temporal one, E=20 for the modality one) — this
   follows directly from the paper's own axis definitions, not an
   extra assumption.

3. Shared gate V_i (Eq. 9): "V_i = sigma(Linear(F''_i))", V_i has the
   SAME (D,E) shape as F''_i. A standard nn.Linear acts on the last
   tensor dimension; to project along the CHANNEL (D) axis while
   preserving the (D,E) shape, we implement this "Linear" as a
   1x1 Conv1d(D,D) -- the standard way to apply a per-position linear
   map to channels-first conv-shaped tensors. Not stated explicitly
   in the paper, but this is the natural PyTorch equivalent, not a
   free design choice with real alternatives.
======================================================================
"""

import torch
import torch.nn as nn

from mamba_ref import MambaBlock


class DualAxisMambaBlock(nn.Module):
    """
    Single DAM layer. Shape-preserving: (batch, D, E) -> (batch, D, E).
    Implements Eq. 9-12 (everything except the final AvgPool of Eq. 13,
    which lives in DAMStack below so multiple DAM layers can be chained).
    """

    def __init__(self, D=128, E=20, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.D = D
        self.E = E

        self.gate_proj = nn.Conv1d(D, D, kernel_size=1)  # Eq. 9's "Linear", channels-first form

        # Eq. 10: SSM_temp scans the temporal axis E, feature dim = D
        self.ssm_temp = MambaBlock(d_model=D, d_state=d_state, d_conv=d_conv, expand=expand)
        # Eq. 11: SSM_mod scans the modality/channel axis D, feature dim = E
        self.ssm_mod = MambaBlock(d_model=E, d_state=d_state, d_conv=d_conv, expand=expand)

    def forward(self, F):
        # F: (batch, D, E)
        V = torch.sigmoid(self.gate_proj(F))  # (batch, D, E), Eq. 9

        # --- intra-modal temporal path, Eq. 10 ---
        x_temp = F.transpose(1, 2)              # (batch, E, D)
        z_intra = self.ssm_temp(x_temp)          # (batch, E, D)
        z_intra = z_intra.transpose(1, 2)        # (batch, D, E)
        z_intra = z_intra * V

        # --- inter-modal structural path, Eq. 11 ---
        z_inter = self.ssm_mod(F)                 # (batch, D, E) treated as (batch, L=D, d_model=E)
        z_inter = z_inter * V

        # --- fuse, Eq. 12 ---
        Z = z_intra + z_inter                     # (batch, D, E)
        return Z


class DAMStack(nn.Module):
    """
    Chains `num_layers` DualAxisMambaBlock layers (shape-preserving),
    then applies the modality-wise average pool of Eq. 13 exactly once,
    at the end, to produce the compact epoch descriptor g_i.
    """

    def __init__(self, D=128, E=20, num_layers=2, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.layers = nn.ModuleList([
            DualAxisMambaBlock(D=D, E=E, d_state=d_state, d_conv=d_conv, expand=expand)
            for _ in range(num_layers)
        ])

    def forward(self, F):
        # F: (batch, D, E)  -- MLE output
        Z = F
        for layer in self.layers:
            Z = layer(Z)
        g = Z.mean(dim=2)  # Eq. 13: AvgPool over E -> (batch, D)
        return g


def _quick_selfcheck():
    print("=== Step 4b (part 2) self-check: DAM (single layer + stacked) ===")
    batch, D, E = 4, 128, 20

    single = DualAxisMambaBlock(D=D, E=E)
    dummy_F = torch.randn(batch, D, E)
    z = single(dummy_F)
    print(f"Single DAM layer: input {tuple(dummy_F.shape)} -> output {tuple(z.shape)}"
          f"   (expected: ({batch}, {D}, {E}), shape-preserving)")
    assert z.shape == (batch, D, E)

    n_params_single = sum(p.numel() for p in single.parameters())
    print(f"Single DAM layer parameter count: {n_params_single:,}")

    stack = DAMStack(D=D, E=E, num_layers=2)
    g = stack(dummy_F)
    print(f"\nDAMStack (2 layers): input {tuple(dummy_F.shape)} -> output {tuple(g.shape)}"
          f"   (expected: ({batch}, {D}))")
    assert g.shape == (batch, D)

    n_params_stack = sum(p.numel() for p in stack.parameters())
    print(f"DAMStack (2 layers) parameter count: {n_params_stack:,}")

    print("\nSelf-check passed.")


if __name__ == "__main__":
    _quick_selfcheck()