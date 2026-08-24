"""
SleepMamba reproduction — Step 4a: Multimodal Local Encoder (MLE)
=====================================================================
Implements Section 3.2 / Figure 2 of the paper: multi-scale CNN feature
extraction (small-scale + large-scale branches) followed by the Sparse
Dual Attention Module (SDAM: channel attention + temporal attention).

Input:  X_i  (batch, C, L)   C = n_modalities (e.g. 2 for EEG+EOG), L=3000
Output: F''  (batch, D, E)   D=128 latent channels, E = reduced temporal length

======================================================================
ASSUMPTIONS FLAGGED (paper does not specify these numerically):
======================================================================
1. Small-scale branch first conv: kernel=50, stride=6.
   Large-scale branch first conv: kernel=400, stride=50 (this one IS
   given literally in Fig. 2's "Conv1d(64,400,50)" label).
   The small-scale value is inferred from the DeepSleepNet-style
   dual-branch design this paper's Section 3.2.1 prose describes
   (large kernel = low-freq/long-duration, small kernel = high-freq
   transients), since Fig. 2's text extraction showed identical labels
   on both branches (almost certainly a figure/OCR artifact, since
   using literally the same kernel defeats the purpose of "multi-scale").

2. CRITICAL FIX — branch length mismatch: with the above kernel sizes,
   the small branch reduces a 3000-sample epoch to 116 time steps and
   the large branch to 6 time steps (verified by direct computation).
   Fig. 1(a)/Fig. 2 show the two branches being concatenated after
   SDAM, which requires matching temporal length. The paper does not
   address this mismatch anywhere in the text. We resolve it by adding
   an AdaptiveAvgPool1d(E) at the end of each branch to force both to
   a common length E before concatenation. This adaptive pooling layer
   is OUR addition, not stated in the paper.

3. E (target common temporal length after adaptive pooling) = 20.
   Not given anywhere in the paper. Chosen as a round, reasonable value.

4. D (latent feature channels) = 128, taken directly from Fig. 2's
   `Conv1d(128,7,1)` layers, which both branches share — this one is
   NOT an assumption, it's read directly off the figure.

5. SDAM channel-attention MLP reduction ratio: not specified. Using a
   standard reduction ratio of 16 (SE-block convention), with a floor
   of 4 to avoid degenerate bottlenecks on small channel counts.

6. DenseLayer (Fig. 2, after concatenation) activation: the paper shows
   "Concatenation -> DenseLayer -> Dropout(0.5)" with no activation
   drawn. We add a GELU after the DenseLayer for consistency with the
   GELU used everywhere else in the MLE (assumption, reasonable but
   not stated).
======================================================================
"""

import torch
import torch.nn as nn


class SparseDualAttentionModule(nn.Module):
    """SDAM: channel-wise attention gate + temporal attention gate. Section 3.2.2."""

    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.channel_mlp = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.GELU(),
            nn.Linear(hidden, channels),
        )
        self.temporal_conv = nn.Conv1d(1, 1, kernel_size=1)

    def forward(self, F):
        # F: (batch, D, E)
        gap = F.mean(dim=2)              # (batch, D)
        gmp = F.amax(dim=2)              # (batch, D)
        Mc = torch.sigmoid(self.channel_mlp(gap) + self.channel_mlp(gmp))  # (batch, D)
        F_prime = F * Mc.unsqueeze(-1)   # broadcast over E

        chan_avg = F_prime.mean(dim=1, keepdim=True)   # (batch, 1, E)
        Mt = torch.sigmoid(self.temporal_conv(chan_avg))  # (batch, 1, E)
        F_double_prime = F_prime * Mt    # broadcast over D

        return F_double_prime


class MSCNNBranch(nn.Module):
    """One scale branch of the Multi-Scale CNN, Section 3.2.1 / Fig. 2(a)."""

    def __init__(self, in_channels, first_kernel, first_stride, target_E, dropout=0.5):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=first_kernel, stride=first_stride)
        self.bn1 = nn.BatchNorm1d(64)
        self.act1 = nn.GELU()
        self.pool1 = nn.MaxPool1d(kernel_size=4, stride=2)
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(64, 128, kernel_size=7, stride=1)
        self.act2 = nn.GELU()
        self.conv3 = nn.Conv1d(128, 128, kernel_size=7, stride=1)
        self.act3 = nn.GELU()
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        # NOT in the paper's figure — our fix for the branch-length mismatch (see module docstring)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(target_E)

    def forward(self, x):
        # x: (batch, C, L)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.pool1(x)
        x = self.drop1(x)
        x = self.conv2(x)
        x = self.act2(x)
        x = self.conv3(x)
        x = self.act3(x)
        x = self.pool2(x)
        x = self.adaptive_pool(x)
        return x  # (batch, 128, target_E)


class MultimodalLocalEncoder(nn.Module):
    """
    Full MLE: two MSCNN branches (small/large scale), each refined by its
    own SDAM, concatenated, then projected back to D=128 via DenseLayer.
    """

    def __init__(self, n_modalities=2, D=128, E=20, dropout=0.5):
        super().__init__()
        self.E = E
        self.small_branch = MSCNNBranch(
            in_channels=n_modalities, first_kernel=50, first_stride=6,
            target_E=E, dropout=dropout,
        )
        self.large_branch = MSCNNBranch(
            in_channels=n_modalities, first_kernel=400, first_stride=50,
            target_E=E, dropout=dropout,
        )
        self.sdam_small = SparseDualAttentionModule(channels=128)
        self.sdam_large = SparseDualAttentionModule(channels=128)

        self.dense = nn.Conv1d(256, D, kernel_size=1)  # "DenseLayer" applied per time step
        self.dense_act = nn.GELU()  # not explicitly drawn in Fig. 2 — our addition
        self.dense_dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch, C, L)  e.g. (batch, 2, 3000) for EEG+EOG
        f_small = self.small_branch(x)          # (batch, 128, E)
        f_large = self.large_branch(x)          # (batch, 128, E)

        f_small = self.sdam_small(f_small)       # (batch, 128, E)
        f_large = self.sdam_large(f_large)       # (batch, 128, E)

        f_cat = torch.cat([f_small, f_large], dim=1)  # (batch, 256, E)
        f = self.dense(f_cat)                    # (batch, D, E)
        f = self.dense_act(f)
        f = self.dense_dropout(f)
        return f  # (batch, D, E) == F'' in the paper's notation


def _quick_selfcheck():
    print("=== Step 4a self-check: Multimodal Local Encoder ===")
    batch = 4
    n_modalities = 2  # EEG + EOG
    L = 3000
    mle = MultimodalLocalEncoder(n_modalities=n_modalities, D=128, E=20)

    dummy_x = torch.randn(batch, n_modalities, L)
    out = mle(dummy_x)
    print(f"Input shape:  {tuple(dummy_x.shape)}")
    print(f"Output shape: {tuple(out.shape)}   (expected: ({batch}, 128, 20))")

    n_params = sum(p.numel() for p in mle.parameters())
    print(f"MLE parameter count: {n_params:,}")

    assert out.shape == (batch, 128, 20), "Shape mismatch!"
    print("\nSelf-check passed.")


if __name__ == "__main__":
    _quick_selfcheck()