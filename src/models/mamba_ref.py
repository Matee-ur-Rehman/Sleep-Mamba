"""
SleepMamba reproduction — Step 4b (part 1): Selective SSM (S6) + Mamba block
================================================================================
Pure-PyTorch reference implementation of the selective state-space
mechanism, following the paper's own Eq. 3-5 (continuous->discretized
SSM) and Eq. 6-8 (Mamba block with gating), which in turn follow
Gu & Dao's Mamba (ref [20]).

WHY THIS EXISTS INSTEAD OF THE OFFICIAL mamba-ssm PACKAGE:
  mamba-ssm's fast path requires CUDA kernels (selective_scan_cuda,
  causal-conv1d) that need an NVIDIA GPU + CUDA toolchain to build —
  unavailable on this CPU-only local dev machine. This module is used
  ONLY for local architecture/shape/logic debugging in VS Code.

  On Kaggle (GPU-available), we MUST:
    1. pip install mamba-ssm causal-conv1d
    2. Run the numerical equivalence check in verify_against_mamba_ssm.py
       (written once we're on Kaggle) to confirm this module's output
       matches the real CUDA-backed mamba-ssm to floating-point tolerance
    3. Use the REAL mamba-ssm for all actual training runs whose results
       get compared against the paper's Table 3/4 numbers.

  This module is a bridge for local development speed, not the source
  of truth for final reproduction results.

======================================================================
ASSUMPTIONS FLAGGED (Mamba internal hyperparameters, unspecified in paper):
======================================================================
  - d_state (SSM state dimension N)      = 16   (official Mamba-1 default;
                                                   barely affects param count,
                                                   left at default)
  - d_conv  (causal conv kernel size)    = 4    (official Mamba-1 default)
  - expand  (inner-dimension expansion)  = 1    ***CHANGED FROM DEFAULT***

  REASONING FOR expand VALUE:
  We initially considered expand=1 to stay under the paper's reported
  1.76M total parameter count (Table 5, T=5). However, once the FULL
  model was assembled (MLE + DAMStack + SBMStack + head), expand=1
  undershoots the paper's total by ~53%, while even the Mamba-1
  standard default of expand=2 still undershoots by ~31.5% -- no
  clean/standard expand value (1, 2, or 4) lands close to 1.76M, and
  the nearest numeric match (expand=3) is architecturally non-standard
  and would only be curve-fit to one aggregate statistic rather than
  a principled choice.

  DECISION: use expand=2, the Mamba-1 standard default, since the
  paper explicitly builds on Mamba (ref [20]) without stating a custom
  expansion factor, and standard-practice defaults are more likely to
  produce functionally representative model behavior than a value
  chosen solely to minimize a parameter-count gap. The resulting
  ~31.5% parameter undercount vs. the paper is an ACKNOWLEDGED,
  UNRESOLVED gap -- likely coming from some combination of other
  unspecified dimensions (E, SDAM reduction ratio, possible extra
  normalization layers, or a different head structure) that cannot be
  disentangled from a single aggregate number without the authors'
  source code. The real test of reproduction fidelity is downstream
  Accuracy/kappa/MF1 after training (compared to paper's Table 5),
  not parameter-count parity.
  ======================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SelectiveSSM(nn.Module):
    """
    Reference (non-CUDA) selective scan, Eq. 3-5 of the paper:
        h'(t) = A h(t) + B x(t),   y(t) = C h(t)
    discretized under zero-order hold with input-dependent Delta, B, C
    (the "selective"/S6 part — A,B,C,Delta depend on the input x).

    Input:  u  (batch, d_inner, L)   -- already through the causal conv + SiLU
    Output: y  (batch, d_inner, L)
    """

    def __init__(self, d_inner, d_state=16, dt_rank=None):
        super().__init__()
        self.d_inner = d_inner
        self.d_state = d_state
        self.dt_rank = dt_rank if dt_rank is not None else max(d_inner // 16, 1)

        # Input-dependent projections: from u -> (Delta, B, C)
        self.x_proj = nn.Linear(d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, d_inner, bias=True)

        # A is a learned, input-INDEPENDENT parameter (per Mamba design):
        # stored in log-space, one real row of length d_state per channel,
        # negative to keep the continuous-time system stable.
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(d_inner, 1)  # (d_inner, d_state)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(d_inner))  # skip connection (Eq. 8's residual-like term)

    def forward(self, u):
        # u: (batch, d_inner, L)
        batch, d_inner, L = u.shape
        u_t = u.transpose(1, 2)  # (batch, L, d_inner)

        x_dbl = self.x_proj(u_t)  # (batch, L, dt_rank + 2*d_state)
        delta, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        delta = F.softplus(self.dt_proj(delta))  # (batch, L, d_inner), positive step size

        A = -torch.exp(self.A_log)  # (d_inner, d_state), negative for stability

        # Discretize: Abar = exp(delta * A), Bbar = delta * B  (ZOH approx for B, Eq. 4)
        # delta: (batch, L, d_inner) -> (batch, L, d_inner, 1)
        # A:     (d_inner, d_state)  -> (1, 1, d_inner, d_state)
        deltaA = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))  # (batch, L, d_inner, d_state)
        deltaB_u = delta.unsqueeze(-1) * B.unsqueeze(2) * u_t.unsqueeze(-1)     # (batch, L, d_inner, d_state)

        # Sequential scan over time (reference implementation — slow but exact;
        # the CUDA kernel computes this same recurrence in parallel/fused form)
        h = torch.zeros(batch, d_inner, self.d_state, device=u.device, dtype=u.dtype)
        ys = []
        for t in range(L):
            h = deltaA[:, t] * h + deltaB_u[:, t]           # (batch, d_inner, d_state)
            y_t = torch.einsum("bdn,bn->bd", h, C[:, t])     # (batch, d_inner)
            ys.append(y_t)
        y = torch.stack(ys, dim=1)  # (batch, L, d_inner)

        y = y + u_t * self.D  # skip/residual term
        return y.transpose(1, 2)  # (batch, d_inner, L)


class MambaBlock(nn.Module):
    """
    Full Mamba block, Eq. 6-8:
        U = SiLU(Conv1d(Linear_u(x)))
        V = SiLU(Linear_v(x))
        Mamba(x) = Linear_o(SSM(U) . V)
    """

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_inner = expand * d_model

        self.in_proj_u = nn.Linear(d_model, self.d_inner, bias=False)
        self.in_proj_v = nn.Linear(d_model, self.d_inner, bias=False)

        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner, kernel_size=d_conv,
            groups=self.d_inner, padding=d_conv - 1, bias=True,
        )
        self.ssm = SelectiveSSM(self.d_inner, d_state=d_state)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x):
        # x: (batch, L, d_model)  -- sequence-last-dim convention, like nn.Transformer
        L = x.shape[1]

        u = self.in_proj_u(x).transpose(1, 2)   # (batch, d_inner, L)
        u = self.conv1d(u)[:, :, :L]              # causal conv, trim padding tail
        u = F.silu(u)

        v = F.silu(self.in_proj_v(x))            # (batch, L, d_inner)

        y = self.ssm(u)                            # (batch, d_inner, L)
        y = y.transpose(1, 2)                       # (batch, L, d_inner)

        out = self.out_proj(y * v)                # (batch, L, d_model)
        return out


def _quick_selfcheck():
    print("=== Step 4b (part 1) self-check: SelectiveSSM + MambaBlock ===")
    batch, L, d_model = 2, 20, 128
    block = MambaBlock(d_model=d_model, d_state=16, d_conv=4, expand=2)

    dummy_x = torch.randn(batch, L, d_model)
    out = block(dummy_x)
    print(f"Input shape:  {tuple(dummy_x.shape)}")
    print(f"Output shape: {tuple(out.shape)}   (expected: ({batch}, {L}, {d_model}))")

    n_params = sum(p.numel() for p in block.parameters())
    print(f"Single MambaBlock parameter count: {n_params:,}")

    assert out.shape == (batch, L, d_model), "Shape mismatch!"
    print("\nNOTE: this is the reference PyTorch implementation for CPU debugging.")
    print("Real training MUST use mamba-ssm's CUDA kernels on Kaggle, verified")
    print("against this implementation first.")
    print("\nSelf-check passed.")


if __name__ == "__main__":
    _quick_selfcheck()