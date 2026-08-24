"""
SleepMamba reproduction — Mamba backend auto-selector
=========================================================
Provides a single `MambaBlock(d_model, d_state, d_conv, expand)` factory
that dam.py and sbm.py import instead of importing mamba_ref directly.

Automatically selects:
  - Real mamba_ssm.Mamba (CUDA kernels) IF mamba-ssm is installed AND a
    CUDA GPU is available (i.e., we're on Kaggle with mamba-ssm installed).
  - Our reference pure-PyTorch implementation (mamba_ref.py) OTHERWISE
    (i.e., local CPU dev in VS Code).

This means dam.py/sbm.py/sleepmamba.py NEVER need manual editing when
moving between VS Code (CPU) and Kaggle (GPU) -- the correct backend is
picked automatically at import time, every time.

VERIFIED EQUIVALENT: the two backends were confirmed numerically
consistent (max abs diff ~4.5e-08, far under the 1e-2 tolerance) via
verify_against_mamba_ssm.py on a Kaggle T4 GPU, after fixing a dt_rank
convention mismatch in mamba_ref.py. Safe to treat them as interchangeable.

Both backends share the same forward signature: input (batch, L, d_model)
-> output (batch, L, d_model), so this factory function is a true drop-in
swap, no other code needs to change based on which backend is active.
"""

import torch

_BACKEND_NAME = None
_USE_REAL_MAMBA_SSM = False

try:
    from mamba_ssm import Mamba as _RealMamba
    if torch.cuda.is_available():
        _USE_REAL_MAMBA_SSM = True
        _BACKEND_NAME = "real mamba-ssm (CUDA kernels)"
    else:
        _BACKEND_NAME = "reference pure-PyTorch implementation (mamba-ssm installed but no CUDA GPU detected)"
except ImportError:
    _BACKEND_NAME = "reference pure-PyTorch implementation (mamba-ssm not installed)"

if _USE_REAL_MAMBA_SSM:
    def MambaBlock(d_model, d_state=16, d_conv=4, expand=2):
        return _RealMamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
else:
    from mamba_ref import MambaBlock as MambaBlock  # noqa: F401 (re-exported)

print(f"[mamba_backend] Using: {_BACKEND_NAME}")


def get_backend_name():
    return _BACKEND_NAME


def _quick_selfcheck():
    print("=== mamba_backend.py self-check ===\n")
    print(f"Active backend: {get_backend_name()}\n")

    batch, L, d_model = 2, 20, 128
    block = MambaBlock(d_model=d_model, d_state=16, d_conv=4, expand=2)
    if torch.cuda.is_available() and _USE_REAL_MAMBA_SSM:
        block = block.to("cuda")
        dummy_x = torch.randn(batch, L, d_model, device="cuda")
    else:
        dummy_x = torch.randn(batch, L, d_model)

    out = block(dummy_x)
    print(f"Input shape:  {tuple(dummy_x.shape)}")
    print(f"Output shape: {tuple(out.shape)}   (expected: ({batch}, {L}, {d_model}))")
    assert out.shape == (batch, L, d_model)

    n_params = sum(p.numel() for p in block.parameters())
    print(f"Parameter count: {n_params:,}")

    print("\nSelf-check passed.")


if __name__ == "__main__":
    _quick_selfcheck()