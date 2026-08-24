"""
SleepMamba reproduction — Kaggle Step: mamba-ssm equivalence check
========================================================================
MUST be run on a CUDA GPU (Kaggle). Confirms our pure-PyTorch reference
MambaBlock (src/models/mamba_ref.py, used for CPU dev in VS Code)
produces numerically consistent output with the REAL mamba-ssm CUDA
implementation, before we trust it for the real training runs whose
results get compared against the paper's Table 3/4/5.

This is not a formality -- it's the actual verification step that lets
us swap from "our hand-written S6 scan" to "the library everyone else
uses" with confidence that we haven't silently diverged from the real
Mamba mechanism the paper builds on (ref [20]).

USAGE (run this as an early cell in the Kaggle notebook, after
`!pip install mamba-ssm causal-conv1d` and after cloning the repo):

    !python Sleep-Mamba/src/models/verify_against_mamba_ssm.py

WHAT IT DOES:
  1. Builds our reference MambaBlock (mamba_ref.py) and a matching real
     mamba-ssm Mamba block with IDENTICAL hyperparameters (d_model,
     d_state, d_conv, expand).
  2. Copies weights from one into the other, so both start from the
     EXACT SAME parameters (otherwise comparing outputs would be
     meaningless -- different random init = different output regardless
     of implementation correctness).
  3. Runs the same random input through both.
  4. Reports max absolute difference and max relative difference.

PASS CRITERION: max absolute difference should be small (~1e-3 to 1e-4
range is expected and fine -- differences at this scale come from
floating-point operation ORDER differences between the sequential
Python scan and the fused CUDA kernel, not from a logic error. A large
difference, e.g. >0.1, or output shapes not matching, indicates a real
bug in our reference implementation and must be investigated before
proceeding to real training.
"""

import sys
import torch
import torch.nn as nn

sys.path.append("Sleep-Mamba/src/models")  # adjust if repo cloned elsewhere
from mamba_ref import MambaBlock as ReferenceMambaBlock


def copy_weights_reference_to_mambassm(ref_block, real_block):
    """
    Copies parameters from our reference MambaBlock into a real
    mamba_ssm.Mamba block. Requires knowing both modules' internal
    parameter names/shapes -- these are matched by hand below based on
    mamba-ssm's public source layout (in_proj, conv1d, x_proj, dt_proj,
    A_log, D, out_proj). If mamba-ssm's internal structure has changed
    versions since this was written, this mapping may need updating --
    check with `print(real_block)` and `print(dict(real_block.named_parameters()).keys())`
    first if this fails.
    """
    with torch.no_grad():
        # mamba-ssm's in_proj is a SINGLE Linear producing both U and V
        # concatenated (unlike our reference's two separate Linears).
        # Concatenate our in_proj_u and in_proj_v weights to match.
        combined_in_proj_weight = torch.cat(
            [ref_block.in_proj_u.weight, ref_block.in_proj_v.weight], dim=0
        )
        real_block.in_proj.weight.copy_(combined_in_proj_weight)

        real_block.conv1d.weight.copy_(ref_block.conv1d.weight)
        real_block.conv1d.bias.copy_(ref_block.conv1d.bias)

        real_block.x_proj.weight.copy_(ref_block.ssm.x_proj.weight)
        real_block.dt_proj.weight.copy_(ref_block.ssm.dt_proj.weight)
        real_block.dt_proj.bias.copy_(ref_block.ssm.dt_proj.bias)

        real_block.A_log.copy_(ref_block.ssm.A_log)
        real_block.D.copy_(ref_block.ssm.D)

        real_block.out_proj.weight.copy_(ref_block.out_proj.weight)


def run_equivalence_check():
    if not torch.cuda.is_available():
        print("ERROR: this check requires a CUDA GPU. Are you running this on Kaggle "
              "with a GPU accelerator selected?")
        sys.exit(1)

    device = torch.device("cuda")
    print(f"Running on: {torch.cuda.get_device_name(0)}\n")

    try:
        from mamba_ssm import Mamba as RealMamba
    except ImportError:
        print("ERROR: mamba-ssm not installed. Run:")
        print("  !pip install mamba-ssm causal-conv1d")
        sys.exit(1)

    d_model, d_state, d_conv, expand = 128, 16, 4, 2  # our paper-derived defaults
    batch, L = 4, 20

    torch.manual_seed(0)
    ref_block = ReferenceMambaBlock(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand).to(device)
    real_block = RealMamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand).to(device)

    print("Copying weights from reference implementation into real mamba-ssm block...")
    try:
        copy_weights_reference_to_mambassm(ref_block, real_block)
    except Exception as e:
        print(f"WEIGHT COPY FAILED: {e}")
        print("This likely means mamba-ssm's internal module structure differs from")
        print("what this script expects. Run `print(real_block)` and inspect named")
        print("parameters to update the mapping in copy_weights_reference_to_mambassm().")
        sys.exit(1)

    dummy_x = torch.randn(batch, L, d_model, device=device)

    ref_block.eval()
    real_block.eval()
    with torch.no_grad():
        out_ref = ref_block(dummy_x)
        out_real = real_block(dummy_x)

    print(f"\nReference output shape: {tuple(out_ref.shape)}")
    print(f"Real mamba-ssm output shape: {tuple(out_real.shape)}")

    if out_ref.shape != out_real.shape:
        print("\nFAIL: output shapes do not match. Cannot compare further.")
        sys.exit(1)

    abs_diff = (out_ref - out_real).abs()
    max_abs_diff = abs_diff.max().item()
    mean_abs_diff = abs_diff.mean().item()
    rel_diff = (abs_diff / (out_real.abs() + 1e-8)).max().item()

    print(f"\nMax absolute difference:  {max_abs_diff:.6e}")
    print(f"Mean absolute difference: {mean_abs_diff:.6e}")
    print(f"Max relative difference:  {rel_diff:.6e}")

    if max_abs_diff < 1e-2:
        print("\nPASS: reference implementation is numerically consistent with real "
              "mamba-ssm CUDA kernels (differences within expected floating-point/"
              "operation-order tolerance).")
        print("Safe to proceed with real mamba-ssm for training.")
    else:
        print("\nFAIL: difference exceeds tolerance. Do NOT proceed to training until "
              "this is investigated -- likely a bug in the reference implementation's "
              "math (check mamba_ref.py's SelectiveSSM against Eq. 3-8 again) or a "
              "weight-copy mapping error above.")
        sys.exit(1)


if __name__ == "__main__":
    run_equivalence_check()