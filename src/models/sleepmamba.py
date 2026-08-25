"""
SleepMamba reproduction — Step 4d: Full model assembly (Algorithm 1)
=========================================================================
Wires together: MultimodalLocalEncoder -> DAMStack -> SBMStack -> head.

Input:  X  (batch, T, C, L)   T consecutive 30s epochs, C modality channels, L=3000
Output: Y_hat (batch, T, 5)   per-epoch softmax probabilities over sleep stages

Processing per Algorithm 1:
  1. For each epoch t in 1..T: MLE(X_t) -> F''_t  (all T epochs processed in
     one batched call for efficiency: reshape (batch,T,C,L) -> (batch*T,C,L))
  2. DAMStack(F''_t) -> g_t  (also batched: (batch*T,D,E) -> (batch*T,D))
  3. Reshape g back to sequence: (batch*T,D) -> (batch,T,D)
  4. SBMStack(G) -> O  (batch,T,D), models inter-epoch dependencies
  5. Per-epoch head: Softmax(Linear(o_t)) -> (batch,T,5)

======================================================================
NOTE ON OUTPUT FORMAT / LOSS FUNCTION (flagged for Step 5 training code):
======================================================================
  Algorithm 1 explicitly writes "y_hat_t <- Softmax(Linear(o_t))", so this
  module returns PROBABILITIES (post-softmax), matching the paper's
  Eq. 17 categorical cross-entropy loss which operates on y_hat directly:
      L = -(1/T) sum_t sum_k Y_{t,k} log(Y_hat_{t,k})
  This means the training loop (Step 5) must use nn.NLLLoss on
  log(y_hat), NOT nn.CrossEntropyLoss directly (which expects raw
  logits and applies its own internal log-softmax) -- using
  CrossEntropyLoss on already-softmaxed output would double-apply
  softmax and be mathematically wrong. This module intentionally keeps
  the softmax OUTSIDE so the returned tensor matches the paper's
  notation exactly; Step 5 will handle the loss correctly.
======================================================================
"""

import torch
import torch.nn as nn

from mle import MultimodalLocalEncoder
from dam import DAMStack
from sbm import SBMStack


class SleepMamba(nn.Module):
    def __init__(self, n_modalities=2, D=128, E=20, n_classes=5,
                 dam_layers=2, sbm_layers=2, d_state=16, d_conv=4, expand=2,
                 dropout=0.5):
        super().__init__()
        self.mle = MultimodalLocalEncoder(n_modalities=n_modalities, D=D, E=E, dropout=dropout)
        self.dam_stack = DAMStack(D=D, E=E, num_layers=dam_layers,
                                   d_state=d_state, d_conv=d_conv, expand=expand)
        self.sbm_stack = SBMStack(D=D, num_layers=sbm_layers,
                                   d_state=d_state, d_conv=d_conv, expand=expand)
        self.head = nn.Linear(D, n_classes)

    def forward(self, X):
        # X: (batch, T, C, L)
        batch, T, C, L = X.shape

        x_flat = X.reshape(batch * T, C, L)         # (batch*T, C, L)
        F_pp = self.mle(x_flat)                        # (batch*T, D, E)
        g_flat = self.dam_stack(F_pp)                    # (batch*T, D)

        D = g_flat.shape[-1]
        G = g_flat.reshape(batch, T, D)                  # (batch, T, D)

        O = self.sbm_stack(G)                             # (batch, T, D)

        logits = self.head(O)                              # (batch, T, n_classes)
        y_hat = torch.softmax(logits, dim=-1)               # (batch, T, n_classes), per Algorithm 1

        return y_hat


def _quick_selfcheck():
    print("=== Step 4d self-check: Full SleepMamba model ===\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running self-check on device: {device}\n")

    for T in (5, 15, 21, 30):
        model = SleepMamba(n_modalities=2, D=128, E=20, n_classes=5).to(device)
        batch = 2
        dummy_X = torch.randn(batch, T, 2, 3000, device=device)

        y_hat = model(dummy_X)
        n_params = sum(p.numel() for p in model.parameters())

        prob_sums = y_hat.sum(dim=-1)  # should be ~1.0 everywhere, since softmax
        max_dev = (prob_sums - 1.0).abs().max().item()

        print(f"T={T:2d}  input={tuple(dummy_X.shape)}  output={tuple(y_hat.shape)}  "
              f"params={n_params:,}  softmax_row_sum_max_dev={max_dev:.6f}")

        assert y_hat.shape == (batch, T, 5)
        assert max_dev < 1e-4, "Softmax rows don't sum to 1 -- something is wrong!"

    print(f"\nPaper's reported total (Table 5, T=5, EEG+EOG): 1,760,000 params (1.76M)")
    print(f"Our T=5 model: {n_params:,} params" if T == 5 else "")

    model5 = SleepMamba(n_modalities=2, D=128, E=20, n_classes=5).to(device)
    n_params5 = sum(p.numel() for p in model5.parameters())
    print(f"\nOur SleepMamba (T=5, EEG+EOG) total parameter count: {n_params5:,}")
    print(f"Paper's reported total:                                1,760,000")
    print(f"Difference: {n_params5 - 1_760_000:+,}  ({100*(n_params5-1_760_000)/1_760_000:+.1f}%)")

    print("\nSelf-check passed (shapes and softmax normalization both correct).")


if __name__ == "__main__":
    _quick_selfcheck()