"""
SleepMamba reproduction — Table 5: Baseline sequence models
================================================================================
Section 5.2: "we compare it with representative sequence modeling
baselines, including Transformer, Single-layer LSTM, and Bidirectional
LSTM (Bi-LSTM)... all models use the same EEG and EOG inputs with a fixed
temporal context of T=5... training protocol, optimizer, batch size,
early stopping strategy, and network depth are kept consistent."

================================================================================
KEY DESIGN DECISION (flagged, not explicitly stated by the paper):
================================================================================
"Representative sequence modeling baselines" is read here as isolating
JUST the sequence-modeling backbone (the role DAM+SBM plays), not
reinventing entire alternative architectures. So:
  - The MLE (multi-scale CNN + SDAM) is REUSED UNCHANGED from SleepMamba
    for all three baselines -- same feature extraction for everyone.
  - DAM's dual-axis fusion is SleepMamba's own contribution and is NOT
    given to baselines; instead, each epoch's MLE output is reduced to a
    simple per-epoch descriptor via average pooling over E (analogous to
    DAM's own final AvgPool step, Eq. 13, but without the dual-axis
    modeling that precedes it).
  - Only the INTER-EPOCH sequence model differs: Transformer encoder,
    single-layer LSTM, or Bi-LSTM, in place of SBM.

PROTOCOL: full 10-fold CV, matching Table 3's rigor -- justified because
Table 5's own reported SleepMamba row (83.4/0.768/77.2) is IDENTICAL to
Table 3's 10-fold SleepMamba-5 result, indicating the baselines were
evaluated under the same 10-fold protocol for a fair comparison, not a
lighter single-split protocol.

HYPERPARAMETERS NOT SPECIFIED BY THE PAPER (flagged):
  - Depth = 2 layers for all three baselines, matching SBM's "stacked
    twice" as our best available reference point for "network depth
    kept consistent."
  - Hidden width = 128 (matching D) for LSTM/Bi-LSTM; Transformer
    d_model=128, nhead=4, dim_feedforward=256. The paper states hidden
    width was "tuned according to best validation MF1" -- we do not have
    time/compute budget for exhaustive tuning, so these are reasonable,
    clearly-flagged defaults rather than paper-verified values.
"""

import math
import torch
import torch.nn as nn

from mle import MultimodalLocalEncoder


class EpochEncoder(nn.Module):
    """Shared frontend for all Table 5 baselines: MLE + average pool over E,
    producing one D-dim descriptor per epoch. Reused unchanged for
    Transformer/LSTM/Bi-LSTM -- only the downstream sequence model differs."""

    def __init__(self, n_modalities=2, D=128, E=20, dropout=0.5):
        super().__init__()
        self.mle = MultimodalLocalEncoder(n_modalities=n_modalities, D=D, E=E, dropout=dropout)

    def forward(self, X):
        # X: (batch, T, C, L)
        batch, T, C, L = X.shape
        x_flat = X.reshape(batch * T, C, L)
        F_pp = self.mle(x_flat)          # (batch*T, D, E)
        g_flat = F_pp.mean(dim=2)          # (batch*T, D) -- simple avg pool over E
        D = g_flat.shape[-1]
        G = g_flat.reshape(batch, T, D)      # (batch, T, D)
        return G


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding, needed since Transformers
    have no inherent sequence order awareness (unlike LSTM/Mamba)."""

    def __init__(self, d_model, max_len=64):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        # x: (batch, T, d_model)
        return x + self.pe[:, :x.size(1)]


class TransformerBaseline(nn.Module):
    def __init__(self, n_modalities=2, D=128, E=20, n_classes=5,
                 nhead=4, num_layers=2, dim_feedforward=256, dropout=0.1):
        super().__init__()
        self.encoder = EpochEncoder(n_modalities=n_modalities, D=D, E=E)
        self.pos_enc = PositionalEncoding(D, max_len=64)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=D, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(D, n_classes)

    def forward(self, X):
        G = self.encoder(X)          # (batch, T, D)
        G = self.pos_enc(G)
        O = self.transformer(G)        # (batch, T, D)
        logits = self.head(O)
        return torch.softmax(logits, dim=-1)


class LSTMBaseline(nn.Module):
    """Single-layer LSTM per Table 5 naming, but 'depth=2' consistency is
    achieved via 2 stacked LSTM layers (num_layers=2 in nn.LSTM) --
    ASSUMPTION FLAGGED: the paper's "Single-layer LSTM" name and its
    "network depth kept consistent" requirement are in tension; we
    prioritize depth consistency with SBM (2 layers) over the literal
    "single-layer" name, since matching depth is the paper's explicit
    fairness criterion for this comparison."""

    def __init__(self, n_modalities=2, D=128, E=20, n_classes=5,
                 hidden_size=128, num_layers=2, dropout=0.0):
        super().__init__()
        self.encoder = EpochEncoder(n_modalities=n_modalities, D=D, E=E)
        self.lstm = nn.LSTM(input_size=D, hidden_size=hidden_size,
                             num_layers=num_layers, batch_first=True,
                             dropout=dropout if num_layers > 1 else 0.0)
        self.head = nn.Linear(hidden_size, n_classes)

    def forward(self, X):
        G = self.encoder(X)
        O, _ = self.lstm(G)
        logits = self.head(O)
        return torch.softmax(logits, dim=-1)


class BiLSTMBaseline(nn.Module):
    def __init__(self, n_modalities=2, D=128, E=20, n_classes=5,
                 hidden_size=128, num_layers=2, dropout=0.0):
        super().__init__()
        self.encoder = EpochEncoder(n_modalities=n_modalities, D=D, E=E)
        self.lstm = nn.LSTM(input_size=D, hidden_size=hidden_size,
                             num_layers=num_layers, batch_first=True,
                             bidirectional=True,
                             dropout=dropout if num_layers > 1 else 0.0)
        self.head = nn.Linear(hidden_size * 2, n_classes)  # *2 for bidirectional concat

    def forward(self, X):
        G = self.encoder(X)
        O, _ = self.lstm(G)
        logits = self.head(O)
        return torch.softmax(logits, dim=-1)


def _quick_selfcheck():
    print("=== Table 5 baseline models self-check ===\n")
    batch, T, C, L = 2, 5, 2, 3000
    dummy_X = torch.randn(batch, T, C, L)

    models = {
        "Transformer": TransformerBaseline(),
        "LSTM": LSTMBaseline(),
        "Bi-LSTM": BiLSTMBaseline(),
    }

    for name, model in models.items():
        y_hat = model(dummy_X)
        n_params = sum(p.numel() for p in model.parameters())
        prob_sums = y_hat.sum(dim=-1)
        max_dev = (prob_sums - 1.0).abs().max().item()
        assert y_hat.shape == (batch, T, 5)
        assert max_dev < 1e-4
        print(f"{name:12s}  output={tuple(y_hat.shape)}  params={n_params:,}  "
              f"softmax_check={'OK' if max_dev < 1e-4 else 'FAIL'}")

    print("\nAll 3 baselines pass shape/softmax checks.")
    print("\nPaper's Table 5 reference params (M): Transformer=25.10, Single-LSTM=11.96, Bi-LSTM=13.67")


if __name__ == "__main__":
    _quick_selfcheck()