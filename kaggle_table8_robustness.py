"""
SleepMamba reproduction — Table 8: Robustness evaluation
================================================================================
Section 5.6: test-time corruption experiments on the trained SleepMamba-5
(EEG+EOG) model, evaluated WITHOUT retraining/fine-tuning.

Corruptions implemented, per Section 5.6's exact description:
  1. Temporal masking: continuous segments within each epoch and channel
     randomly replaced with zeros, at ratios {10%, 20%, 30%, 40%}.
  2. Channel masking: entire EEG or EOG channel set to zero (EEG-only /
     EOG-only conditions).
  3. Gaussian noise: added to normalized test signals at SNR levels
     {30, 25, 20, 15, 10} dB, using the paper's exact formula:
       sigma_n = sigma_X / 10^(SNR/20)

NOTE: our data is already z-score normalized at preprocessing time
(Step 1), matching the paper's "added to the normalized test signals"
description directly -- no extra normalization needed here.

We evaluate ONLY our own SleepMamba model (not SeqSleepNet/XSleepNet2/
SCMT-5, which the paper also compares against -- reproducing those three
additional architectures was explicitly descoped earlier as
disproportionate to this table's value; see prior discussion).

USAGE (after running kaggle_quick_checkpoint.py):
    !python Sleep-Mamba/kaggle_table8_robustness.py
"""

import sys
import os
import json
import torch
import numpy as np
from torch.utils.data import DataLoader

sys.path.append("Sleep-Mamba/src/models")
sys.path.append("Sleep-Mamba/src/data")

from sleepmamba import SleepMamba
from dataset import SleepEDF78WindowedDataset
from train import compute_metrics

KAGGLE_PREPROCESSED_DIR = "/kaggle/input/datasets/mateeurrehman15/sleepmamba-preprocessed-sleepedf78/preprocessed"
KAGGLE_SPLITS_JSON = f"{KAGGLE_PREPROCESSED_DIR}/splits_sleepedf78.json"
CHECKPOINT_PATH = "/kaggle/working/checkpoint_eeg_eog_quick.pt"
RESULTS_PATH = "/kaggle/working/table8_robustness_results.json"

T = 5
TEST_FOLD = 9
EEG_CH, EOG_CH = 0, 1  # channel order in our windowed dataset


def temporal_mask(x, ratio, rng):
    # x: (batch, T, C, L) -- mask a random contiguous segment per channel, per epoch
    x = x.clone()
    L = x.shape[-1]
    mask_len = int(ratio * L)
    if mask_len == 0:
        return x
    batch, T_, C, _ = x.shape
    for b in range(batch):
        for t in range(T_):
            for c in range(C):
                start = rng.integers(0, L - mask_len + 1)
                x[b, t, c, start:start + mask_len] = 0.0
    return x


def channel_mask(x, keep_channel):
    x = x.clone()
    if keep_channel == "eeg":
        x[:, :, EOG_CH, :] = 0.0
    elif keep_channel == "eog":
        x[:, :, EEG_CH, :] = 0.0
    return x


def gaussian_noise(x, snr_db, rng):
    x = x.clone()
    sigma_x = x.std(dim=-1, keepdim=True)  # per (batch,T,C) signal std
    sigma_n = sigma_x / (10 ** (snr_db / 20))
    noise = torch.from_numpy(rng.normal(0, 1, size=x.shape)).float() * sigma_n
    return x + noise


@torch.no_grad()
def evaluate_corrupted(model, loader, device, corrupt_fn):
    model.eval()
    all_true, all_pred = [], []
    for x, y in loader:
        if corrupt_fn is not None:
            x = corrupt_fn(x)
        x, y = x.to(device), y.to(device)
        y_hat = model(x)
        pred = y_hat.argmax(dim=-1)
        all_true.append(y.reshape(-1).cpu().numpy())
        all_pred.append(pred.reshape(-1).cpu().numpy())
    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    return compute_metrics(y_true, y_pred)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if not os.path.exists(CHECKPOINT_PATH):
        print(f"ERROR: checkpoint not found at {CHECKPOINT_PATH}. "
              f"Run kaggle_quick_checkpoint.py first.")
        sys.exit(1)

    test_ds = SleepEDF78WindowedDataset(
        KAGGLE_PREPROCESSED_DIR, KAGGLE_SPLITS_JSON,
        fold_ids=[TEST_FOLD], T=T, modalities=("EEG", "EOG"),
    )
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0)
    print(f"Test windows: {len(test_ds):,}")

    model = SleepMamba(n_modalities=2, D=128, E=20, n_classes=5).to(device)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    print("Checkpoint loaded.")

    rng = np.random.default_rng(42)
    results = {}

    print("\n--- Clean baseline ---")
    results["clean"] = evaluate_corrupted(model, test_loader, device, None)
    print(f"  Acc={results['clean']['acc']:.4f}")

    print("\n--- Temporal masking ---")
    for ratio in (0.10, 0.20, 0.30, 0.40):
        key = f"temporal_mask_{int(ratio*100)}pct"
        fn = lambda x, r=ratio: temporal_mask(x, r, rng)
        results[key] = evaluate_corrupted(model, test_loader, device, fn)
        print(f"  {key}: Acc={results[key]['acc']:.4f}")

    print("\n--- Channel masking ---")
    for keep in ("eeg", "eog"):
        key = f"channel_mask_{keep}_only"
        fn = lambda x, k=keep: channel_mask(x, k)
        results[key] = evaluate_corrupted(model, test_loader, device, fn)
        print(f"  {key}: Acc={results[key]['acc']:.4f}")

    print("\n--- Gaussian noise ---")
    for snr in (30, 25, 20, 15, 10):
        key = f"gaussian_noise_{snr}dB"
        fn = lambda x, s=snr: gaussian_noise(x, s, rng)
        results[key] = evaluate_corrupted(model, test_loader, device, fn)
        print(f"  {key}: Acc={results[key]['acc']:.4f}")

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nAll results saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()