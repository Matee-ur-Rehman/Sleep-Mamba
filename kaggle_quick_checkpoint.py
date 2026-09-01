"""
SleepMamba reproduction — Quick checkpoint training (EEG+EOG, single split)
================================================================================
Table 8 (robustness) and Fig. 4 (t-SNE) both need a trained checkpoint of
the paper's actual Table 3 config: SleepMamba-5, EEG+EOG. We never saved
this specific checkpoint during our Table 3 runs (checkpoint-saving was
added later, starting with Table 6). This script trains ONE quick model
(single split: fold 9 test, folds 0-8 train, single seed) to get a usable
checkpoint fast, given a tight GPU time budget this week.

NOT a full reproduction run -- single split, single seed, not 10-fold.
Only used to produce a checkpoint for the evaluation-only experiments
that follow (Table 8, Fig. 4), which don't require multi-fold averaging.

USAGE:
    !python Sleep-Mamba/kaggle_quick_checkpoint.py
"""

import sys
import os
import torch
from torch.utils.data import DataLoader

sys.path.append("Sleep-Mamba/src/models")
sys.path.append("Sleep-Mamba/src/data")

from sleepmamba import SleepMamba
from dataset import SleepEDF78WindowedDataset
from train import train_one_fold

KAGGLE_PREPROCESSED_DIR = "/kaggle/input/datasets/mateeurrehman15/sleepmamba-preprocessed-sleepedf78/preprocessed"
KAGGLE_SPLITS_JSON = f"{KAGGLE_PREPROCESSED_DIR}/splits_sleepedf78.json"

T = 5
TEST_FOLD = 9
TRAIN_FOLDS = list(range(9))
CHECKPOINT_PATH = "/kaggle/working/checkpoint_eeg_eog_quick.pt"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type != "cuda":
        print("ERROR: no GPU detected. Aborting.")
        sys.exit(1)

    torch.manual_seed(0)

    train_ds = SleepEDF78WindowedDataset(
        KAGGLE_PREPROCESSED_DIR, KAGGLE_SPLITS_JSON,
        fold_ids=TRAIN_FOLDS, T=T, modalities=("EEG", "EOG"),
    )
    test_ds = SleepEDF78WindowedDataset(
        KAGGLE_PREPROCESSED_DIR, KAGGLE_SPLITS_JSON,
        fold_ids=[TEST_FOLD], T=T, modalities=("EEG", "EOG"),
    )
    print(f"Train windows: {len(train_ds):,}   Test windows: {len(test_ds):,}")

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0)

    model = SleepMamba(n_modalities=2, D=128, E=20, n_classes=5).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    best_metrics = train_one_fold(
        model, train_loader, test_loader, device,
        max_epochs=100, patience=10, lr=5e-4, weight_decay=0.01,
    )

    print(f"\nFinal: Acc={best_metrics['acc']:.4f}  Kappa={best_metrics['kappa']:.4f}  "
          f"MF1={best_metrics['mf1']:.4f}")

    torch.save(model.state_dict(), CHECKPOINT_PATH)
    print(f"Checkpoint saved to {CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()