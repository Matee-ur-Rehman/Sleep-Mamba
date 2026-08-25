"""
SleepMamba reproduction — Kaggle Step: first real-data sanity training run
================================================================================
NOT the full reproduction run. This is a small, fast check to confirm the
ENTIRE pipeline (real .npz data -> windowed Dataset -> real SleepMamba model
with real mamba-ssm backend -> training loop -> metrics) works correctly on
actual SleepEDF-78 data, before committing GPU-hours to the full 10-fold x
4-T sweep described in the paper.

Uses: T=5 only, a SINGLE fold as held-out test (fold 9), all other folds as
train, and a small max_epochs so it finishes quickly (minutes, not hours).

USAGE (as a Kaggle notebook cell, after cloning the repo and confirming the
dataset path):

    !python Sleep-Mamba/kaggle_sanity_run.py
"""

import sys
import torch
from torch.utils.data import DataLoader

sys.path.append("Sleep-Mamba/src/models")
sys.path.append("Sleep-Mamba/src/data")

from sleepmamba import SleepMamba
from dataset import SleepEDF78WindowedDataset
from train import train_one_fold

# Real path confirmed on Kaggle (note: NOT the standard /kaggle/input/<name>/
# pattern -- this Kaggle account's mount uses an extra datasets/<username>/
# namespacing layer, confirmed by direct `ls` exploration)
KAGGLE_PREPROCESSED_DIR = "/kaggle/input/datasets/mateeurrehman15/sleepmamba-preprocessed-sleepedf78/preprocessed"
KAGGLE_SPLITS_JSON = f"{KAGGLE_PREPROCESSED_DIR}/splits_sleepedf78.json"

T = 5
TEST_FOLD = 9
TRAIN_FOLDS = list(range(9))  # folds 0-8

SANITY_MAX_EPOCHS = 5   # small on purpose -- this is a pipeline check, not real training
SANITY_PATIENCE = 3


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type != "cuda":
        print("WARNING: no GPU detected. This sanity run will be very slow "
              "and won't reflect real training conditions.")

    print(f"\nBuilding datasets (T={T}, test_fold={TEST_FOLD})...")
    train_ds = SleepEDF78WindowedDataset(
        KAGGLE_PREPROCESSED_DIR, KAGGLE_SPLITS_JSON,
        fold_ids=TRAIN_FOLDS, T=T, modalities=("EEG", "EOG"),
    )
    test_ds = SleepEDF78WindowedDataset(
        KAGGLE_PREPROCESSED_DIR, KAGGLE_SPLITS_JSON,
        fold_ids=[TEST_FOLD], T=T, modalities=("EEG", "EOG"),
    )
    print(f"Train windows: {len(train_ds):,}   Test windows: {len(test_ds):,}")

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=2)

    print("\nBuilding model...")
    model = SleepMamba(n_modalities=2, D=128, E=20, n_classes=5).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    print(f"\nStarting sanity training run (max_epochs={SANITY_MAX_EPOCHS}, "
          f"patience={SANITY_PATIENCE})...\n")
    best_metrics = train_one_fold(
        model, train_loader, test_loader, device,
        max_epochs=SANITY_MAX_EPOCHS, patience=SANITY_PATIENCE,
        lr=5e-4, weight_decay=0.01,
    )

    print("\n=== Sanity run finished ===")
    print(f"Best test accuracy this short run: {best_metrics['acc']:.4f}")
    print(f"Best test kappa this short run:    {best_metrics['kappa']:.4f}")
    print(f"Best test MF1 this short run:      {best_metrics['mf1']:.4f}")
    print("\n(These numbers are from only 5 epochs on 1 fold -- NOT comparable")
    print("to the paper's Table 3 results yet. This only confirms the full")
    print("real-data pipeline works correctly end-to-end.)")


if __name__ == "__main__":
    main()