"""
SleepMamba reproduction — Full 10-fold CV run, T=5, EEG+EOG
================================================================================
This is the REAL reproduction run (not a sanity check), following the
paper's exact protocol from Section 4.2:
  - Subject-level 10-fold CV (fold i held out as test, other 9 as train)
  - max_epochs=100, early stopping patience=10, monitored on held-out fold
  - AdamW, lr=5e-4, weight_decay=0.01
  - T=5 (matches Table 5's directly-comparable config)

Target to compare against (paper Table 3, SleepMamba-5, SleepEDF-78):
  Acc=83.4%  kappa=0.768  MF1=77.2%  Sens=77.3%  Spec=95.5%

RESUMABILITY: results are saved incrementally to a JSON file after each
fold completes. If the Kaggle session disconnects or times out partway
through, re-running this script will automatically skip already-completed
folds and continue from where it left off, rather than starting over.

USAGE:
    !python Sleep-Mamba/kaggle_full_kfold_run.py
"""

import sys
import os
import json
import time
import torch
from torch.utils.data import DataLoader

sys.path.append("Sleep-Mamba/src/models")
sys.path.append("Sleep-Mamba/src/data")

from sleepmamba import SleepMamba
from dataset import SleepEDF78WindowedDataset
from train import train_one_fold

KAGGLE_PREPROCESSED_DIR = "/kaggle/input/datasets/mateeurrehman15/sleepmamba-preprocessed-sleepedf78/preprocessed"
KAGGLE_SPLITS_JSON = f"{KAGGLE_PREPROCESSED_DIR}/splits_sleepedf78.json"

# /kaggle/working/ is the writable output directory that persists for the
# session and can be downloaded from the notebook afterward.
RESULTS_PATH = "/kaggle/working/kfold_results_T5.json"

T = 5
K_FOLDS = 10
MAX_EPOCHS = 100
PATIENCE = 10
LR = 5e-4
WEIGHT_DECAY = 0.01

PAPER_TARGET = {"acc": 0.834, "kappa": 0.768, "mf1": 0.772, "sensitivity": 0.773, "specificity": 0.955}


def load_existing_results():
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH, "r") as f:
            return json.load(f)
    return {}


def save_results(results):
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)


def average_metrics(results):
    keys = ["acc", "kappa", "mf1", "sensitivity", "specificity"]
    avg = {}
    for k in keys:
        vals = [results[str(fold)][k] for fold in range(K_FOLDS) if str(fold) in results]
        avg[k] = sum(vals) / len(vals) if vals else None
    return avg


def print_comparison(avg):
    print("\n=== Average across completed folds so far ===")
    for k in ["acc", "kappa", "mf1", "sensitivity", "specificity"]:
        ours = avg[k]
        paper = PAPER_TARGET[k]
        if ours is not None:
            print(f"  {k:12s}  ours={ours:.4f}   paper={paper:.4f}   diff={ours-paper:+.4f}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type != "cuda":
        print("ERROR: no GPU detected. This full run requires GPU. Aborting.")
        sys.exit(1)

    results = load_existing_results()
    already_done = [int(k) for k in results.keys()]
    print(f"Already completed folds (resuming if any): {sorted(already_done)}\n")

    for test_fold in range(K_FOLDS):
        if test_fold in already_done:
            print(f"Fold {test_fold}: already completed, skipping.")
            continue

        print(f"\n{'='*70}")
        print(f"FOLD {test_fold}/{K_FOLDS-1}  (test_fold={test_fold}, train_folds=all others)")
        print(f"{'='*70}")

        train_folds = [f for f in range(K_FOLDS) if f != test_fold]

        train_ds = SleepEDF78WindowedDataset(
            KAGGLE_PREPROCESSED_DIR, KAGGLE_SPLITS_JSON,
            fold_ids=train_folds, T=T, modalities=("EEG", "EOG"),
        )
        test_ds = SleepEDF78WindowedDataset(
            KAGGLE_PREPROCESSED_DIR, KAGGLE_SPLITS_JSON,
            fold_ids=[test_fold], T=T, modalities=("EEG", "EOG"),
        )
        print(f"Train windows: {len(train_ds):,}   Test windows: {len(test_ds):,}")

        train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0)
        test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0)

        model = SleepMamba(n_modalities=2, D=128, E=20, n_classes=5).to(device)

        start_time = time.time()
        best_metrics = train_one_fold(
            model, train_loader, test_loader, device,
            max_epochs=MAX_EPOCHS, patience=PATIENCE, lr=LR, weight_decay=WEIGHT_DECAY,
        )
        elapsed_min = (time.time() - start_time) / 60
        print(f"Fold {test_fold} finished in {elapsed_min:.1f} minutes. "
              f"Acc={best_metrics['acc']:.4f}  Kappa={best_metrics['kappa']:.4f}  "
              f"MF1={best_metrics['mf1']:.4f}")

        # save immediately after each fold -- this is the resumability checkpoint
        results[str(test_fold)] = best_metrics
        save_results(results)
        print(f"Saved results checkpoint to {RESULTS_PATH}")

        avg_so_far = average_metrics(results)
        print_comparison(avg_so_far)

        del model, train_ds, test_ds, train_loader, test_loader
        torch.cuda.empty_cache()

    print(f"\n{'='*70}")
    print("ALL 10 FOLDS COMPLETE")
    print(f"{'='*70}")
    final_avg = average_metrics(results)
    print_comparison(final_avg)
    print(f"\nFull per-fold results saved at: {RESULTS_PATH}")


if __name__ == "__main__":
    main()