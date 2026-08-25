"""
SleepMamba reproduction — Full 10-fold CV run, parametrized by T
================================================================================
Generalized version of kaggle_full_kfold_run.py (which was hardcoded to T=5).
Accepts T as a command-line argument so the same script handles all four
paper-specified context lengths: T in {5, 15, 21, 30}.

USAGE (run once per T value, as separate Kaggle sessions/cells):
    !python Sleep-Mamba/kaggle_kfold_run.py 5
    !python Sleep-Mamba/kaggle_kfold_run.py 15
    !python Sleep-Mamba/kaggle_kfold_run.py 21
    !python Sleep-Mamba/kaggle_kfold_run.py 30

Each T gets its own results file (kfold_results_T{T}.json) and is fully
independently resumable, same checkpointing behavior as before.

Targets to compare against (paper Table 3, SleepEDF-78, EEG+EOG):
  T=5:  Acc=83.4%  kappa=0.768  MF1=77.2%  Sens=77.3%  Spec=95.5%
  T=15: Acc=83.8%  kappa=0.774  MF1=78.0%  Sens=78.4%  Spec=95.7%
  T=21: Acc=83.9%  kappa=0.775  MF1=77.9%  Sens=78.2%  Spec=95.7%
  T=30: Acc=84.1%  kappa=0.778  MF1=78.5%  Sens=79.0%  Spec=95.8%
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

K_FOLDS = 10
MAX_EPOCHS = 100
PATIENCE = 10
LR = 5e-4
WEIGHT_DECAY = 0.01

PAPER_TARGETS = {
    5:  {"acc": 0.834, "kappa": 0.768, "mf1": 0.772, "sensitivity": 0.773, "specificity": 0.955},
    15: {"acc": 0.838, "kappa": 0.774, "mf1": 0.780, "sensitivity": 0.784, "specificity": 0.957},
    21: {"acc": 0.839, "kappa": 0.775, "mf1": 0.779, "sensitivity": 0.782, "specificity": 0.957},
    30: {"acc": 0.841, "kappa": 0.778, "mf1": 0.785, "sensitivity": 0.790, "specificity": 0.958},
}


def load_existing_results(results_path):
    if os.path.exists(results_path):
        with open(results_path, "r") as f:
            return json.load(f)
    return {}


def save_results(results, results_path):
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)


def average_metrics(results):
    keys = ["acc", "kappa", "mf1", "sensitivity", "specificity"]
    avg = {}
    for k in keys:
        vals = [results[str(fold)][k] for fold in range(K_FOLDS) if str(fold) in results]
        avg[k] = sum(vals) / len(vals) if vals else None
    return avg


def print_comparison(avg, paper_target):
    print("\n=== Average across completed folds so far ===")
    for k in ["acc", "kappa", "mf1", "sensitivity", "specificity"]:
        ours = avg[k]
        paper = paper_target[k]
        if ours is not None:
            print(f"  {k:12s}  ours={ours:.4f}   paper={paper:.4f}   diff={ours-paper:+.4f}")


def main():
    if len(sys.argv) != 2 or int(sys.argv[1]) not in (5, 15, 21, 30):
        print("Usage: python kaggle_kfold_run.py <T>   where T is one of 5, 15, 21, 30")
        sys.exit(1)

    T = int(sys.argv[1])
    results_path = f"/kaggle/working/kfold_results_T{T}.json"
    paper_target = PAPER_TARGETS[T]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"T={T}   Device: {device}")
    if device.type != "cuda":
        print("ERROR: no GPU detected. This full run requires GPU. Aborting.")
        sys.exit(1)

    results = load_existing_results(results_path)
    already_done = [int(k) for k in results.keys()]
    print(f"Already completed folds (resuming if any): {sorted(already_done)}\n")

    for test_fold in range(K_FOLDS):
        if test_fold in already_done:
            print(f"Fold {test_fold}: already completed, skipping.")
            continue

        print(f"\n{'='*70}")
        print(f"T={T}  FOLD {test_fold}/{K_FOLDS-1}  (test_fold={test_fold}, train_folds=all others)")
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

        results[str(test_fold)] = best_metrics
        save_results(results, results_path)
        print(f"Saved results checkpoint to {results_path}")

        avg_so_far = average_metrics(results)
        print_comparison(avg_so_far, paper_target)

        del model, train_ds, test_ds, train_loader, test_loader
        torch.cuda.empty_cache()

    print(f"\n{'='*70}")
    print(f"T={T}: ALL 10 FOLDS COMPLETE")
    print(f"{'='*70}")
    final_avg = average_metrics(results)
    print_comparison(final_avg, paper_target)
    print(f"\nFull per-fold results saved at: {results_path}")


if __name__ == "__main__":
    main()