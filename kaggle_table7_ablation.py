"""
SleepMamba reproduction — Table 7 training driver
================================================================================
Runs all 7 configs (full model + 6 ablation variants) x 3 random seeds each,
on a SINGLE FIXED train/test split (fold 9 as test, folds 0-8 as train --
the exact split validated in our earliest sanity check).

Protocol interpretation (flagged, not stated by the paper): "three
independent runs" = 3 random seeds on one fixed split, measuring training
stability. See ablation_variants.py's module docstring for full reasoning.

Reports mean +/- std across the 3 seeds per variant, matching the paper's
Table 7 format.

USAGE:
    !python Sleep-Mamba/kaggle_table7_ablation.py <variant>
where <variant> is one of:
    full, no_sdam, no_dam_intra, no_dam_inter, no_sbm,
    dam_independent_gates, dam_cross_attention
"""

import sys
import os
import json
import time
import random
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.append("Sleep-Mamba/src/models")
sys.path.append("Sleep-Mamba/src/data")

from ablation_variants import SleepMambaAblation
from dataset import SleepEDF78WindowedDataset
from train import train_one_fold

KAGGLE_PREPROCESSED_DIR = "/kaggle/input/datasets/mateeurrehman15/sleepmamba-preprocessed-sleepedf78/preprocessed"
KAGGLE_SPLITS_JSON = f"{KAGGLE_PREPROCESSED_DIR}/splits_sleepedf78.json"

T = 5
TEST_FOLD = 9
TRAIN_FOLDS = list(range(9))
SEEDS = [0, 1, 2]
MAX_EPOCHS = 100
PATIENCE = 10
LR = 5e-4
WEIGHT_DECAY = 0.01

VALID_VARIANTS = ("full", "no_sdam", "no_dam_intra", "no_dam_inter",
                   "no_sbm", "dam_independent_gates", "dam_cross_attention")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_existing_results(results_path):
    if os.path.exists(results_path):
        with open(results_path, "r") as f:
            return json.load(f)
    return {}


def save_results(results, results_path):
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)


def summarize(results):
    keys = ["acc", "kappa", "mf1"]
    summary = {}
    for k in keys:
        vals = [results[str(s)][k] for s in SEEDS if str(s) in results]
        if vals:
            summary[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
    return summary


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in VALID_VARIANTS:
        print(f"Usage: python kaggle_table7_ablation.py <variant>")
        print(f"  variant must be one of: {VALID_VARIANTS}")
        sys.exit(1)

    variant = sys.argv[1]
    results_path = f"/kaggle/working/table7_results_{variant}.json"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Variant: {variant}  Device: {device}")
    if device.type != "cuda":
        print("ERROR: no GPU detected. Aborting.")
        sys.exit(1)

    # Build datasets ONCE -- same fixed split reused across all 3 seeds
    train_ds = SleepEDF78WindowedDataset(
        KAGGLE_PREPROCESSED_DIR, KAGGLE_SPLITS_JSON,
        fold_ids=TRAIN_FOLDS, T=T, modalities=("EEG", "EOG"),
    )
    test_ds = SleepEDF78WindowedDataset(
        KAGGLE_PREPROCESSED_DIR, KAGGLE_SPLITS_JSON,
        fold_ids=[TEST_FOLD], T=T, modalities=("EEG", "EOG"),
    )
    print(f"Train windows: {len(train_ds):,}   Test windows: {len(test_ds):,}\n")

    results = load_existing_results(results_path)
    already_done = [int(k) for k in results.keys()]
    print(f"Already completed seeds (resuming if any): {sorted(already_done)}\n")

    for seed in SEEDS:
        if seed in already_done:
            print(f"Seed {seed}: already completed, skipping.")
            continue

        print(f"\n{'='*70}")
        print(f"[{variant}]  SEED {seed}")
        print(f"{'='*70}")
        set_seed(seed)

        train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0)
        test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0)

        model = SleepMambaAblation(variant=variant, n_modalities=2, D=128, E=20, n_classes=5).to(device)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {n_params:,}")

        start_time = time.time()
        best_metrics = train_one_fold(
            model, train_loader, test_loader, device,
            max_epochs=MAX_EPOCHS, patience=PATIENCE, lr=LR, weight_decay=WEIGHT_DECAY,
        )
        elapsed_min = (time.time() - start_time) / 60
        print(f"Seed {seed} finished in {elapsed_min:.1f} minutes. "
              f"Acc={best_metrics['acc']:.4f}  Kappa={best_metrics['kappa']:.4f}  "
              f"MF1={best_metrics['mf1']:.4f}")

        best_metrics["n_params"] = n_params
        results[str(seed)] = best_metrics
        save_results(results, results_path)
        print(f"Saved results checkpoint to {results_path}")

        del model
        torch.cuda.empty_cache()

    print(f"\n{'='*70}")
    print(f"[{variant}]: ALL 3 SEEDS COMPLETE")
    print(f"{'='*70}")
    summary = summarize(results)
    for k, v in summary.items():
        print(f"  {k:8s}  mean={v['mean']:.4f}  std={v['std']:.4f}")
    print(f"\nFull results saved at: {results_path}")


if __name__ == "__main__":
    main()