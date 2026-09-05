"""
SleepMamba reproduction — Table 5: Baseline full 10-fold training
================================================================================
Full 10-fold CV training for the three Table 5 baselines (Transformer,
LSTM, Bi-LSTM), matching Table 3's exact protocol (justified: Table 5's
own SleepMamba row is identical to Table 3's 10-fold result).

USAGE:
    !python Sleep-Mamba/kaggle_table5_baseline.py transformer
    !python Sleep-Mamba/kaggle_table5_baseline.py lstm
    !python Sleep-Mamba/kaggle_table5_baseline.py bilstm
"""

import sys
import os
import json
import time
import torch
from torch.utils.data import DataLoader

sys.path.append("Sleep-Mamba/src/models")
sys.path.append("Sleep-Mamba/src/data")

from baseline_models import TransformerBaseline, LSTMBaseline, BiLSTMBaseline
from dataset import SleepEDF78WindowedDataset
from train import train_one_fold

KAGGLE_PREPROCESSED_DIR = "/kaggle/input/datasets/mateeurrehman15/sleepmamba-preprocessed-sleepedf78/preprocessed"
KAGGLE_SPLITS_JSON = f"{KAGGLE_PREPROCESSED_DIR}/splits_sleepedf78.json"

K_FOLDS = 10
T = 5
MAX_EPOCHS = 100
PATIENCE = 10
LR = 5e-4
WEIGHT_DECAY = 0.01

MODEL_FACTORIES = {
    "transformer": lambda: TransformerBaseline(n_modalities=2, D=128, E=20, n_classes=5),
    "lstm":        lambda: LSTMBaseline(n_modalities=2, D=128, E=20, n_classes=5),
    "bilstm":      lambda: BiLSTMBaseline(n_modalities=2, D=128, E=20, n_classes=5),
}

PAPER_TARGETS = {
    "transformer": {"acc": 0.816, "kappa": 0.742, "mf1": 0.741},
    "lstm":        {"acc": 0.826, "kappa": 0.755, "mf1": 0.760},
    "bilstm":      {"acc": 0.828, "kappa": 0.759, "mf1": 0.767},
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
    for k in ["acc", "kappa", "mf1"]:
        ours = avg[k]
        paper = paper_target[k]
        if ours is not None:
            print(f"  {k:8s}  ours={ours:.4f}   paper={paper:.4f}   diff={ours-paper:+.4f}")


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in MODEL_FACTORIES:
        print(f"Usage: python kaggle_table5_baseline.py <model>")
        print(f"  model must be one of: {list(MODEL_FACTORIES.keys())}")
        sys.exit(1)

    model_name = sys.argv[1]
    results_path = f"/kaggle/working/table5_results_{model_name}.json"
    paper_target = PAPER_TARGETS[model_name]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Model: {model_name}  Device: {device}")
    if device.type != "cuda":
        print("ERROR: no GPU detected. Aborting.")
        sys.exit(1)

    results = load_existing_results(results_path)
    already_done = [int(k) for k in results.keys()]
    print(f"Already completed folds (resuming if any): {sorted(already_done)}\n")

    for test_fold in range(K_FOLDS):
        if test_fold in already_done:
            print(f"Fold {test_fold}: already completed, skipping.")
            continue

        print(f"\n{'='*70}")
        print(f"[{model_name}]  FOLD {test_fold}/{K_FOLDS-1}")
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

        model = MODEL_FACTORIES[model_name]().to(device)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {n_params:,}")

        start_time = time.time()
        best_metrics = train_one_fold(
            model, train_loader, test_loader, device,
            max_epochs=MAX_EPOCHS, patience=PATIENCE, lr=LR, weight_decay=WEIGHT_DECAY,
        )
        elapsed_min = (time.time() - start_time) / 60
        print(f"Fold {test_fold} finished in {elapsed_min:.1f} minutes. "
              f"Acc={best_metrics['acc']:.4f}  Kappa={best_metrics['kappa']:.4f}  "
              f"MF1={best_metrics['mf1']:.4f}")

        best_metrics["n_params"] = n_params
        results[str(test_fold)] = best_metrics
        save_results(results, results_path)
        print(f"Saved results checkpoint to {results_path}")

        avg_so_far = average_metrics(results)
        print_comparison(avg_so_far, paper_target)

        del model, train_ds, test_ds, train_loader, test_loader
        torch.cuda.empty_cache()

    print(f"\n{'='*70}")
    print(f"[{model_name}]: ALL 10 FOLDS COMPLETE")
    print(f"{'='*70}")
    final_avg = average_metrics(results)
    print_comparison(final_avg, paper_target)
    print(f"\nFull per-fold results saved at: {results_path}")


if __name__ == "__main__":
    main()