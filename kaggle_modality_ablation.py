"""
SleepMamba reproduction — Table 6: Modality ablation
================================================================================
Section 5.4: "we assessed SleepMamba using four distinct modality
configurations: EEG only, EEG+EOG, EEG+EMG, and EEG+EOG+EMG, with all
models trained and tested under a consistent five-epoch input window."
(T=5 fixed for all four configs.)

We already have EEG+EOG (= our T=5 Table 3 result, in kfold_results_T5.json).
This script covers the 3 remaining configs: EEG, EEG+EMG, EEG+EOG+EMG.

Paper Table 6 targets (SleepEDF-78, T=5):
  EEG:            Acc=81.8%  kappa=0.746  MF1=74.9%  Sens=75.3%  Spec=95.2%
  EEG+EOG:        Acc=83.4%  kappa=0.768  MF1=77.2%  Sens=77.3%  Spec=95.6%  (already have this)
  EEG+EMG:        Acc=83.3%  kappa=0.767  MF1=76.9%  Sens=76.9%  Spec=95.5%
  EEG+EOG+EMG:    Acc=83.7%  kappa=0.773  MF1=77.8%  Sens=77.9%  Spec=95.6%

NEW IN THIS SCRIPT: model checkpoint saving. Previous k-fold runs only
saved metrics, not weights. Table 8 (robustness) and Fig. 4 (t-SNE) will
need real trained weights, so from this point on we save the best-epoch
model state_dict per fold, not just its metrics. This does increase
Kaggle output storage usage -- monitor /kaggle/working/ size if running
many configs.

USAGE:
    !python Sleep-Mamba/kaggle_modality_ablation.py eeg
    !python Sleep-Mamba/kaggle_modality_ablation.py eeg_emg
    !python Sleep-Mamba/kaggle_modality_ablation.py eeg_eog_emg
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
from train import train_one_fold, evaluate, compute_metrics

KAGGLE_PREPROCESSED_DIR = "/kaggle/input/datasets/mateeurrehman15/sleepmamba-preprocessed-sleepedf78/preprocessed"
KAGGLE_SPLITS_JSON = f"{KAGGLE_PREPROCESSED_DIR}/splits_sleepedf78.json"

K_FOLDS = 10
T = 5  # fixed for Table 6, per Section 5.4
MAX_EPOCHS = 100
PATIENCE = 10
LR = 5e-4
WEIGHT_DECAY = 0.01

MODALITY_CONFIGS = {
    "eeg":         (("EEG",),              1, {"acc": 0.818, "kappa": 0.746, "mf1": 0.749, "sensitivity": 0.753, "specificity": 0.952}),
    "eeg_emg":     (("EEG", "EMG"),        2, {"acc": 0.833, "kappa": 0.767, "mf1": 0.769, "sensitivity": 0.769, "specificity": 0.955}),
    "eeg_eog_emg": (("EEG", "EOG", "EMG"), 3, {"acc": 0.837, "kappa": 0.773, "mf1": 0.778, "sensitivity": 0.779, "specificity": 0.956}),
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
    if len(sys.argv) != 2 or sys.argv[1] not in MODALITY_CONFIGS:
        print(f"Usage: python kaggle_modality_ablation.py <config>")
        print(f"  config must be one of: {list(MODALITY_CONFIGS.keys())}")
        sys.exit(1)

    config_name = sys.argv[1]
    modalities, n_modalities, paper_target = MODALITY_CONFIGS[config_name]

    results_path = f"/kaggle/working/modality_results_{config_name}.json"
    checkpoint_dir = f"/kaggle/working/checkpoints_{config_name}"
    os.makedirs(checkpoint_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Config: {config_name}  Modalities: {modalities}  Device: {device}")
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
        print(f"[{config_name}] FOLD {test_fold}/{K_FOLDS-1}")
        print(f"{'='*70}")

        train_folds = [f for f in range(K_FOLDS) if f != test_fold]

        train_ds = SleepEDF78WindowedDataset(
            KAGGLE_PREPROCESSED_DIR, KAGGLE_SPLITS_JSON,
            fold_ids=train_folds, T=T, modalities=modalities,
        )
        test_ds = SleepEDF78WindowedDataset(
            KAGGLE_PREPROCESSED_DIR, KAGGLE_SPLITS_JSON,
            fold_ids=[test_fold], T=T, modalities=modalities,
        )
        print(f"Train windows: {len(train_ds):,}   Test windows: {len(test_ds):,}")

        train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0)
        test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0)

        model = SleepMamba(n_modalities=n_modalities, D=128, E=20, n_classes=5).to(device)

        start_time = time.time()
        best_metrics = train_one_fold(
            model, train_loader, test_loader, device,
            max_epochs=MAX_EPOCHS, patience=PATIENCE, lr=LR, weight_decay=WEIGHT_DECAY,
        )
        elapsed_min = (time.time() - start_time) / 60
        print(f"Fold {test_fold} finished in {elapsed_min:.1f} minutes. "
              f"Acc={best_metrics['acc']:.4f}  Kappa={best_metrics['kappa']:.4f}  "
              f"MF1={best_metrics['mf1']:.4f}")

        # NEW: save the trained model weights, not just metrics.
        ckpt_path = os.path.join(checkpoint_dir, f"fold{test_fold}.pt")
        torch.save(model.state_dict(), ckpt_path)
        print(f"Saved model checkpoint to {ckpt_path}")

        results[str(test_fold)] = best_metrics
        save_results(results, results_path)
        print(f"Saved results checkpoint to {results_path}")

        avg_so_far = average_metrics(results)
        print_comparison(avg_so_far, paper_target)

        del model, train_ds, test_ds, train_loader, test_loader
        torch.cuda.empty_cache()

    print(f"\n{'='*70}")
    print(f"[{config_name}]: ALL 10 FOLDS COMPLETE")
    print(f"{'='*70}")
    final_avg = average_metrics(results)
    print_comparison(final_avg, paper_target)
    print(f"\nFull per-fold results saved at: {results_path}")
    print(f"Checkpoints saved at: {checkpoint_dir}/foldN.pt")


if __name__ == "__main__":
    main()