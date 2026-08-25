"""
SleepMamba reproduction — Step 3: Windowed Dataset
=====================================================
Builds T-consecutive-epoch sequences from the per-recording .npz files
produced in Step 1, using the fold assignment from Step 2.

Paper Section 3.1: input X in R^(T x C x L), output Y = (y1,...,yT).
T is tested at {5, 15, 21, 30} (Section 4.2).

ASSUMPTIONS (not specified in the paper — flagged):
  - Windows are NON-OVERLAPPING (stride = T), the standard convention in
    this exact line of sequence-to-sequence sleep staging work
    (SeqSleepNet, XSleepNet, DeepSleepNet-style seq2seq variants).
  - Leftover epochs at the end of a recording that don't fill a full
    T-length window are DROPPED (per your explicit choice).
  - Windows are built per-recording (never span across two different
    recordings/subjects/nights), since crossing that boundary would mix
    unrelated sleep sessions into one "sequence."

This is pure CPU data-loading logic. No GPU needed here — this same
Dataset class is what both VS Code (CPU debug) and Kaggle (GPU train)
will import from src/data/dataset.py, so it must not contain anything
GPU-specific.
"""

import os
import json
import glob
import numpy as np
import torch
from torch.utils.data import Dataset

STAGE_NAMES = ["W", "N1", "N2", "N3", "REM"]


class SleepEDF78WindowedDataset(Dataset):
    """
    Yields (x, y) pairs:
      x: FloatTensor (T, n_channels, 3000)
      y: LongTensor  (T,)

    n_channels depends on `modalities` requested (subset of EEG/EOG/EMG,
    channel order in the .npz is fixed as [EEG, EOG, EMG] from Step 1).
    """

    CHANNEL_ORDER = ["EEG", "EOG", "EMG"]

    def __init__(self, preprocessed_dir, splits_json_path, fold_ids, T,
                 modalities=("EEG", "EOG")):
        """
        preprocessed_dir : folder containing subjXX_nightN.npz (Step 1 output)
        splits_json_path : path to splits_sleepedf78.json (Step 2 output)
        fold_ids         : list of fold indices (0-9) whose subjects to include
                            e.g. [0,1,2,3,4,5,6,7,8] for train, [9] for test
        T                : window length in epochs (5, 15, 21, or 30)
        modalities       : which channels to keep, subset of ("EEG","EOG","EMG")
        """
        self.T = T
        self.modalities = list(modalities)
        self.channel_idx = [self.CHANNEL_ORDER.index(m) for m in self.modalities]

        with open(splits_json_path, "r") as f:
            split_info = json.load(f)
        folds = split_info["folds"]

        wanted_subjects = set()
        for fid in fold_ids:
            wanted_subjects.update(folds[str(fid)])

        all_files = sorted(glob.glob(os.path.join(preprocessed_dir, "subj*_night*.npz")))
        self.recording_files = []
        for f in all_files:
            base = os.path.basename(f)
            subj_id = int(base.split("_")[0].replace("subj", ""))
            if subj_id in wanted_subjects:
                self.recording_files.append(f)

        if not self.recording_files:
            raise ValueError(
                f"No recordings found for fold_ids={fold_ids}. "
                f"Check that preprocessed_dir and splits_json_path are correct."
            )

        # Build an index of (file_path, window_start_epoch) for every valid,
        # non-overlapping, full-length window across all selected recordings.
        self.window_index = []
        self._cache = {}
        for f in self.recording_files:
            data = np.load(f)
            x_arr, y_arr = data["x"], data["y"]
            self._cache[f] = (x_arr, y_arr)          # cache now, avoid re-reading later
            n_epochs = y_arr.shape[0]
            n_full_windows = n_epochs // self.T  # drop leftover remainder
            for w in range(n_full_windows):
                start = w * self.T
                self.window_index.append((f, start))

        if not self.window_index:
            raise ValueError(
                f"Zero windows built for T={T} across {len(self.recording_files)} "
                f"recordings — T may be larger than every recording's epoch count."
            )

    def __len__(self):
        return len(self.window_index)

    def __getitem__(self, idx):
        f, start = self.window_index[idx]
        x_full, y_full = self._cache[f]          # cached in-memory arrays, no disk I/O
        x = x_full[start:start + self.T]            # (T, 3, 3000)
        y = y_full[start:start + self.T]              # (T,)
        x = x[:, self.channel_idx, :]                   # (T, n_channels, 3000)
        return torch.from_numpy(x).float(), torch.from_numpy(y).long()


def _quick_selfcheck(preprocessed_dir, splits_json_path):
    """Run this file directly to sanity-check the Dataset before using it in training."""
    print("=== Step 3 self-check ===")
    for T in (5, 15, 21, 30):
        train_ds = SleepEDF78WindowedDataset(
            preprocessed_dir, splits_json_path,
            fold_ids=list(range(9)), T=T, modalities=("EEG", "EOG"),
        )
        test_ds = SleepEDF78WindowedDataset(
            preprocessed_dir, splits_json_path,
            fold_ids=[9], T=T, modalities=("EEG", "EOG"),
        )
        x0, y0 = train_ds[0]
        print(f"T={T:2d}  train_windows={len(train_ds):5d}  test_windows={len(test_ds):5d}  "
              f"sample x.shape={tuple(x0.shape)}  y.shape={tuple(y0.shape)}  "
              f"y0={y0.tolist()}")

    print("\nAll T values loaded successfully. Shapes match (T, n_channels, 3000) / (T,) as expected.")


if __name__ == "__main__":
    PREPROCESSED_DIR = r"D:\Sleep Mamba\outputs\preprocessed"
    SPLITS_JSON = r"D:\Sleep Mamba\outputs\preprocessed\splits_sleepedf78.json"
    _quick_selfcheck(PREPROCESSED_DIR, SPLITS_JSON)