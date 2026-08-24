"""
SleepMamba reproduction — Step 2: Subject-level 10-fold CV split
====================================================================
Paper Section 4.2: "subject-level k-fold cross-validation protocol ...
in which the recordings of each subject are kept within a single fold.
Specifically, we set k = 10 for SleepEDF-78."

This script does NOT touch signal data at all — it only reads the
subject_id stored in each preprocessed .npz (from Step 1) and assigns
subjects to folds. Runs in under a second. No GPU needed, ever.

Output: a single JSON file, splits_sleepedf78.json, of the form:
{
  "k": 10,
  "folds": {
    "0": [subject_id, subject_id, ...],
    "1": [...],
    ...
    "9": [...]
  }
}
Each subject_id appears in EXACTLY ONE fold. Both nights of a subject
always land in the same fold, satisfied automatically since we group
by subject_id before splitting, never by recording.

This JSON is the single source of truth for CV membership — both the
VS Code CPU-debug code and the eventual Kaggle GPU training code must
load THIS file rather than regenerating splits independently.
"""

import os
import glob
import json
import numpy as np

PREPROCESSED_DIR = r"D:\Sleep Mamba\outputs\preprocessed"
OUT_JSON = r"D:\Sleep Mamba\outputs\preprocessed\splits_sleepedf78.json"

K_FOLDS = 10
SEED = 42  # NOT specified in the paper — our own choice, for reproducibility of OUR run only


def get_all_subject_ids(preprocessed_dir):
    files = sorted(glob.glob(os.path.join(preprocessed_dir, "subj*_night*.npz")))
    if not files:
        raise FileNotFoundError(
            f"No .npz files found in {preprocessed_dir} — did Step 1 (preprocessing) "
            f"actually write its output there? Check OUT_DIR in preprocess_sleepedf78.py."
        )
    subject_ids = set()
    for f in files:
        data = np.load(f)
        subject_ids.add(int(data["subject_id"]))
    return sorted(subject_ids)


def make_subject_level_kfold(subject_ids, k, seed):
    rng = np.random.default_rng(seed)
    subject_ids = np.array(subject_ids)
    rng.shuffle(subject_ids)  # shuffle once, then split into k contiguous chunks

    folds = {str(i): [] for i in range(k)}
    for i, subj in enumerate(subject_ids):
        fold_idx = i % k  # round-robin assignment -> balanced fold sizes
        folds[str(fold_idx)].append(int(subj))

    for i in range(k):
        folds[str(i)] = sorted(folds[str(i)])

    return folds


def main():
    subject_ids = get_all_subject_ids(PREPROCESSED_DIR)
    print(f"Found {len(subject_ids)} unique subjects across preprocessed recordings.")
    print(f"Subject IDs: {subject_ids}")

    if len(subject_ids) != 78:
        print(
            f"WARNING: expected 78 subjects (paper: 78 subjects, 153 recordings), "
            f"found {len(subject_ids)}. Check Step 1 output before proceeding."
        )

    folds = make_subject_level_kfold(subject_ids, K_FOLDS, SEED)

    print("\nFold sizes (in subjects):")
    for i in range(K_FOLDS):
        print(f"  Fold {i}: {len(folds[str(i)])} subjects -> {folds[str(i)]}")

    # sanity check: no subject appears in more than one fold, all subjects covered
    all_assigned = sorted([s for f in folds.values() for s in f])
    assert all_assigned == sorted(subject_ids), "Subject coverage/overlap mismatch!"
    print("\nSanity check passed: every subject assigned to exactly one fold.")

    out = {"k": K_FOLDS, "seed": SEED, "folds": folds}
    with open(OUT_JSON, "w") as fp:
        json.dump(out, fp, indent=2)
    print(f"\nSaved fold assignment to: {OUT_JSON}")


if __name__ == "__main__":
    main()