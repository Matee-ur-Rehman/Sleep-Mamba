"""
SleepMamba reproduction — Fig. 4: t-SNE embeddings
================================================================================
Extracts epoch-level embeddings (the model's final contextualized
representation, before the classification head) on the test set, runs
t-SNE, and plots colored by true sleep stage -- reproducing Fig. 4's
visualization of learned latent structure.

We use the SBM output (the "o_t" per-epoch representation just before the
classification head, per Algorithm 1) as the embedding, since that's the
final learned representation the paper's Fig. 4 caption describes
("final epoch-level embeddings").

USAGE (after running kaggle_quick_checkpoint.py):
    !python Sleep-Mamba/kaggle_fig4_tsne.py
"""

import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE

sys.path.append("Sleep-Mamba/src/models")
sys.path.append("Sleep-Mamba/src/data")

from sleepmamba import SleepMamba
from dataset import SleepEDF78WindowedDataset

KAGGLE_PREPROCESSED_DIR = "/kaggle/input/datasets/mateeurrehman15/sleepmamba-preprocessed-sleepedf78/preprocessed"
KAGGLE_SPLITS_JSON = f"{KAGGLE_PREPROCESSED_DIR}/splits_sleepedf78.json"
CHECKPOINT_PATH = "/kaggle/working/checkpoint_eeg_eog_quick.pt"
OUT_PLOT_PATH = "/kaggle/working/fig4_tsne_sleepedf78.png"

T = 5
TEST_FOLD = 9
STAGE_NAMES = ["W", "N1", "N2", "N3", "REM"]
STAGE_COLORS = ["#FFD700", "#87CEEB", "#4CAF50", "#9C27B0", "#FF5722"]


@torch.no_grad()
def extract_embeddings(model, loader, device, D=128):
    model.eval()
    all_embeds, all_labels = [], []
    for x, y in loader:
        x = x.to(device)
        batch, T_, C, L = x.shape

        x_flat = x.reshape(batch * T_, C, L)
        F_pp = model.mle(x_flat)
        g_flat = model.dam_stack(F_pp)
        G = g_flat.reshape(batch, T_, D)
        O = model.sbm_stack(G)  # (batch, T, D) -- the embedding we want

        all_embeds.append(O.reshape(-1, D).cpu().numpy())
        all_labels.append(y.reshape(-1).numpy())

    return np.concatenate(all_embeds), np.concatenate(all_labels)


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
    print(f"Test windows: {len(test_ds):,} (-> {len(test_ds)*T:,} individual epochs)")

    model = SleepMamba(n_modalities=2, D=128, E=20, n_classes=5).to(device)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    print("Checkpoint loaded.")

    print("Extracting embeddings...")
    embeddings, labels = extract_embeddings(model, test_loader, device)
    print(f"Embeddings shape: {embeddings.shape}")

    # subsample if very large, to keep t-SNE runtime reasonable
    max_points = 5000
    if len(embeddings) > max_points:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(embeddings), max_points, replace=False)
        embeddings, labels = embeddings[idx], labels[idx]
        print(f"Subsampled to {max_points} points for t-SNE speed.")

    print("Running t-SNE (this may take a minute)...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, init="pca")
    embeddings_2d = tsne.fit_transform(embeddings)

    plt.figure(figsize=(8, 7))
    for stage_idx, (name, color) in enumerate(zip(STAGE_NAMES, STAGE_COLORS)):
        mask = labels == stage_idx
        plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                    c=color, label=name, s=8, alpha=0.6)
    plt.legend(title="Sleep Stage", loc="best")
    plt.title("t-SNE of SleepMamba epoch-level embeddings (SleepEDF-78, fold 9 test)")
    plt.xlabel("t-SNE dim 1")
    plt.ylabel("t-SNE dim 2")
    plt.tight_layout()
    plt.savefig(OUT_PLOT_PATH, dpi=150)
    print(f"\nPlot saved to {OUT_PLOT_PATH}")


if __name__ == "__main__":
    main()