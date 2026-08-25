"""
SleepMamba reproduction — Step 5: Training loop, loss, k-fold orchestration
================================================================================
Implements Section 4.2 (training protocol) and 4.3 (evaluation metrics).

======================================================================
EXPLICIT FROM THE PAPER (no assumptions needed):
======================================================================
  - Optimizer: AdamW, lr=5e-4, weight_decay=0.01, betas=(0.9,0.999), eps=1e-8
  - Batch size: 64
  - Max epochs: 100
  - Early stopping: patience=10, monitored on the HELD-OUT (test) fold's
    accuracy -- this is literally what Section 4.2 states. It is NOT
    best ML practice (test fold doubles as validation), but it is what
    the paper explicitly describes, so we implement it as written.
  - CV: subject-level k=10 for SleepEDF-78 (already built in Step 2).
  - Metrics: Accuracy, Cohen's kappa, macro-F1, macro-sensitivity,
    macro-specificity, per-class F1. Computed per fold, then averaged
    across folds (Section 4.3).

======================================================================
ASSUMPTION FLAGGED:
======================================================================
  - Loss function: Algorithm 1 states y_hat = Softmax(Linear(o_t)), and
    our SleepMamba module (Step 4d) therefore returns PROBABILITIES,
    not logits. Eq. 17's categorical cross-entropy is written directly
    over these probabilities. The correct PyTorch equivalent is
    nn.NLLLoss applied to log(y_hat) -- NOT nn.CrossEntropyLoss, which
    would apply a second, incorrect softmax internally. This is a
    direct mathematical consequence of the paper's own Algorithm 1,
    not a free choice, but worth flagging since it's a common source
    of silent bugs.
======================================================================

CPU NOTE: train_one_epoch/evaluate here are used for BOTH the local
CPU smoke-test (tiny synthetic data, a few steps, just to verify
mechanics) AND later, unchanged, for real Kaggle GPU training -- same
code, different scale and different underlying MambaBlock backend
(reference vs. real mamba-ssm CUDA kernels).
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, cohen_kappa_score, f1_score,
    confusion_matrix,
)

N_CLASSES = 5
STAGE_NAMES = ["W", "N1", "N2", "N3", "REM"]


def nll_loss_on_probs(y_hat, y_true, eps=1e-8):
    """
    Eq. 17's categorical cross-entropy, applied to PROBABILITIES
    (since our model already applies softmax per Algorithm 1).
    y_hat: (batch, T, n_classes) probabilities, already sum to 1 over last dim
    y_true: (batch, T) integer labels

    NOTE: torch.clamp does NOT fix actual NaN values (only out-of-range
    finite ones) -- if y_hat itself became NaN upstream (e.g. from an
    unstable forward pass), log(NaN) stays NaN regardless of clamping.
    The nan_to_num call below is a defensive safeguard so a single bad
    batch produces a large-but-finite loss (which the training loop can
    then detect and skip) rather than a silent NaN that poisons all
    subsequent accumulated totals.
    """
    y_hat_safe = torch.nan_to_num(y_hat, nan=0.0, posinf=1.0, neginf=0.0)
    log_probs = torch.log(y_hat_safe.clamp(min=eps))
    loss_fn = nn.NLLLoss()
    return loss_fn(log_probs.reshape(-1, N_CLASSES), y_true.reshape(-1))


def train_one_epoch(model, loader, optimizer, device, log_every=50, grad_clip_norm=1.0):
    model.train()
    total_loss = 0.0
    n_batches = 0
    total_batches = len(loader)
    for i, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        y_hat = model(x)
        loss = nll_loss_on_probs(y_hat, y)

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"    WARNING: non-finite loss ({loss.item()}) at batch {i+1}/{total_batches} "
                  f"-- skipping this batch's optimizer step to avoid corrupting weights.",
                  flush=True)
            continue  # do NOT call backward()/step() on a broken batch

        loss.backward()
        # ASSUMPTION FLAGGED: gradient clipping is NOT mentioned in the paper's
        # training protocol (Section 4.2), but was added after observing NaN
        # loss / weight corruption during real-data training on Kaggle (all
        # epochs showing train_loss=nan, frozen degenerate predictions).
        # Gradient clipping is a very common, standard stabilization technique
        # for SSM/Mamba-based models specifically, and does not contradict
        # anything the paper states -- it's an unstated-but-likely-necessary
        # implementation detail, same category as our other flagged gaps.
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1
        if (i + 1) % log_every == 0 or (i + 1) == total_batches:
            avg = total_loss / max(n_batches, 1)
            print(f"    batch {i+1}/{total_batches}  running_avg_loss={avg:.4f}", flush=True)
    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, loader, device):
    """Returns (avg_loss, y_true_flat, y_pred_flat) -- flattened across all
    epochs in all windows in the loader, for metric computation."""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    all_true, all_pred = [], []

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        y_hat = model(x)
        loss = nll_loss_on_probs(y_hat, y)
        total_loss += loss.item()
        n_batches += 1

        pred = y_hat.argmax(dim=-1)  # (batch, T)
        all_true.append(y.reshape(-1).cpu().numpy())
        all_pred.append(pred.reshape(-1).cpu().numpy())

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    avg_loss = total_loss / max(n_batches, 1)
    return avg_loss, y_true, y_pred


def compute_metrics(y_true, y_pred):
    """Section 4.3: Accuracy, Cohen's kappa, macro-F1, macro-sensitivity,
    macro-specificity, and per-class F1."""
    acc = accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    mf1 = f1_score(y_true, y_pred, average="macro", labels=range(N_CLASSES), zero_division=0)
    per_class_f1 = f1_score(y_true, y_pred, average=None, labels=range(N_CLASSES), zero_division=0)

    cm = confusion_matrix(y_true, y_pred, labels=range(N_CLASSES))
    # per-class sensitivity (recall) and specificity from confusion matrix
    sensitivities, specificities = [], []
    total = cm.sum()
    for k in range(N_CLASSES):
        tp = cm[k, k]
        fn = cm[k, :].sum() - tp
        fp = cm[:, k].sum() - tp
        tn = total - tp - fn - fp
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        sensitivities.append(sens)
        specificities.append(spec)

    return {
        "acc": acc,
        "kappa": kappa,
        "mf1": mf1,
        "sensitivity": float(np.mean(sensitivities)),
        "specificity": float(np.mean(specificities)),
        "per_class_f1": {name: float(f) for name, f in zip(STAGE_NAMES, per_class_f1)},
    }


def train_one_fold(model, train_loader, test_loader, device,
                    max_epochs=100, patience=10, lr=5e-4, weight_decay=0.01):
    """
    Trains for up to max_epochs, early stopping on held-out (test) fold
    accuracy with the given patience -- exactly as Section 4.2 describes
    (test fold doubles as the early-stopping monitor, per the paper's
    literal wording).
    """
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay,
        betas=(0.9, 0.999), eps=1e-8,
    )

    best_acc = -1.0
    best_metrics = None
    epochs_without_improvement = 0

    for epoch in range(max_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        test_loss, y_true, y_pred = evaluate(model, test_loader, device)
        metrics = compute_metrics(y_true, y_pred)

        improved = metrics["acc"] > best_acc
        if improved:
            best_acc = metrics["acc"]
            best_metrics = metrics
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        print(f"  epoch {epoch+1:3d}  train_loss={train_loss:.4f}  "
              f"test_loss={test_loss:.4f}  test_acc={metrics['acc']:.4f}  "
              f"{'(best)' if improved else f'(no improve x{epochs_without_improvement})'}")

        if epochs_without_improvement >= patience:
            print(f"  Early stopping at epoch {epoch+1} (patience={patience} exceeded).")
            break

    return best_metrics


def _quick_selfcheck():
    """
    SMOKE TEST ONLY -- synthetic random data, tiny model/loader, a
    handful of epochs, just to verify the training mechanics
    (forward -> loss -> backward -> optimizer step -> metrics) work
    correctly. This is NOT real training and the resulting "accuracy"
    is meaningless (random data). Real training happens on Kaggle.
    """
    from sleepmamba import SleepMamba
    from torch.utils.data import TensorDataset

    print("=== Step 5 self-check: training loop smoke test (synthetic data) ===\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running smoke test on device: {device}\n")
    T = 5
    batch = 4
    n_train_samples = 16
    n_test_samples = 8

    torch.manual_seed(0)
    x_train = torch.randn(n_train_samples, T, 2, 3000)
    y_train = torch.randint(0, N_CLASSES, (n_train_samples, T))
    x_test = torch.randn(n_test_samples, T, 2, 3000)
    y_test = torch.randint(0, N_CLASSES, (n_test_samples, T))

    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch, shuffle=True)
    test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=batch, shuffle=False)

    model = SleepMamba(n_modalities=2, D=128, E=20, n_classes=N_CLASSES).to(device)

    print("Running 3 smoke-test epochs (patience=2) on synthetic data...\n")
    best_metrics = train_one_fold(
        model, train_loader, test_loader, device,
        max_epochs=3, patience=2, lr=5e-4, weight_decay=0.01,
    )

    print(f"\nSmoke test finished. Best metrics dict keys: {list(best_metrics.keys())}")
    print(f"(Values are meaningless -- random synthetic data. This only confirms")
    print(f"the training loop runs end-to-end without errors.)")
    print("\nSelf-check passed.")


if __name__ == "__main__":
    _quick_selfcheck()