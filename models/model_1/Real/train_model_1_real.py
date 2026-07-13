"""
Model 1 (Real) -- Single-lens vs. Binary-lens classifier
=========================================================

Trains a 1D Convolutional Neural Network to classify gravitational
microlensing light curves as single-lens (1) or binary-lens (2).

"Real" framework: trained on OGLE-IV-like I(t) light curves WITH photometric
noise and cadence gaps. This is the realistic counterpart to the "Simple"
model (``models/model_1/Simple/train_model_1_simple.py``), which is trained on
perfect curves and sets the upper-bound performance.

The two new complications versus Simple, and how they are handled:

  1. Photometric noise -- N(0, sigma(I)^2) added to observed magnitudes.
     Nothing special is needed; the network learns to see through it.

  2. Cadence gaps -- some t_nnn columns are NaN (the point was never observed).
     A CNN cannot ingest NaNs, so each curve is turned into a TWO-channel
     input:
         channel 0 : per-curve z-scored magnitude, gaps filled with 0
         channel 1 : observed mask (1 = real measurement, 0 = cadence gap)
     The mask lets the model distinguish a genuinely flat baseline from a
     stretch of missing data. The only architectural change from Simple is
     the first conv layer: in_channels 1 -> 2.

Dataset
-------
CSV with 401 columns:
    event_lenses , t_000 , t_001 , ... , t_399
where event_lenses is 1 (single) or 2 (binary) and t_nnn are I-band
magnitudes sampled on tau in [-3, 3]. Cadence-gap columns are empty (NaN).

Generate it from the webapp (I(t) mode with the OGLE noise option enabled).
Drop the CSV into this folder; the script auto-detects it (prefers a filename
containing "OGLE").

Labels are remapped 1 -> 0 (single), 2 -> 1 (binary) for the sigmoid output.
Class imbalance (90/10) is handled with a weighted loss (pos_weight = 9).

Decision threshold
------------------
The network outputs a probability, and a cut-off is needed to turn it into a
class. 0.5 is NOT that cut-off here: pos_weight=9 deliberately shifts the
probabilities upward, so cutting at 0.5 corrects for the imbalance a second time
and floods the binary class with false positives (measured: precision 0.83 vs
recall 0.91). The threshold is therefore selected by maximising F1 on the
VALIDATION set -- never on the test set, which would leak and inflate the result
-- and the test set is then evaluated once at that threshold. It is stored in the
checkpoint as ``decision_threshold`` and inference must use it.

Outputs (written next to this script)
-------------------------------------
    model_1_real.pt            trained weights + normalization stats + config
    training_history.png       loss / F1 curves over epochs
    confusion_matrix.png       confusion matrix on the held-out test set

Run
---
    venv/Scripts/python.exe models/model_1/Real/train_model_1_real.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
import torch.nn as nn
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
# pyrefly: ignore [missing-import]
from torch.utils.data import DataLoader, TensorDataset

import matplotlib

matplotlib.use("Agg")  # headless -- save figures without a display
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
HERE = Path(__file__).resolve().parent

MODEL_OUT = HERE / "model_1_real.pt"
HISTORY_PLOT = HERE / "training_history.png"
CONFUSION_PLOT = HERE / "confusion_matrix.png"

N_POINTS = 400          # light-curve length
BATCH_SIZE = 256
EPOCHS = 40
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
TEST_SIZE = 0.15        # held-out test fraction
VAL_SIZE = 0.15         # validation fraction (of the remaining train pool)
PATIENCE = 8            # early-stopping patience (epochs without val-F1 gain)
SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def find_dataset(folder: Path) -> Path:
    """Locate the training CSV in ``folder``.

    Prefers a filename containing "OGLE" (the noisy/gapped export); otherwise
    falls back to the only CSV present. Raises if zero or ambiguous.
    """
    csvs = sorted(folder.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(
            f"No CSV dataset found in {folder}\n"
            "Generate one from the webapp (I(t) mode, OGLE noise enabled) and "
            "drop it into this folder."
        )
    ogle = [c for c in csvs if "ogle" in c.name.lower()]
    if ogle:
        return ogle[0]
    if len(csvs) == 1:
        return csvs[0]
    raise FileNotFoundError(
        f"Multiple CSVs found in {folder} and none named *OGLE*; "
        f"please leave only the intended one:\n  "
        + "\n  ".join(c.name for c in csvs)
    )


# --------------------------------------------------------------------------- #
# Data loading & preprocessing
# --------------------------------------------------------------------------- #
def load_dataset(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load the CSV and return (X, y).

    X : (N, 400) float32 light-curve magnitudes -- may contain NaN (cadence gaps)
    y : (N,)     float32 labels -- 0 = single-lens, 1 = binary-lens
    """
    print(f"Loading dataset: {csv_path.name}")
    df = pd.read_csv(csv_path)

    label_col = "event_lenses"
    # Only true light-curve columns t_000..t_399 -- exclude parameter columns
    # that also start with "t_" (e.g. t_E_days), matching webapp/app/model1.py.
    point_cols = [c for c in df.columns if c.startswith("t_") and str(c)[2:].isdigit()]
    if len(point_cols) != N_POINTS:
        raise ValueError(
            f"Expected {N_POINTS} light-curve columns, found {len(point_cols)}"
        )

    # Order columns by numeric time index (t_000, t_001, ...).
    point_cols = sorted(point_cols, key=lambda c: int(str(c)[2:]))
    X = df[point_cols].to_numpy(dtype=np.float32)
    raw_labels = df[label_col].to_numpy()

    # 1 (single) -> 0, 2 (binary) -> 1
    y = (raw_labels == 2).astype(np.float32)

    n_single = int((y == 0).sum())
    n_binary = int((y == 1).sum())
    frac_gap = float(np.isnan(X).mean())
    print(f"  events: {len(y):,}  |  single: {n_single:,}  binary: {n_binary:,}")
    print(f"  imbalance ratio (single:binary): {n_single / max(n_binary, 1):.1f} : 1")
    print(f"  cadence gaps: {frac_gap:.1%} of all samples are NaN")
    return X, y


def to_masked_channels(X: np.ndarray) -> np.ndarray:
    """Turn (N, 400) magnitudes (with NaN gaps) into (N, 2, 400) input.

    channel 0 : per-curve z-scored magnitude, gaps filled with 0
    channel 1 : observed mask (1 = real measurement, 0 = cadence gap)

    The z-score uses only the observed points of each curve (nan-aware), so the
    arbitrary per-event baseline is removed and the model sees only the lensing
    *shape* -- exactly as in the Simple model, but robust to missing samples.
    The op is per-row, so applying it after the train/test split leaks nothing.
    """
    observed = ~np.isnan(X)                       # (N, 400) bool
    mask = observed.astype(np.float32)

    # nan-aware per-curve mean / std over observed points only.
    mean = np.nanmean(X, axis=1, keepdims=True)
    std = np.nanstd(X, axis=1, keepdims=True)
    # Guard flat curves and the (degenerate) all-NaN curve.
    mean = np.where(np.isnan(mean), 0.0, mean)
    std = np.where(np.isnan(std) | (std < 1e-8), 1.0, std)

    norm = (np.where(observed, X, mean) - mean) / std   # gaps -> 0 after centering
    return np.stack([norm.astype(np.float32), mask], axis=1)  # (N, 2, 400)


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class LightCurveCNN(nn.Module):
    """1D CNN for single vs. binary light-curve classification (masked input).

    Identical to the Simple model except the first conv accepts 2 input
    channels (magnitude + observed mask). Three convolutional blocks pick up
    local morphology (smooth Paczynski peak vs. sharp caustic-crossing spikes),
    followed by global average pooling and a small classifier head. Outputs a
    single logit (BCE).
    """

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=7, padding=3),   # 2 channels: mag + mask
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),                       # 400 -> 200

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),                       # 200 -> 100

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),               # global average pool -> 128
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 1),                      # single logit
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 2, 400)
        x = self.features(x)
        return self.classifier(x).squeeze(1)       # (B,)


# --------------------------------------------------------------------------- #
# Training / evaluation helpers
# --------------------------------------------------------------------------- #
def make_loader(X: np.ndarray, y: np.ndarray, shuffle: bool) -> DataLoader:
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader) -> tuple[float, np.ndarray, np.ndarray]:
    """Return (mean BCE loss, probabilities, true labels) over a loader."""
    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    losses, probs, trues = [], [], []
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        logits = model(xb)
        losses.append(criterion(logits, yb).item())
        probs.append(torch.sigmoid(logits).cpu().numpy())
        trues.append(yb.cpu().numpy())
    return float(np.mean(losses)), np.concatenate(probs), np.concatenate(trues)


def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    print(f"Device: {DEVICE}")

    csv_path = find_dataset(HERE)

    # ---- load & split ----------------------------------------------------- #
    X, y = load_dataset(csv_path)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=VAL_SIZE, stratify=y_train, random_state=SEED
    )

    # Build masked 2-channel inputs (per-row op -- no train/test leakage).
    X_train = to_masked_channels(X_train)
    X_val = to_masked_channels(X_val)
    X_test = to_masked_channels(X_test)

    print(f"  train: {len(y_train):,}  val: {len(y_val):,}  test: {len(y_test):,}")

    train_loader = make_loader(X_train, y_train, shuffle=True)
    val_loader = make_loader(X_val, y_val, shuffle=False)
    test_loader = make_loader(X_test, y_test, shuffle=False)

    # ---- model, loss, optimizer ------------------------------------------ #
    model = LightCurveCNN().to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # Weighted loss for 90/10 imbalance: weight positives (binary) by
    # n_single / n_binary so the rare class isn't ignored.
    pos_weight = torch.tensor(
        [(y_train == 0).sum() / max((y_train == 1).sum(), 1)], device=DEVICE
    )
    print(f"pos_weight (binary): {pos_weight.item():.2f}")
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    # ---- training loop with early stopping -------------------------------- #
    history = {"train_loss": [], "val_loss": [], "val_f1": [], "val_auc": []}
    best_val_f1 = -1.0
    best_state = None
    epochs_no_improve = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        train_loss = float(np.mean(epoch_losses))
        val_loss, val_probs, val_true = evaluate(model, val_loader)
        val_pred = (val_probs >= 0.5).astype(int)
        val_f1 = f1_score(val_true, val_pred, zero_division=0)
        val_auc = roc_auc_score(val_true, val_probs)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_f1"].append(val_f1)
        history["val_auc"].append(val_auc)

        print(
            f"Epoch {epoch:02d}/{EPOCHS}  "
            f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
            f"val_F1={val_f1:.4f}  val_AUC={val_auc:.4f}"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"Early stopping at epoch {epoch} (best val_F1={best_val_f1:.4f})")
                break

    # ---- restore best & pick the decision threshold ------------------------ #
    if best_state is not None:
        model.load_state_dict(best_state)

    # The model outputs a probability, not a class; a cut-off turns it into one.
    # 0.5 is only natural for a balanced problem with an unweighted loss, and this
    # is neither: pos_weight=9 deliberately pushes the probabilities upward, so
    # cutting at 0.5 corrects for the imbalance a second time and floods the binary
    # class with false positives. The cut-off is therefore chosen properly -- on the
    # VALIDATION set, never on the test set, which would leak and inflate the score.
    _, val_probs, val_true = evaluate(model, val_loader)
    grid = np.linspace(0.05, 0.95, 91)
    val_f1s = [f1_score(val_true, (val_probs >= t).astype(int), zero_division=0)
               for t in grid]
    best_threshold = float(grid[int(np.argmax(val_f1s))])
    print(f"\nDecision threshold selected on validation: {best_threshold:.2f} "
          f"(val F1={max(val_f1s):.4f}; F1 at 0.50 would be "
          f"{f1_score(val_true, (val_probs >= 0.5).astype(int), zero_division=0):.4f})")

    # ---- final test evaluation (test set touched once, at that threshold) --- #
    test_loss, test_probs, test_true = evaluate(model, test_loader)
    test_pred = (test_probs >= best_threshold).astype(int)
    test_f1 = f1_score(test_true, test_pred, zero_division=0)
    test_auc = roc_auc_score(test_true, test_probs)
    test_f1_at_half = f1_score(test_true, (test_probs >= 0.5).astype(int),
                               zero_division=0)

    print("\n" + "=" * 60)
    print("TEST SET PERFORMANCE")
    print("=" * 60)
    print(f"  threshold : {best_threshold:.2f}  (chosen on validation)")
    print(f"  loss      : {test_loss:.4f}")
    print(f"  F1        : {test_f1:.4f}   (at the naive 0.50: {test_f1_at_half:.4f})")
    print(f"  AUC       : {test_auc:.4f}   (threshold-independent)")
    print("\n" + classification_report(
        test_true, test_pred, target_names=["single", "binary"], digits=4,
        zero_division=0,
    ))

    # ---- save model + metadata -------------------------------------------- #
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "architecture": "LightCurveCNN",
            "n_points": N_POINTS,
            "in_channels": 2,
            "normalization": "per_curve_zscore_masked",
            "label_map": {"single": 0, "binary": 1},
            # Inference MUST use this, not 0.5 -- see the threshold selection above.
            "decision_threshold": best_threshold,
            "config": {
                "batch_size": BATCH_SIZE,
                "epochs": EPOCHS,
                "lr": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
                "pos_weight": float(pos_weight.item()),
                "seed": SEED,
            },
            "test_metrics": {
                "f1": test_f1,
                "auc": test_auc,
                "loss": test_loss,
                "threshold": best_threshold,
                "f1_at_0.5": test_f1_at_half,
            },
        },
        MODEL_OUT,
    )
    print(f"\nSaved model -> {MODEL_OUT.name}")

    # ---- plots ------------------------------------------------------------ #
    epochs_ran = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax1.plot(epochs_ran, history["train_loss"], label="train loss")
    ax1.plot(epochs_ran, history["val_loss"], label="val loss")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("BCE loss")
    ax1.set_title("Training / validation loss")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(epochs_ran, history["val_f1"], label="val F1", color="green")
    ax2.plot(epochs_ran, history["val_auc"], label="val AUC", color="purple")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("score")
    ax2.set_title("Validation F1 / AUC")
    ax2.legend()
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(HISTORY_PLOT, dpi=120)
    print(f"Saved history plot -> {HISTORY_PLOT.name}")

    cm = confusion_matrix(test_true, test_pred)
    fig2, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], labels=["single", "binary"])
    ax.set_yticks([0, 1], labels=["single", "binary"])
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(
        f"Confusion matrix (test)\n"
        f"F1={test_f1:.3f}  AUC={test_auc:.3f}  threshold={best_threshold:.2f}"
    )
    for i in range(2):
        for j in range(2):
            ax.text(
                j, i, f"{cm[i, j]:,}", ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black",
            )
    fig2.colorbar(im, ax=ax)
    fig2.tight_layout()
    fig2.savefig(CONFUSION_PLOT, dpi=120)
    print(f"Saved confusion matrix -> {CONFUSION_PLOT.name}")


if __name__ == "__main__":
    main()
