"""
Model 1 (Simple) -- Single-lens vs. Binary-lens classifier
===========================================================

Trains a 1D Convolutional Neural Network to classify gravitational
microlensing light curves as single-lens (1) or binary-lens (2).

"Simple" framework: trained on PERFECT I(t) light curves -- no photometric
noise, no cadence gaps. This establishes the upper-bound performance the
later "Real" model is compared against.

Dataset
-------
CSV with 401 columns:
    event_lenses , t_000 , t_001 , ... , t_399
where event_lenses is 1 (single) or 2 (binary) and t_nnn are I-band
magnitudes sampled on tau in [-3, 3].

Labels are remapped 1 -> 0 (single), 2 -> 1 (binary) for the sigmoid output.

Class imbalance (90/10) is handled with a weighted loss (pos_weight = 9).

Outputs (written next to this script)
-------------------------------------
    model_1_simple.pt          trained weights + normalization stats + config
    training_history.png       loss / F1 curves over epochs
    confusion_matrix.png       confusion matrix on the held-out test set

Run
---
    venv/Scripts/python.exe models/model_1/Simple/train_model_1_simple.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

import matplotlib

matplotlib.use("Agg")  # headless -- save figures without a display
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
HERE = Path(__file__).resolve().parent
CSV_PATH = HERE / "Microlensing_Dataset_100000_10%_400pts_I(t)_Classification.csv"

MODEL_OUT = HERE / "model_1_simple.pt"
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


# --------------------------------------------------------------------------- #
# Data loading & preprocessing
# --------------------------------------------------------------------------- #
def load_dataset(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load the CSV and return (X, y).

    X : (N, 400) float32 light-curve magnitudes
    y : (N,)     float32 labels -- 0 = single-lens, 1 = binary-lens
    """
    print(f"Loading dataset: {csv_path.name}")
    df = pd.read_csv(csv_path)

    label_col = "event_lenses"
    point_cols = [c for c in df.columns if c.startswith("t_")]
    if len(point_cols) != N_POINTS:
        raise ValueError(
            f"Expected {N_POINTS} light-curve columns, found {len(point_cols)}"
        )

    X = df[point_cols].to_numpy(dtype=np.float32)
    raw_labels = df[label_col].to_numpy()

    # 1 (single) -> 0, 2 (binary) -> 1
    y = (raw_labels == 2).astype(np.float32)

    n_single = int((y == 0).sum())
    n_binary = int((y == 1).sum())
    print(f"  events: {len(y):,}  |  single: {n_single:,}  binary: {n_binary:,}")
    print(f"  imbalance ratio (single:binary): {n_single / max(n_binary, 1):.1f} : 1")
    return X, y


def normalize_per_curve(X: np.ndarray) -> np.ndarray:
    """Standardize each light curve independently (z-score per row).

    Magnitudes carry an arbitrary per-event baseline (source brightness I_s).
    Removing each curve's own mean/std makes the model focus on the *shape*
    of the lensing signal rather than absolute brightness. This is also
    robust to the magnitude->flux sign convention.
    """
    mean = X.mean(axis=1, keepdims=True)
    std = X.std(axis=1, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)  # guard flat curves
    return (X - mean) / std


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class LightCurveCNN(nn.Module):
    """1D CNN for single vs. binary light-curve classification.

    Three convolutional blocks pick up local morphology (smooth Paczynski
    peak vs. sharp caustic-crossing spikes), followed by global average
    pooling and a small classifier head. Outputs a single logit (BCE).
    """

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
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
        # x: (B, 400) -> (B, 1, 400)
        x = x.unsqueeze(1)
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

    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {CSV_PATH}\n"
            "Generate it from the webapp (100k, 90/10, 400 pts, I(t), no OGLE)."
        )

    # ---- load & split ----------------------------------------------------- #
    X, y = load_dataset(CSV_PATH)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=VAL_SIZE, stratify=y_train, random_state=SEED
    )

    # Normalize each curve independently (no train/test leakage -- per-row op).
    X_train = normalize_per_curve(X_train)
    X_val = normalize_per_curve(X_val)
    X_test = normalize_per_curve(X_test)

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

    # ---- restore best & final test evaluation ----------------------------- #
    if best_state is not None:
        model.load_state_dict(best_state)

    test_loss, test_probs, test_true = evaluate(model, test_loader)
    test_pred = (test_probs >= 0.5).astype(int)
    test_f1 = f1_score(test_true, test_pred, zero_division=0)
    test_auc = roc_auc_score(test_true, test_probs)

    print("\n" + "=" * 60)
    print("TEST SET PERFORMANCE")
    print("=" * 60)
    print(f"  loss : {test_loss:.4f}")
    print(f"  F1   : {test_f1:.4f}")
    print(f"  AUC  : {test_auc:.4f}")
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
            "normalization": "per_curve_zscore",
            "label_map": {"single": 0, "binary": 1},
            "config": {
                "batch_size": BATCH_SIZE,
                "epochs": EPOCHS,
                "lr": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
                "pos_weight": float(pos_weight.item()),
                "seed": SEED,
            },
            "test_metrics": {"f1": test_f1, "auc": test_auc, "loss": test_loss},
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
    ax.set_title(f"Confusion matrix (test)\nF1={test_f1:.3f}  AUC={test_auc:.3f}")
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
