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

Class imbalance (85/15) is handled with a weighted loss (pos_weight = n_single
/ n_binary, computed from the training split).

Input representation -- the symmetry residual
---------------------------------------------
A single-lens (Paczynski) curve is EXACTLY symmetric in tau about the peak:
u(tau) = sqrt(u0^2 + tau^2) is even, so I(tau) = I(-tau). A binary lens breaks
that symmetry -- the anomaly IS the departure from Paczynski symmetry. The
network is therefore given two channels:

    channel 0 : per-curve z-scored magnitude          (the light curve)
    channel 1 : per-curve z-scored fold residual      R(tau) = I(tau) - I(-tau)

Channel 1 matters because most binaries in a realistic mass-ratio population
(median q ~ 1e-3) perturb the curve by only a few millimagnitudes on top of a
~1.7 mag lensing peak, and global average pooling discards the positional
information that asymmetry lives in. Folding the curve makes that anomaly the
signal instead of a rounding-level wiggle on a large peak.

BOTH channels are divided by the SAME scale (the curve's std). Do not normalize
the residual on its own scale: that discards amplitude, so a 1e-7 mag artefact
and a 0.5 mag caustic spike both become an O(1) pattern. Since a noiseless
Paczynski curve is symmetric to the last bit, that turns channel 1 into an exact
"is this curve symmetric?" giveaway and the model scores a meaningless F1 = 1.000
off floating-point dust -- a zero-parameter script does just as well. Sharing the
curve's scale keeps every anomaly at its true physical size, so the network can
only detect what is genuinely there.

Decision threshold
------------------
pos_weight already compensates the class imbalance by shifting probabilities
upward, so cutting at 0.50 would correct for the imbalance a second time and
flood the binary class with false positives. The threshold is therefore selected
by maximising F1 on the VALIDATION set -- never on the test set, which would leak
and inflate the result -- and the test set is then evaluated once at that
threshold. It is stored in the checkpoint as ``decision_threshold`` and inference
must use it.

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

import sys
from contextlib import contextmanager
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
CSV_PATH = HERE / "Microlensing_Dataset_100000_15pct_400pts_I_Classification.csv"

MODEL_OUT = HERE / "model_1_simple.pt"
HISTORY_PLOT = HERE / "training_history.png"
CONFUSION_PLOT = HERE / "confusion_matrix.png"
LOG_PATH = HERE / "training_log.txt"    # full console transcript of the run

N_POINTS = 400          # light-curve length
BATCH_SIZE = 256
EPOCHS = 60             # upper bound; early stopping usually ends it sooner
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
LR_FACTOR = 0.5         # multiply LR by this when val AUC plateaus
LR_PATIENCE = 4         # epochs without val-AUC gain before dropping LR
TEST_SIZE = 0.15        # held-out test fraction
VAL_SIZE = 0.15         # validation fraction (of the remaining train pool)
PATIENCE = 12           # early-stopping patience (epochs without val-AUC gain)
SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------- #
# Console transcript
# --------------------------------------------------------------------------- #
class _Tee:
    """A stdout proxy that writes to the terminal AND a log file at once.

    Every print() during the run therefore lands in training_log.txt as well,
    so the full details (metrics, threshold, detection-efficiency table) can be
    reviewed later without re-running. Flushes on every write so a killed run
    still leaves a readable partial log.
    """

    def __init__(self, stream, file_handle) -> None:
        self._stream = stream
        self._file = file_handle

    def write(self, data: str) -> int:
        self._stream.write(data)
        self._file.write(data)
        self._stream.flush()
        self._file.flush()
        return len(data)

    def flush(self) -> None:
        self._stream.flush()
        self._file.flush()


@contextmanager
def tee_output(log_path: Path):
    """Mirror stdout AND stderr into ``log_path`` for the duration.

    stderr is included so a crash traceback (or a library warning) lands in the
    log too -- that is exactly the moment the transcript is most useful.
    """
    with open(log_path, "w", encoding="utf-8") as fh:
        orig_out, orig_err = sys.stdout, sys.stderr
        sys.stdout = _Tee(orig_out, fh)
        sys.stderr = _Tee(orig_err, fh)
        try:
            yield
        finally:
            sys.stdout, sys.stderr = orig_out, orig_err


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
    print(f"  events: {len(y):,}  |  single: {n_single:,}  binary: {n_binary:,}")
    print(f"  imbalance ratio (single:binary): {n_single / max(n_binary, 1):.1f} : 1")
    return X, y


def build_inputs(X: np.ndarray) -> np.ndarray:
    """Turn (N, 400) magnitudes into the (N, 2, 400) two-channel input.

    channel 0 : magnitude, centred and scaled by the curve's own std
    channel 1 : fold residual R(tau) = I(tau) - I(-tau), scaled by the SAME std

    Centring removes each event's arbitrary baseline (source brightness I_s) so
    the model sees only the lensing *shape*; the shared scale then keeps the
    anomaly's size relative to the lensing peak intact (see the comments below).

    The tau grid is symmetric about 0, so reversing a row folds it about the
    peak; for a single lens I(tau) == I(-tau) exactly and the residual is zero.
    Both ops are per-row, so applying them after the train/test split leaks
    nothing.
    """
    # The residual is scaled by the CURVE's std, NOT its own. This is the whole
    # ballgame. Dividing the residual by its own std would discard amplitude --
    # a 1e-7 mag rounding artefact and a 0.5 mag caustic spike would both come out
    # as an O(1) pattern, handing the network an exact "is it symmetric?" bit and a
    # meaningless F1 = 1.000 (a zero-parameter script scores the same). Sharing the
    # curve's scale keeps the anomaly at its true physical size relative to the
    # lensing peak, so the model can only detect what is actually there: a 1e-6 mag
    # artefact stays ~1e-5 input units while a real caustic spike is ~0.7.
    # (BatchNorm does not undo this: it rescales each conv channel across the BATCH,
    # one global factor, so relative amplitude between samples survives -- unlike a
    # per-sample z-score.)
    X64 = X.astype(np.float64)
    mean = X64.mean(axis=1, keepdims=True)
    std = X64.std(axis=1, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    curve = (X64 - mean) / std
    residual = (X64 - X64[:, ::-1]) / std

    return np.stack([curve, residual], axis=1).astype(np.float32)   # (N, 2, 400)


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class LightCurveCNN(nn.Module):
    """1D CNN for single vs. binary light-curve classification.

    Three convolutional blocks pick up local morphology (smooth Paczynski
    peak vs. sharp caustic-crossing spikes), followed by global average
    pooling and a small classifier head. Outputs a single logit (BCE).

    The first conv takes 2 channels: the z-scored light curve and the z-scored
    fold residual R(tau) = I(tau) - I(-tau) (see build_inputs).
    """

    def __init__(self) -> None:
        super().__init__()
        # Conv trunk only -- pooling is done in forward() so we can take BOTH the
        # global average AND the global max. The two intermediate pools are MAX
        # (not avg), so a sharp caustic spike survives the 400->100 downsampling.
        self.features = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=7, padding=3),   # 2 channels: mag + residual
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
        )
        # Global average + global max, concatenated -> 256 features. Average pool
        # alone (the previous head) diluted a localized anomaly over all 100
        # positions -- a few-point caustic spike contributed ~1/100 of the pooled
        # value and was effectively erased, which is why faint (1e-4..1e-3 mag)
        # anomalies were missed. Max pooling keeps the height of the sharpest
        # deviation regardless of WHERE on the curve it falls; average pooling
        # still captures broad, smooth perturbations. Together they push the
        # detection floor to fainter anomalies. (This does NOT revive the
        # float-dust shortcut: a single lens has an exactly-zero residual channel,
        # so its features are position-constant and max == avg; batch-norm scales
        # the residual channels by the batch std set by real O(0.1) anomalies, so a
        # ~1e-7 mag numerical wiggle stays ~1e-6 and cannot be thresholded apart
        # from a true single lens without also firing on it. Verify via the
        # detection-efficiency report: the < 1e-5 mag bin must stay near recall 0.)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 64),                    # 128 avg + 128 max
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 1),                      # single logit
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 2, 400)
        x = self.features(x)                       # (B, 128, 100)
        x = torch.cat([self.avg_pool(x), self.max_pool(x)], dim=1)  # (B, 256, 1)
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


def anomaly_amplitude(X: np.ndarray) -> np.ndarray:
    """Max |I(tau) - I(-tau)| per curve, in magnitudes -- the true anomaly size.

    Zero for a single lens (Paczynski is exactly symmetric in tau), so for a
    binary this is the physical size of the deviation the model has to find.
    """
    X64 = X.astype(np.float64)
    return np.abs(X64 - X64[:, ::-1]).max(axis=1)


def report_detection_efficiency(
    amp: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray
) -> None:
    """Recall vs. the PHYSICAL size of the anomaly -- the honesty check.

    A model doing real physics detects big anomalies and misses tiny ones, so
    recall must RISE with amplitude. Flat recall near 1.0 in the smallest bin is
    the tell-tale of a shortcut: nothing physical is detectable at 1e-6 mag, so a
    model that "finds" those is keying on the exact-symmetry bit (a noiseless
    Paczynski curve is symmetric to the last bit, so ANY nonzero residual is a
    perfect label) rather than on morphology. See also the zero-parameter baseline.
    """
    print("\nDetection efficiency vs. anomaly amplitude (test set, binaries only)")
    print("  anomaly [mag]        n      recall")
    edges = [0.0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, np.inf]
    labels = ["< 1e-5 (numerical)", "1e-5 .. 1e-4", "1e-4 .. 1e-3",
              "1e-3 .. 1e-2", "1e-2 .. 0.1", "> 0.1 (obvious)"]
    for lo, hi, lab in zip(edges[:-1], edges[1:], labels):
        sel = (y_true == 1) & (amp >= lo) & (amp < hi)
        if sel.sum():
            print(f"  {lab:20s} {sel.sum():5d}    {y_pred[sel].mean():.3f}")

    # A rule with no weights: "the curve is not perfectly symmetric" -> binary.
    # On noiseless data this is EXACTLY separable, so it scores ~1.0. If the model
    # matches it, the model has learned nothing a two-line script cannot do.
    trivial = (amp > 0).astype(int)
    print(f"\n  Zero-parameter baseline ('residual != 0' -> binary): "
          f"F1 = {f1_score(y_true, trivial, zero_division=0):.4f}")
    print(f"  Model:                                               "
          f"F1 = {f1_score(y_true, y_pred, zero_division=0):.4f}")
    print("  (The baseline is the noiseless ceiling and is physically empty -- it")
    print("   fires on 1e-7 mag rounding dust. The model should NOT match it.)")


def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    print(f"Device: {DEVICE}")

    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {CSV_PATH}\n"
            "Generate it from the webapp (100k, 85/15, 400 pts, I(t), no OGLE)."
        )

    # ---- load & split ----------------------------------------------------- #
    X, y = load_dataset(CSV_PATH)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=VAL_SIZE, stratify=y_train, random_state=SEED
    )

    # True anomaly size of each test curve, kept for the detection-efficiency
    # report below (measured on the RAW magnitudes, before any normalization).
    test_amplitude = anomaly_amplitude(X_test)

    # Build the 2-channel input: curve + fold residual, on a shared scale.
    # Per-row ops, so doing this after the split leaks nothing.
    X_train = build_inputs(X_train)
    X_val = build_inputs(X_val)
    X_test = build_inputs(X_test)

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
    # Drop the LR when val AUC stops improving so the model can settle into a
    # finer minimum and crack the fainter anomalies instead of thrashing at a
    # fixed step size (the previous run's val metric oscillated badly).
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=LR_FACTOR, patience=LR_PATIENCE
    )

    # ---- training loop with early stopping -------------------------------- #
    # Selection & early stopping track val AUC, not F1-at-0.50. AUC is
    # threshold-independent -- it measures how well the model RANKS binaries above
    # singles across every cut-off, which is exactly "how faint an anomaly can it
    # still separate". F1-at-0.50 was a noisy target (pos_weight shifts the
    # probabilities, so a fixed 0.50 cut jumps around epoch to epoch); the actual
    # decision threshold is chosen once, later, on the validation set.
    history = {"train_loss": [], "val_loss": [], "val_f1": [], "val_auc": []}
    best_val_auc = -1.0
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
        scheduler.step(val_auc)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_f1"].append(val_f1)
        history["val_auc"].append(val_auc)

        print(
            f"Epoch {epoch:02d}/{EPOCHS}  "
            f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
            f"val_F1={val_f1:.4f}  val_AUC={val_auc:.4f}  "
            f"lr={optimizer.param_groups[0]['lr']:.1e}"
        )

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"Early stopping at epoch {epoch} (best val_AUC={best_val_auc:.4f})")
                break

    # ---- restore best & pick the decision threshold ------------------------ #
    if best_state is not None:
        model.load_state_dict(best_state)

    # pos_weight already shifts probabilities upward to compensate the class
    # imbalance; cutting at 0.50 would correct for it twice and flood the binary
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

    # ---- honesty check ----------------------------------------------------- #
    report_detection_efficiency(test_amplitude, test_true, test_pred)

    # ---- save model + metadata -------------------------------------------- #
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "architecture": "LightCurveCNN",
            "n_points": N_POINTS,
            "normalization": "per_curve_zscore_fold_residual",
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
    print(f"Saved run transcript -> {LOG_PATH.name}")


if __name__ == "__main__":
    # Mirror the whole run's console output into training_log.txt. On a crash the
    # traceback is printed INSIDE the context (before stderr is restored) so it
    # lands in the log too; we then exit non-zero without re-raising, which would
    # otherwise print the same traceback a second time to the terminal.
    import traceback

    with tee_output(LOG_PATH):
        try:
            main()
        except BaseException:
            traceback.print_exc()
            sys.exit(1)
