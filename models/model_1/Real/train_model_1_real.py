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
     A CNN cannot ingest NaNs, so each curve is turned into a FOUR-channel input
     (see to_masked_channels):
         channel 0 : per-curve z-scored magnitude, gaps filled with 0
         channel 1 : observed mask (1 = real measurement, 0 = cadence gap)
         channel 2 : fold residual I(tau)-I(-tau) over co-observed points
         channel 3 : fold mask (1 where both tau and -tau were observed)
     Channels 2-3 hand the model the binary signature directly (its departure
     from single-lens time-symmetry); with only the masked magnitude the CNN
     could not learn it and stalled at chance. Even so, on this realistic
     planet-heavy population the signal is faint -- expect val AUC ~0.55, the
     measured information ceiling, not a Simple-like score.

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
Class imbalance is handled with a weighted loss (pos_weight = n_single /
n_binary, computed from the training split -- ~5.7 for an 85/15 dataset).

Training set-up (mirrors the improved Simple model)
---------------------------------------------------
  * avg + max global pooling  -- max preserves a localized caustic spike that
    average pooling alone would dilute over the whole curve.
  * model selection & early stopping on validation AUC (threshold-independent),
    not F1-at-0.50, plus a ReduceLROnPlateau schedule -- stable convergence.

Calibration and the two-stage output
------------------------------------
The raw sigmoid is NOT a probability: pos_weight deliberately inflates it. A
Platt calibrator (fitted on VALIDATION only -- never test, which would leak) maps
the logit onto true frequencies, so a calibrated 0.66 really does mean "~66% of
events scoring this are binary". Calibration is monotonic: AUC is unchanged, only
the meaning of the number.

That makes a threshold equal a precision target, and predictions are reported at
two operating points on the same calibrated score:

  general (0.5)  -- permissive candidate list. Deliberately loose: it is the
                    hand-off product, to be reviewed by a human or by a future
                    second-opinion model.
  strict         -- the clean catalogue. Selected on validation as the LOWEST
                    threshold reaching STRICT_PRECISION_TARGET, i.e. the most
                    true positives keepable without breaching that precision.

Do NOT select the threshold by maximising F1. With a near-random ranker and a 15%
base rate, F1 is provably maximised by flagging almost everything binary -- that
is exactly how an earlier run produced 28,491 false positives.

Note the cascade "strict applied to the general candidates" is, for a single
score, identical to the strict stage ({p>=0.5} AND {p>=strict} == {p>=strict}).
It is still exported as its own product so a different second-opinion model can
be dropped in later without reworking the pipeline.

Both thresholds and the calibration coefficients are stored in the checkpoint;
inference must apply the calibration and then cut.

Outputs (written next to this script)
-------------------------------------
    model_1_real.pt            trained weights + normalization stats + config
    training_history.png       loss / F1 curves over epochs
    confusion_matrix.png       confusion matrix on the held-out test set
    training_log.txt           full console output of the run

Run
---
    venv/Scripts/python.exe models/model_1/Real/train_model_1_real.py
"""

from __future__ import annotations

import gc
import sys
import time
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
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
# pyrefly: ignore [missing-import]
from torch.utils.data import DataLoader, Subset, TensorDataset

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
LOG_OUT = HERE / "training_log.txt"

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

# ---- resource limits (this machine is RAM-constrained and has crashed) ----- #
# The single biggest memory saving is NOT materialising the (N, 2, 400) masked
# array: we keep ONE float32 copy of the raw curves and build the 2-channel input
# per batch instead (see batch_inputs). These knobs cap the rest.
MAX_EVENTS = None       # cap events used (None = all); lower to e.g. 120_000 if RAM is tight
NUM_THREADS = 2         # cap torch CPU threads -> less RAM + CPU pressure
BATCH_SLEEP_S = 0.01    # brief pause between train batches to let the OS breathe
GC_EVERY = 200          # force garbage collection every N train batches

# ---- imbalance handling ---------------------------------------------------- #
# pos_weight = n_single/n_binary (~5.7) makes logit 0 an EXACT equilibrium of the
# weighted loss -- the model gets stuck on that saddle (flat loss, val AUC 0.500).
# Using sqrt of the ratio keeps a meaningful up-weight for the rare class while
# moving the equilibrium off 0.5 so the features can train. The residual
# imbalance is then handled by the validation-selected decision threshold.
POS_WEIGHT_POWER = 0.5  # pos_weight = (n_single/n_binary) ** this  (1.0 = full, 0.0 = none)

# ---- two-stage output ------------------------------------------------------ #
# Predictions are reported at TWO operating points on the SAME calibrated model:
#   general -- permissive candidate list, handed on for review (by a human or a
#              future second-opinion model). Kept deliberately loose.
#   strict  -- the clean, high-confidence catalogue.
# Because the scores are Platt-calibrated, a threshold IS a precision target:
# GENERAL_THRESHOLD = 0.5 means "more likely binary than not"; the strict cut is
# selected on validation as the LOWEST threshold reaching STRICT_PRECISION_TARGET,
# i.e. the most true positives you can keep without breaching that precision.
GENERAL_THRESHOLD = 0.5
STRICT_PRECISION_TARGET = 0.95

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(NUM_THREADS)


class _Tee:
    """Minimal file-like object that forwards writes to several streams."""

    def __init__(self, *streams) -> None:
        self._streams = streams

    def write(self, text: str) -> int:
        for s in self._streams:
            s.write(text)
            s.flush()
        return len(text)

    def flush(self) -> None:
        for s in self._streams:
            s.flush()


@contextmanager
def tee_output(path: Path):
    """Mirror stdout AND stderr into ``path`` for the duration of the block.

    stderr is included so a crash traceback (or a library warning) lands in the
    log too -- that is exactly the moment the transcript is most useful. Flushes
    on every write, so a killed run still leaves a readable partial log.
    """
    with path.open("w", encoding="utf-8") as fh:
        orig_out, orig_err = sys.stdout, sys.stderr
        sys.stdout = _Tee(orig_out, fh)
        sys.stderr = _Tee(orig_err, fh)
        try:
            yield
        finally:
            sys.stdout, sys.stderr = orig_out, orig_err


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
    label_col = "event_lenses"

    # Read the header alone to resolve the light-curve columns, then read the file
    # with an explicit float32 dtype: this halves the parse-time memory vs the
    # float64 default (the dominant RAM spike on a big OGLE export).
    header = pd.read_csv(csv_path, nrows=0)
    point_cols = [c for c in header.columns
                  if c.startswith("t_") and str(c)[2:].isdigit()]
    if len(point_cols) != N_POINTS:
        raise ValueError(
            f"Expected {N_POINTS} light-curve columns, found {len(point_cols)}"
        )
    point_cols = sorted(point_cols, key=lambda c: int(str(c)[2:]))
    dtypes = {c: np.float32 for c in point_cols}
    dtypes[label_col] = np.int16

    df = pd.read_csv(csv_path, usecols=[label_col] + point_cols, dtype=dtypes)
    X = df[point_cols].to_numpy(dtype=np.float32)       # (N, 400), NaN = cadence gap
    y = (df[label_col].to_numpy() == 2).astype(np.float32)   # 1->0 single, 2->1 binary
    del df
    gc.collect()                                        # free the big DataFrame now

    # Optional subsample (stratified) to bound the working set on tight RAM.
    if MAX_EVENTS is not None and MAX_EVENTS < len(y):
        rng = np.random.RandomState(SEED)
        keep = np.concatenate([
            rng.choice(np.where(y == 0)[0],
                       int(round(MAX_EVENTS * (y == 0).mean())), replace=False),
            rng.choice(np.where(y == 1)[0],
                       int(round(MAX_EVENTS * (y == 1).mean())), replace=False),
        ])
        rng.shuffle(keep)
        X, y = X[keep].copy(), y[keep].copy()
        gc.collect()
        print(f"  subsampled to MAX_EVENTS={MAX_EVENTS:,}")

    n_single = int((y == 0).sum())
    n_binary = int((y == 1).sum())
    frac_gap = float(np.isnan(X).mean())
    print(f"  events: {len(y):,}  |  single: {n_single:,}  binary: {n_binary:,}")
    print(f"  imbalance ratio (single:binary): {n_single / max(n_binary, 1):.1f} : 1")
    print(f"  cadence gaps: {frac_gap:.1%} of all samples are NaN")
    return X, y


IN_CHANNELS = 4


def to_masked_channels(X: np.ndarray) -> np.ndarray:
    """Turn (N, 400) magnitudes (with NaN gaps) into (N, 4, 400) input.

    channel 0 : per-curve z-scored magnitude, gaps filled with 0
    channel 1 : observed mask                (1 = measured, 0 = cadence gap)
    channel 2 : fold residual R(tau)=I(tau)-I(-tau), same scale, gaps -> 0
    channel 3 : fold mask                    (1 where BOTH tau and -tau observed)

    Channels 2-3 are the key addition for the Real model. A binary's signature IS
    its departure from the single-lens time-symmetry, but with a plain masked
    magnitude the CNN has to *learn* to compute that fold from a gappy curve and
    it failed to (val AUC stuck at 0.50, below even a one-line fold-asymmetry
    statistic). Handing it the fold residual directly lifts it to the information
    ceiling (~0.55 on this data). The residual is only defined where BOTH tau and
    -tau were observed, so channel 3 tells the model where channel 2 is real.
    It is scaled by the curve's own std (not its own) so a real anomaly keeps its
    physical size relative to the lensing peak. All ops are per row -- no leakage.
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

    # Fold residual over co-observed pairs, on the SAME per-curve scale.
    rev = X[:, ::-1]
    both = observed & observed[:, ::-1]
    resid = np.where(both, (X - rev) / std, 0.0)
    fold_mask = both.astype(np.float32)

    return np.stack([norm.astype(np.float32), mask,
                     resid.astype(np.float32), fold_mask], axis=1)  # (N, 4, 400)


def fit_platt(logits: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Fit Platt scaling on VALIDATION logits: P(binary) = sigmoid(a*logit + b).

    The raw sigmoid is NOT a probability here -- pos_weight deliberately inflates
    it, so "0.66" means nothing. Platt scaling re-maps the logit onto true
    frequencies, so a calibrated 0.66 really is "~66% of events scoring this are
    binary". It is a monotonic transform, so AUC/ranking are unchanged; only the
    MEANING of the number changes.

    Consequence we rely on below: after calibration a threshold IS a precision
    target -- cutting at 0.9 keeps events that are ~90% likely to be binary.
    """
    lr = LogisticRegression(C=1e10, solver="lbfgs")   # ~unregularised
    lr.fit(logits.reshape(-1, 1), y)
    return float(lr.coef_[0][0]), float(lr.intercept_[0])


def apply_platt(logits: np.ndarray, ab: tuple[float, float]) -> np.ndarray:
    """Map raw logits to calibrated probabilities using fitted Platt (a, b)."""
    a, b = ab
    return 1.0 / (1.0 + np.exp(-(a * np.asarray(logits, dtype=np.float64) + b)))


def select_strict_threshold(p: np.ndarray, y: np.ndarray, target: float) -> float:
    """Lowest threshold whose precision >= target -> most true positives kept.

    This is the "least false positives while conserving the most true positives"
    rule. It replaces maximising F1, which with a near-random ranker and a 15%
    base rate is provably maximised by flagging almost everything binary (that is
    exactly how the previous run produced 28,491 false positives).

    Returns 1.1 (flag nothing) if the precision target is unreachable, rather than
    silently falling back to a permissive cut.
    """
    order = np.argsort(-p)
    ys = y[order]
    tp = np.cumsum(ys)
    fp = np.cumsum(1 - ys)
    prec = tp / np.maximum(tp + fp, 1)
    ok = np.where(prec >= target)[0]
    if len(ok) == 0:
        return 1.1
    best = ok[int(np.argmax(tp[ok]))]      # among those meeting precision, most TP
    return float(p[order][best])


def report_stage(name: str, p: np.ndarray, y: np.ndarray, thr: float) -> dict:
    """Print precision/recall/FP for one stage and return the summary."""
    pred = (p >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-12)
    print(f"  {name:<22s} thr={thr:.3f}  flagged={tp+fp:>6,}  "
          f"TP={tp:>5,}  FP={fp:>6,}  precision={prec:.3f}  recall={rec:.3f}  F1={f1:.3f}")
    return {"threshold": thr, "tp": tp, "fp": fp, "precision": prec,
            "recall": rec, "f1": f1}


def report_recall_vs_coverage(
    gap_frac: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray
) -> None:
    """Binary recall as a function of how much of the curve was observed.

    The Simple model's honesty check binned recall by the true anomaly
    amplitude, but here the curves are NOISY -- max|I(tau)-I(-tau)| would measure
    the noise, not the anomaly, and the clean amplitude is not in the dataset. So
    the Real analog is recall vs. cadence coverage: it shows whether missing data
    (not just faint signal) is what the model trips on. Strongly falling recall
    on sparse curves would say the mask channel is not compensating for the gaps.
    """
    print("\nBinary recall vs. cadence coverage (test set, binaries only)")
    print("  observed fraction     n      recall")
    edges = [0.0, 0.05, 0.10, 0.20, 0.40, 1.01]
    labels = ["< 5%", "5 .. 10%", "10 .. 20%", "20 .. 40%", "> 40%"]
    coverage = 1.0 - gap_frac
    for lo, hi, lab in zip(edges[:-1], edges[1:], labels):
        sel = (y_true == 1) & (coverage >= lo) & (coverage < hi)
        if sel.sum():
            print(f"  {lab:18s} {sel.sum():5d}    {y_pred[sel].mean():.3f}")


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class LightCurveCNN(nn.Module):
    """1D CNN for single vs. binary light-curve classification (masked input).

    Two input channels (magnitude + observed mask). Three convolutional blocks
    pick up local morphology (smooth Paczynski peak vs. sharp caustic-crossing
    spikes), then concatenated global average + max pooling and a small
    classifier head. Outputs a single logit (BCE).
    """

    def __init__(self) -> None:
        super().__init__()
        # Conv trunk only -- pooling is done in forward() to take BOTH global
        # average and global max. The two intermediate pools are MAX, so a sharp
        # caustic spike survives the 400->100 downsampling.
        self.features = nn.Sequential(
            nn.Conv1d(IN_CHANNELS, 32, kernel_size=7, padding=3),  # mag, mask, fold-resid, fold-mask
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
        # alone dilutes a localized caustic spike over all 100 positions; max
        # pooling keeps the height of the sharpest deviation wherever it falls,
        # while average still captures broad, smooth perturbations. Together they
        # push the detection floor to fainter anomalies.
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
def make_loader(dataset, indices: np.ndarray, shuffle: bool) -> DataLoader:
    """Loader over a Subset of the shared raw-curve dataset.

    Subset references the one in-memory copy of the curves by index -- it does
    NOT copy the data, unlike slicing the array per split. num_workers=0 keeps a
    single process (worker processes would duplicate the dataset in RAM).
    """
    return DataLoader(
        Subset(dataset, indices.tolist()),
        batch_size=BATCH_SIZE, shuffle=shuffle, num_workers=0,
    )


def batch_inputs(raw_xb: torch.Tensor) -> torch.Tensor:
    """Build the (B, 4, 400) masked input from a batch of raw curves (B, 400).

    Done per batch rather than once over the whole dataset: materialising the
    full (N, 4, 400) array was ~1.6 GB on its own and a chief cause of the RAM
    crashes. Per batch it is a few MB.
    """
    return torch.from_numpy(to_masked_channels(raw_xb.numpy()))


@torch.no_grad()
def evaluate(
    model: nn.Module, loader: DataLoader
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Return (mean BCE loss, probabilities, true labels, raw logits).

    The logits are needed to fit the Platt calibrator: calibration must be fit on
    the model's raw logit, not on the sigmoid output.
    """
    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    losses, probs, trues, raw = [], [], [], []
    for raw_xb, yb in loader:
        xb = batch_inputs(raw_xb).to(DEVICE)
        yb = yb.to(DEVICE)
        logits = model(xb)
        losses.append(criterion(logits, yb).item())
        probs.append(torch.sigmoid(logits).cpu().numpy())
        trues.append(yb.cpu().numpy())
        raw.append(logits.cpu().numpy())
    return (float(np.mean(losses)), np.concatenate(probs),
            np.concatenate(trues), np.concatenate(raw))


def run() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    print(f"Device: {DEVICE}")

    csv_path = find_dataset(HERE)

    # ---- load & split ----------------------------------------------------- #
    X, y = load_dataset(csv_path)

    # Split by INDEX, not by copying the arrays: one shared raw-curve dataset is
    # referenced by index in each split (the masked 2-channel input is built per
    # batch in the loop). This avoids the several extra full-size array copies
    # (train_test_split + per-split masking) that drove RAM into the crash zone.
    all_idx = np.arange(len(y))
    train_idx, test_idx = train_test_split(
        all_idx, test_size=TEST_SIZE, stratify=y, random_state=SEED
    )
    train_idx, val_idx = train_test_split(
        train_idx, test_size=VAL_SIZE, stratify=y[train_idx], random_state=SEED
    )
    y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]

    # Per-curve cadence-gap fraction of the test set, for the coverage report
    # (measured on the raw curves, before the NaNs are filled per batch).
    test_gap_frac = np.isnan(X[test_idx]).mean(axis=1)

    full_ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    train_loader = make_loader(full_ds, train_idx, shuffle=True)
    val_loader = make_loader(full_ds, val_idx, shuffle=False)
    test_loader = make_loader(full_ds, test_idx, shuffle=False)

    print(f"  train: {len(y_train):,}  val: {len(y_val):,}  test: {len(y_test):,}")

    # ---- model, loss, optimizer ------------------------------------------ #
    model = LightCurveCNN().to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # Initialise the output bias to the base-rate log-odds, log(p / (1 - p)).
    # Without this the net starts at logit 0 (p = 0.5). Combined with
    # pos_weight = n_single / n_binary, logit 0 is an EXACT equilibrium of the
    # weighted loss -- the up-pull from the 15% binaries cancels the down-pull
    # from the 85% singles -- so the model sits on a saddle with a zero bias
    # gradient and never leaves it (observed: flat loss, val AUC pinned at 0.500).
    # Seeding the bias at the true base rate moves the start off that saddle so
    # gradients flow and the conv features actually train. (Standard fix for
    # imbalanced classification -- Lin et al. 2017, "Focal Loss", prior init.)
    base_rate = float((y_train == 1).mean())
    with torch.no_grad():
        model.classifier[-1].bias.fill_(float(np.log(base_rate / (1.0 - base_rate))))
    print(f"output bias initialised to base-rate log-odds: "
          f"{model.classifier[-1].bias.item():.3f}  (base rate {base_rate:.3f})")

    # Weighted loss for the class imbalance. pos_weight = (n_single/n_binary) **
    # POS_WEIGHT_POWER: the full ratio (power 1) puts the loss equilibrium exactly
    # at logit 0 -- the saddle the model got stuck on -- so we damp it with the
    # square root (power 0.5). The rare class is still up-weighted, but logit 0 is
    # no longer an equilibrium, and the decision threshold (chosen on validation)
    # mops up the remaining imbalance.
    ratio = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    pos_weight = torch.tensor([float(ratio ** POS_WEIGHT_POWER)], device=DEVICE)
    print(f"pos_weight (binary): {pos_weight.item():.2f}  "
          f"(full ratio {ratio:.2f} ** {POS_WEIGHT_POWER})")
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    # Drop the LR when val AUC plateaus so the model can settle into a finer
    # minimum instead of thrashing at a fixed step size.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=LR_FACTOR, patience=LR_PATIENCE
    )

    # ---- training loop with early stopping -------------------------------- #
    # Selection & early stopping track val AUC, not F1-at-0.50. AUC is
    # threshold-independent -- it measures how well the model RANKS binaries above
    # singles across every cut-off; the decision threshold is chosen once, later.
    history = {"train_loss": [], "val_loss": [], "val_f1": [], "val_auc": []}
    best_val_auc = -1.0
    best_state = None
    epochs_no_improve = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_losses = []
        for step, (raw_xb, yb) in enumerate(train_loader):
            xb = batch_inputs(raw_xb).to(DEVICE)
            yb = yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
            # Ease the CPU/RAM pressure on this constrained machine: a short pause
            # between batches and a periodic garbage collect keep it from spiking.
            if BATCH_SLEEP_S:
                time.sleep(BATCH_SLEEP_S)
            if GC_EVERY and step % GC_EVERY == 0:
                gc.collect()

        train_loss = float(np.mean(epoch_losses))
        val_loss, val_probs, val_true, _ = evaluate(model, val_loader)
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

    # ---- calibrate on validation ------------------------------------------ #
    # Everything downstream (both thresholds, the reported probability) uses the
    # CALIBRATED score. Fit on validation only -- never the test set, which would
    # leak and flatter the result.
    _, val_probs, val_true, val_logits = evaluate(model, val_loader)
    platt = fit_platt(val_logits, val_true)
    val_cal = apply_platt(val_logits, platt)
    print(f"\nPlatt calibration fitted on validation: "
          f"P(binary) = sigmoid({platt[0]:.4f} * logit + {platt[1]:.4f})")

    # ---- two operating points, both on the calibrated score ---------------- #
    strict_threshold = select_strict_threshold(val_cal, val_true,
                                               STRICT_PRECISION_TARGET)
    if strict_threshold > 1.0:
        print(f"  WARNING: precision target {STRICT_PRECISION_TARGET:.2f} is "
              f"unreachable on validation -- the strict stage will flag nothing.")
    print(f"Thresholds: general={GENERAL_THRESHOLD:.3f} (fixed), "
          f"strict={strict_threshold:.3f} "
          f"(lowest reaching validation precision >= {STRICT_PRECISION_TARGET:.2f})")

    # ---- final test evaluation (test set touched once) --------------------- #
    test_loss, test_probs, test_true, test_logits = evaluate(model, test_loader)
    test_cal = apply_platt(test_logits, platt)
    test_auc = roc_auc_score(test_true, test_cal)   # calibration is monotonic: AUC unchanged

    print("\n" + "=" * 78)
    print("TEST SET PERFORMANCE (two-stage)")
    print("=" * 78)
    print(f"  loss : {test_loss:.4f}    AUC : {test_auc:.4f}  (threshold-independent)")
    print("\n  stage outputs:")
    general = report_stage("general (candidates)", test_cal, test_true, GENERAL_THRESHOLD)
    strict = report_stage("strict (catalogue)", test_cal, test_true, strict_threshold)

    # The cascade view: the strict stage applied to the general stage's candidates.
    # With a single score this is exactly {general} AND {strict} == {strict}, so the
    # rows match the strict stage by construction -- it is exported as its own
    # product so a different second-opinion model can be dropped in later.
    both_pred = ((test_cal >= GENERAL_THRESHOLD) & (test_cal >= strict_threshold))
    n_general = int((test_cal >= GENERAL_THRESHOLD).sum())
    print(f"\n  cascade: general flagged {n_general:,} candidates -> "
          f"strict keeps {int(both_pred.sum()):,} of them "
          f"(identical to the strict stage, as expected for one score)")

    print("\n" + classification_report(
        test_true, (test_cal >= strict_threshold).astype(int),
        target_names=["single", "binary"], digits=4, zero_division=0,
    ))

    # ---- Real-specific diagnostic: does missing data trip the model? ------- #
    report_recall_vs_coverage(test_gap_frac, test_true,
                              (test_cal >= strict_threshold).astype(int))

    # ---- save model + metadata -------------------------------------------- #
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "architecture": "LightCurveCNN",
            "n_points": N_POINTS,
            "in_channels": IN_CHANNELS,
            "normalization": "per_curve_zscore_masked_fold",
            "label_map": {"single": 0, "binary": 1},
            # Inference MUST apply the Platt calibration to the raw logit and then
            # cut at these thresholds -- the raw sigmoid is not a probability.
            "calibration": {"type": "platt", "a": platt[0], "b": platt[1]},
            "general_threshold": GENERAL_THRESHOLD,
            "strict_threshold": strict_threshold,
            "strict_precision_target": STRICT_PRECISION_TARGET,
            # Back-compat alias: the single cut-off a naive consumer would use.
            "decision_threshold": strict_threshold,
            "config": {
                "batch_size": BATCH_SIZE,
                "epochs": EPOCHS,
                "lr": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
                "pos_weight": float(pos_weight.item()),
                "pos_weight_power": POS_WEIGHT_POWER,
                "seed": SEED,
                "max_events": MAX_EVENTS,
            },
            "test_metrics": {
                "auc": test_auc,
                "loss": test_loss,
                "general": general,
                "strict": strict,
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

    # Both stages side by side -- the general stage's false positives are the whole
    # reason the strict stage exists, so showing one without the other misleads.
    fig2, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, (name, thr, stats) in zip(axes, [
        ("general (candidates)", GENERAL_THRESHOLD, general),
        ("strict (catalogue)", strict_threshold, strict),
    ]):
        cm = confusion_matrix(test_true, (test_cal >= thr).astype(int),
                              labels=[0, 1])
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0, 1], labels=["single", "binary"])
        ax.set_yticks([0, 1], labels=["single", "binary"])
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.set_title(f"{name}\nthr={thr:.3f}  P={stats['precision']:.3f}  "
                     f"R={stats['recall']:.3f}  FP={stats['fp']:,}")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        fig2.colorbar(im, ax=ax)
    fig2.suptitle(f"Model 1 Real -- test confusion matrices (AUC={test_auc:.3f})")
    fig2.tight_layout()
    fig2.savefig(CONFUSION_PLOT, dpi=120)
    print(f"Saved confusion matrix -> {CONFUSION_PLOT.name}")


def main() -> None:
    # Mirror the whole run's console output into training_log.txt. On a crash the
    # traceback is printed INSIDE the context (before stderr is restored) so it
    # lands in the log too; we then exit non-zero without re-raising, which would
    # otherwise print the same traceback a second time to the terminal.
    import traceback

    with tee_output(LOG_OUT):
        try:
            run()
            print(f"Saved training log -> {LOG_OUT.name}")
        except BaseException:
            traceback.print_exc()
            sys.exit(1)


if __name__ == "__main__":
    main()
