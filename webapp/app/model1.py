"""Inference wrapper for Model 1 (Simple) -- single vs. binary classifier.

Loads the trained PyTorch CNN from ``models/model_1/Simple/model_1_simple.pt``
and runs predictions on uploaded "model" datasets (light curves only).

The architecture and the per-curve normalization mirror
``models/model_1/Simple/train_model_1_simple.py`` exactly -- if that training
script changes, keep this in sync.

torch is imported lazily (only when this module is first imported, i.e. on the
first prediction request) so the webapp still starts without torch installed.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
import torch.nn as nn

# models/model_1/Simple/model_1_simple.pt  (three levels up from this file)
MODEL_PATH = (
    Path(__file__).resolve().parents[2]
    / "models" / "model_1" / "Simple" / "model_1_simple.pt"
)

N_POINTS = 400                  # the Simple model is fixed to 400-point curves
# Fallback only. The real cut-off is chosen by maximising F1 on the validation set
# during training and stored in the checkpoint as "decision_threshold"; pos_weight
# already shifts probabilities upward, so a naive 0.5 double-corrects the imbalance.
_FALLBACK_THRESHOLD = 0.5
_TIME_COL_RE = re.compile(r"^t_\d+$")


class ModelDatasetError(ValueError):
    """Raised when an uploaded dataset is not a valid 400-point model dataset."""


class LightCurveCNN(nn.Module):
    """1D CNN -- mirror of train_model_1_simple.LightCurveCNN."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=7, padding=3),   # 2 channels: mag + residual
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )
        # Concatenated global average + global max pooling (256 features): max
        # preserves a localized caustic spike that average pooling would dilute.
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 64),                    # 128 avg + 128 max
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 2, 400)
        x = self.features(x)
        x = torch.cat([self.avg_pool(x), self.max_pool(x)], dim=1)
        return self.classifier(x).squeeze(1)


_model: LightCurveCNN | None = None
_threshold: float | None = None


def _get_model() -> LightCurveCNN:
    """Lazily load and cache the trained model (loaded once per process).

    Also caches the decision threshold chosen on the validation set at training
    time; it is stored in the checkpoint and must be used instead of 0.5.
    """
    global _model, _threshold
    if _model is None:
        if not MODEL_PATH.exists():
            raise ModelDatasetError(
                "Trained model file not found on the server "
                f"({MODEL_PATH.name}). The model has not been trained yet."
            )
        checkpoint = torch.load(MODEL_PATH, map_location="cpu")
        model = LightCurveCNN()
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        _model = model
        _threshold = float(
            checkpoint.get("decision_threshold", _FALLBACK_THRESHOLD)
        )
    return _model


def _get_threshold() -> float:
    """Decision threshold from the checkpoint (model is loaded if needed)."""
    _get_model()
    return _threshold if _threshold is not None else _FALLBACK_THRESHOLD


def build_inputs(X: np.ndarray) -> np.ndarray:
    """Turn (N, 400) magnitudes into the (N, 2, 400) two-channel input.

    Mirror of train_model_1_simple.build_inputs:
        channel 0 : magnitude, centred and scaled by the curve's own std
        channel 1 : fold residual R(tau) = I(tau) - I(-tau), scaled by the SAME std

    A single-lens (Paczynski) curve is exactly symmetric in tau, so its residual
    is identically zero; a binary's anomaly is precisely what survives the fold.

    Both channels share the curve's scale so the anomaly keeps its true physical
    size. Normalizing the residual on its OWN scale would discard amplitude and
    make a 1e-7 mag rounding artefact look identical to a real caustic spike.
    """
    X64 = X.astype(np.float64)
    mean = X64.mean(axis=1, keepdims=True)
    std = X64.std(axis=1, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    curve = (X64 - mean) / std
    residual = (X64 - X64[:, ::-1]) / std
    return np.stack([curve, residual], axis=1).astype(np.float32)   # (N, 2, 400)


def _extract_lightcurves(df: pd.DataFrame) -> np.ndarray:
    """Extract the (N, 400) light-curve matrix from ``df``.

    Checks that there are exactly 400 light-curve columns, the dataset is
    non-empty, and there are no missing values. Returns the matrix ordered by
    time index. Raises ModelDatasetError otherwise.
    """
    time_cols = [c for c in df.columns if _TIME_COL_RE.match(str(c))]

    if len(time_cols) == 0:
        raise ModelDatasetError(
            "No light-curve columns (t_000 ... t_399) were found in this dataset."
        )

    if len(time_cols) != N_POINTS:
        raise ModelDatasetError(
            f"The Simple model only accepts light curves with exactly "
            f"{N_POINTS} points. This dataset has {len(time_cols)} points "
            "per curve."
        )

    if len(df) == 0:
        raise ModelDatasetError("The dataset is empty (no events / rows).")

    # Order columns by their numeric time index (t_000, t_001, ...).
    time_cols_sorted = sorted(time_cols, key=lambda c: int(str(c)[2:]))
    X = df[time_cols_sorted].to_numpy(dtype=np.float32)

    if np.isnan(X).any():
        raise ModelDatasetError(
            "The dataset contains missing values (NaN), i.e. cadence gaps. "
            "The Simple model only accepts clean, complete light curves. "
            "A model trained for real (noisy / gapped) data is needed for this."
        )

    return X


def validate_model_dataset(df: pd.DataFrame) -> np.ndarray:
    """Validate that ``df`` is a clean 400-point "model" dataset (uploads).

    A valid uploaded model dataset contains ONLY light-curve columns
    (t_000 ... t_399), exactly 400 of them, with no missing values.

    Returns the (N, 400) float32 light-curve matrix, ordered by time index.
    Raises ModelDatasetError (with a user-facing message) otherwise.
    """
    other_cols = [c for c in df.columns if not _TIME_COL_RE.match(str(c))]
    if other_cols:
        preview = ", ".join(str(c) for c in other_cols[:6])
        if len(other_cols) > 6:
            preview += ", ..."
        raise ModelDatasetError(
            "This must be a model dataset containing only light-curve columns "
            f"(t_000 ... t_399). Found {len(other_cols)} other column(s): "
            f"{preview}. Remove every non-light-curve column and try again."
        )
    return _extract_lightcurves(df)


@torch.no_grad()
def predict(X: np.ndarray) -> np.ndarray:
    """Return per-event binary-class probabilities for a validated matrix."""
    model = _get_model()
    Xn = build_inputs(X)
    probs = []
    for start in range(0, len(Xn), 1024):
        batch = torch.from_numpy(Xn[start : start + 1024])
        logits = model(batch)
        probs.append(torch.sigmoid(logits).numpy())
    return np.concatenate(probs)


def _summarize(X: np.ndarray) -> dict:
    """Run the model on a validated matrix and summarize counts."""
    prob_binary = predict(X)
    pred = (prob_binary >= _get_threshold()).astype(int)

    n_total = int(len(pred))
    n_binary = int((pred == 1).sum())
    n_single = n_total - n_binary

    return {
        "pred": pred,
        "prob_binary": prob_binary,
        "n_total": n_total,
        "n_single": n_single,
        "n_binary": n_binary,
    }


def classify_dataframe(df: pd.DataFrame) -> dict:
    """Validate an uploaded model dataset, run the model, summarize results.

    The uploaded file must contain ONLY light-curve columns. Raises
    ModelDatasetError if the dataset does not meet the requirements.
    """
    return _summarize(validate_model_dataset(df))


def classify_generated(df: pd.DataFrame) -> dict:
    """Run the model on a dataset generated in-app.

    Unlike uploads, the generated dataframe may carry parameter columns
    alongside the light curves; the light curves are extracted automatically.
    The 400-point and no-NaN requirements still apply. Raises
    ModelDatasetError otherwise.
    """
    return _summarize(_extract_lightcurves(df))
