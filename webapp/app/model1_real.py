"""Inference wrapper for Model 1 (Real) -- single vs. binary on OGLE-like curves.

Loads the trained CNN from ``models/model_1/Real/model_1_real.pt`` and scores
uploaded or in-app "model" datasets. Unlike the Simple wrapper this one ACCEPTS
cadence gaps (NaN): the Real model is trained on noisy, gapped, blended curves
and takes an observed-mask channel. Clean/complete curves are accepted too (they
simply have a fully-observed mask).

The preprocessing and architecture mirror
``models/model_1/Real/train_model_1_real.py`` exactly -- if that script changes,
keep this in sync.

Two-stage output
----------------
Scores are Platt-calibrated (coefficients stored in the checkpoint), so a
threshold IS a precision target and the reported probability is a real one:
    general -- permissive candidate list, for review/hand-off
    strict  -- clean, high-confidence catalogue
Both thresholds come from the checkpoint; do not hard-code them.

torch is imported when this module is first imported (i.e. on the first Real
prediction request) so the webapp still starts without torch installed.
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

# models/model_1/Real/model_1_real.pt  (three levels up from this file)
MODEL_PATH = (
    Path(__file__).resolve().parents[2]
    / "models" / "model_1" / "Real" / "model_1_real.pt"
)

N_POINTS = 400
IN_CHANNELS = 4
_TIME_COL_RE = re.compile(r"^t_\d+$")

# Fallbacks only if a checkpoint predates the two-stage fields.
_FALLBACK_GENERAL = 0.5
_FALLBACK_STRICT = 0.9


class ModelDatasetError(ValueError):
    """Raised when an uploaded dataset is not a valid 400-point model dataset."""


class LightCurveCNN(nn.Module):
    """1D CNN -- mirror of train_model_1_real.LightCurveCNN (4-channel input)."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(IN_CHANNELS, 32, kernel_size=7, padding=3),
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
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.cat([self.avg_pool(x), self.max_pool(x)], dim=1)
        return self.classifier(x).squeeze(1)


_model: LightCurveCNN | None = None
_calibration: tuple[float, float] | None = None
_general_threshold: float = _FALLBACK_GENERAL
_strict_threshold: float = _FALLBACK_STRICT


def _get_model() -> LightCurveCNN:
    """Lazily load and cache the trained model + calibration + thresholds."""
    global _model, _calibration, _general_threshold, _strict_threshold
    if _model is None:
        if not MODEL_PATH.exists():
            raise ModelDatasetError(
                "Trained Real model file not found on the server "
                f"({MODEL_PATH.name}). The model has not been trained yet."
            )
        checkpoint = torch.load(MODEL_PATH, map_location="cpu")
        model = LightCurveCNN()
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        # A checkpoint from before the calibrated two-stage design cannot be
        # served here. Its raw sigmoid is not a probability, and its single
        # F1-maximised threshold is the degenerate "flag almost everything"
        # cut-off -- reusing it silently produces a nonsensical two-stage result
        # (strict looser than general, so strict is not a subset of general).
        # Fail loudly instead of quietly reporting garbage.
        cal = checkpoint.get("calibration")
        if not (cal and cal.get("type") in ("isotonic", "platt")):
            raise ModelDatasetError(
                "The trained Real model on the server predates the calibrated "
                "two-stage design (no calibration stored in the checkpoint). "
                "Retrain with the current models/model_1/Real/train_model_1_real.py "
                "to produce a checkpoint with calibration and both thresholds."
            )
        _calibration = cal
        _general_threshold = float(
            checkpoint.get("general_threshold", _FALLBACK_GENERAL))
        _strict_threshold = float(
            checkpoint.get("strict_threshold",
                           checkpoint.get("decision_threshold", _FALLBACK_STRICT)))
        if _strict_threshold < _general_threshold:
            raise ModelDatasetError(
                f"Invalid checkpoint: the strict threshold ({_strict_threshold:.3f}) "
                f"is below the general one ({_general_threshold:.3f}), so the strict "
                "catalogue would not be a subset of the general candidates. Retrain "
                "with the current training script."
            )
        _model = model
    return _model


def thresholds() -> tuple[float, float]:
    """(general, strict) decision thresholds from the checkpoint."""
    _get_model()
    return _general_threshold, _strict_threshold


def is_calibrated() -> bool:
    _get_model()
    return _calibration is not None


def to_masked_channels(X: np.ndarray) -> np.ndarray:
    """(N, 400) magnitudes (NaN = gap) -> (N, 4, 400) input.

    Mirror of train_model_1_real.to_masked_channels:
        0 magnitude z-scored on observed points, gaps -> 0
        1 observed mask
        2 fold residual I(tau)-I(-tau) on co-observed points, same scale
        3 fold mask (both tau and -tau observed)
    """
    observed = ~np.isnan(X)
    mask = observed.astype(np.float32)

    mean = np.nanmean(X, axis=1, keepdims=True)
    std = np.nanstd(X, axis=1, keepdims=True)
    mean = np.where(np.isnan(mean), 0.0, mean)
    std = np.where(np.isnan(std) | (std < 1e-8), 1.0, std)

    norm = (np.where(observed, X, mean) - mean) / std

    rev = X[:, ::-1]
    both = observed & observed[:, ::-1]
    resid = np.where(both, (X - rev) / std, 0.0)
    fold_mask = both.astype(np.float32)

    return np.stack([norm.astype(np.float32), mask,
                     resid.astype(np.float32), fold_mask], axis=1)


def _extract_lightcurves(df: pd.DataFrame) -> np.ndarray:
    """Extract the (N, 400) curve matrix. NaNs are allowed (cadence gaps)."""
    time_cols = [c for c in df.columns if _TIME_COL_RE.match(str(c))]

    if len(time_cols) == 0:
        raise ModelDatasetError(
            "No light-curve columns (t_000 ... t_399) were found in this dataset."
        )
    if len(time_cols) != N_POINTS:
        raise ModelDatasetError(
            f"The Real model only accepts light curves with exactly {N_POINTS} "
            f"points. This dataset has {len(time_cols)} points per curve."
        )
    if len(df) == 0:
        raise ModelDatasetError("The dataset is empty (no events / rows).")

    time_cols_sorted = sorted(time_cols, key=lambda c: int(str(c)[2:]))
    X = df[time_cols_sorted].to_numpy(dtype=np.float32)

    # Unlike the Simple model, NaN gaps are expected and fine. Only a curve with
    # NO observed point at all is unusable.
    all_nan = np.isnan(X).all(axis=1)
    if all_nan.any():
        raise ModelDatasetError(
            f"{int(all_nan.sum())} light curve(s) have no observed points at all "
            "(entirely empty). Remove them and try again."
        )
    return X


def validate_model_dataset(df: pd.DataFrame) -> np.ndarray:
    """Validate an uploaded Real "model" dataset (light-curve columns only)."""
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


def _apply_calibration(logits: np.ndarray) -> np.ndarray:
    """Mirror of train_model_1_real.apply_calibrator."""
    cal = _calibration
    if cal["type"] == "isotonic":
        # np.interp reproduces IsotonicRegression.predict (clipped at the ends).
        return np.interp(logits, np.asarray(cal["x"]), np.asarray(cal["y"]))
    return 1.0 / (1.0 + np.exp(-(cal["a"] * logits + cal["b"])))


@torch.no_grad()
def predict_proba(X: np.ndarray) -> np.ndarray:
    """Calibrated P(binary) per event.

    The raw sigmoid is NOT a probability (pos_weight inflates it); the calibrator
    fitted at training time maps the logit onto true frequencies.
    """
    model = _get_model()
    probs = []
    for start in range(0, len(X), 1024):
        batch = to_masked_channels(X[start:start + 1024])
        logits = model(torch.from_numpy(batch)).numpy().astype(np.float64)
        probs.append(_apply_calibration(logits))
    return np.concatenate(probs)


def _summarize(X: np.ndarray) -> dict:
    """Score a validated matrix at both operating points."""
    prob = predict_proba(X)
    general_thr, strict_thr = thresholds()
    general_pred = (prob >= general_thr).astype(int)
    strict_pred = (prob >= strict_thr).astype(int)

    return {
        "prob_binary": prob,
        "general_pred": general_pred,
        "strict_pred": strict_pred,
        "general_threshold": general_thr,
        "strict_threshold": strict_thr,
        "calibrated": is_calibrated(),
        "n_total": int(len(prob)),
        "n_general_binary": int(general_pred.sum()),
        "n_strict_binary": int(strict_pred.sum()),
    }


def classify_dataframe(df: pd.DataFrame) -> dict:
    """Validate an uploaded Real model dataset and score it at both stages."""
    return _summarize(validate_model_dataset(df))


def classify_generated(df: pd.DataFrame) -> dict:
    """Score an in-app generated dataset (parameter columns are ignored)."""
    return _summarize(_extract_lightcurves(df))
