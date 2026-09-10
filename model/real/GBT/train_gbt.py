"""
Model 1 (Real) -- gradient-boosted-tree cross-check on engineered features
==========================================================================

An independent, representation-different check on the CNN's ~0.60 AUC: does a
gradient-boosted tree, fed the chi^2 single-lens-fit features from
extract_features.py (engineered scalars), reach the same number the CNN reaches
from the raw residual curve? It does (AUC ~0.60), which triangulates the result --
the two very different models converge on the same ceiling -- and confirms that
~0.60 is the limit of what these tools can extract, not an artefact of one of them.
The GBT is a cross-check, not a deployed product; the CNN remains the main model.

Uses sklearn's HistGradientBoostingClassifier -- the same histogram gradient
boosting algorithm as LightGBM, already installed, and it ingests NaN features
natively (a curve too sparse to fit leaves chi2 = NaN; the tree splits on
missingness). The headline metric is STRATIFIED K-FOLD cross-validated AUC, so it
comes with an error bar rather than depending on a single split.

Outputs (written next to this script, mirroring the CNN folder)
---------------------------------------------------------------
    training_log.txt       full console output of the run (CV folds + importances)
    model_gbt.joblib       a fitted model (trained on an 85% split)
    confusion_matrix.png   confusion matrix on the held-out 15% test split

Run
---
    venv/Scripts/python.exe model/real/GBT/train_gbt.py
    venv/Scripts/python.exe model/real/GBT/train_gbt.py \
        --features features_subset30k.csv          # check on a subset table
"""

from __future__ import annotations

import argparse
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
# pyrefly: ignore [missing-import]
import joblib

import matplotlib

matplotlib.use("Agg")  # headless -- save figures without a display
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent

LOG_OUT = HERE / "training_log.txt"
MODEL_OUT = HERE / "model_gbt.joblib"
CM_PLOT = HERE / "confusion_matrix.png"

# Feature columns are auto-detected from the CSV (everything except the id/label
# columns), so new features added in extract_features.py are picked up without
# editing this list. fold_max is the CNN's own signal, kept as the comparison
# yardstick; chi2_dof is the headline single-lens-fit feature.
_ID_COLS = {"event_index", "event_lenses"}
FEATURES: list[str] = []                # populated by load()
CNN_AUC = 0.60          # the measured CNN AUC the cross-check is compared against

# Illustrative decision threshold for the confusion matrix ONLY. The GBT is used
# as a threshold-independent AUC cross-check, so this cut is not tuned; with
# balanced sample weights 0.5 behaves like the CNN's permissive "general" point.
CM_THRESHOLD = 0.5


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
    """Mirror stdout AND stderr into ``path`` for the duration of the block."""
    with path.open("w", encoding="utf-8") as fh:
        orig_out, orig_err = sys.stdout, sys.stderr
        sys.stdout = _Tee(orig_out, fh)
        sys.stderr = _Tee(orig_err, fh)
        try:
            yield
        finally:
            sys.stdout, sys.stderr = orig_out, orig_err


def load(features_csv: Path) -> tuple[np.ndarray, np.ndarray]:
    global FEATURES
    df = pd.read_csv(features_csv)
    FEATURES = [c for c in df.columns if c not in _ID_COLS]
    y = (df["event_lenses"].to_numpy() == 2).astype(int)
    X = df[FEATURES].to_numpy(dtype=np.float64)
    X[~np.isfinite(X)] = np.nan            # inf -> nan so the tree treats it as missing
    print(f"Loaded {len(y):,} events  |  single: {(y == 0).sum():,}  "
          f"binary: {(y == 1).sum():,}  ({y.mean():.1%} positive)  "
          f"|  {len(FEATURES)} features")
    return X, y


def make_model(y_train: np.ndarray) -> HistGradientBoostingClassifier:
    """A modest, well-regularised GBT. Class imbalance is handled with balanced
    sample weights at fit time (below); AUC is threshold-independent so this only
    affects the split gains, not the reported ranking metric."""
    return HistGradientBoostingClassifier(
        max_iter=400,
        learning_rate=0.05,
        max_leaf_nodes=31,
        min_samples_leaf=200,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.15,
        random_state=42,
    )


def balanced_weights(y: np.ndarray) -> np.ndarray:
    """sample_weight that balances the two classes (n / (2 * n_class))."""
    w = np.empty(len(y), dtype=np.float64)
    for c in (0, 1):
        w[y == c] = len(y) / (2.0 * max((y == c).sum(), 1))
    return w


def cross_validate(X: np.ndarray, y: np.ndarray, folds: int, seed: int) -> np.ndarray:
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    aucs, fold_max_aucs, chi2_aucs = [], [], []
    fold_max = X[:, FEATURES.index("fold_max")]
    chi2_dof = X[:, FEATURES.index("chi2_dof")]

    for k, (tr, va) in enumerate(skf.split(X, y), 1):
        model = make_model(y[tr])
        model.fit(X[tr], y[tr], sample_weight=balanced_weights(y[tr]))
        p = model.predict_proba(X[va])[:, 1]
        auc = roc_auc_score(y[va], p)
        aucs.append(auc)
        # Single-feature yardsticks on the SAME fold, for a like-for-like read.
        fm = fold_max[va]
        ok = np.isfinite(fm)
        fold_max_aucs.append(max(roc_auc_score(y[va][ok], fm[ok]),
                                 1 - roc_auc_score(y[va][ok], fm[ok])))
        cd = chi2_dof[va]
        ok = np.isfinite(cd)
        chi2_aucs.append(max(roc_auc_score(y[va][ok], cd[ok]),
                             1 - roc_auc_score(y[va][ok], cd[ok])))
        print(f"  fold {k}/{folds}:  GBT AUC = {auc:.4f}   "
              f"(chi2_dof alone {chi2_aucs[-1]:.4f}, fold_max alone {fold_max_aucs[-1]:.4f})")

    a = np.array(aucs)
    print("\n" + "=" * 60)
    print(f"GBT (all features)   AUC = {a.mean():.4f} +/- {a.std():.4f}")
    print(f"chi2_dof alone       AUC = {np.mean(chi2_aucs):.4f}")
    print(f"fold_max alone       AUC = {np.mean(fold_max_aucs):.4f}   (the CNN's signal)")
    print(f"CNN (raw curve)      AUC = {CNN_AUC:.3f}")
    print("=" * 60)
    # CNN_AUC is an approximate single-run figure (no error bar), so only a gap
    # clearly larger than the combined noise counts as a real difference. Anything
    # within TIE_BAND is a tie -- which is the expected, correct result here: the
    # two very different models converging on ~0.60 is what triangulates the ceiling.
    TIE_BAND = 0.01
    delta = a.mean() - CNN_AUC
    if delta > TIE_BAND:
        verdict = "ABOVE the CNN -- extra signal beyond the CNN's representation"
    elif delta < -TIE_BAND:
        verdict = "BELOW the CNN"
    else:
        verdict = ("MATCHES the CNN (within noise) -- the two representations "
                   "converge on the same ceiling")
    print(f"Verdict: GBT is {delta:+.3f} vs the CNN (~{CNN_AUC:.2f}) -- {verdict}")
    return a


def feature_importance(X: np.ndarray, y: np.ndarray, seed: int) -> None:
    """Permutation importance on a held-out split -- which features carry the signal."""
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25,
                                           stratify=y, random_state=seed)
    model = make_model(ytr)
    model.fit(Xtr, ytr, sample_weight=balanced_weights(ytr))
    imp = permutation_importance(model, Xte, yte, scoring="roc_auc",
                                 n_repeats=5, random_state=seed)
    order = np.argsort(-imp.importances_mean)
    print("\nPermutation importance (drop in AUC when the feature is shuffled)")
    for i in order:
        if imp.importances_mean[i] > 1e-4:
            print(f"  {FEATURES[i]:14s}  {imp.importances_mean[i]:.4f} "
                  f"+/- {imp.importances_std[i]:.4f}")


def finalize(X: np.ndarray, y: np.ndarray, seed: int) -> None:
    """Fit one model on an 85% split, save it, and report/plot on the 15% test.

    This produces the on-disk artefacts (model + confusion matrix) so the reported
    numbers are traceable. The held-out AUC is a single-split point estimate; the
    cross-validated AUC above is the headline. The confusion matrix uses an
    untuned, illustrative threshold (see CM_THRESHOLD).
    """
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.15,
                                          stratify=y, random_state=seed)
    model = make_model(ytr)
    model.fit(Xtr, ytr, sample_weight=balanced_weights(ytr))

    p = model.predict_proba(Xte)[:, 1]
    test_auc = roc_auc_score(yte, p)
    pred = (p >= CM_THRESHOLD).astype(int)

    print("\n" + "=" * 60)
    print(f"Held-out test (15% split)   AUC = {test_auc:.4f}")
    print(f"Confusion matrix at threshold {CM_THRESHOLD:.2f} (illustrative):")
    print(classification_report(yte, pred, target_names=["single", "binary"],
                                digits=4, zero_division=0))

    joblib.dump({"model": model, "features": FEATURES, "cm_threshold": CM_THRESHOLD,
                 "test_auc": float(test_auc)}, MODEL_OUT)
    print(f"Saved model -> {MODEL_OUT.name}")

    cm = confusion_matrix(yte, pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.5, 4.2))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], labels=["single", "binary"])
    ax.set_yticks([0, 1], labels=["single", "binary"])
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(f"GBT cross-check -- test confusion matrix\n"
                 f"AUC={test_auc:.3f}, threshold={CM_THRESHOLD:.2f} (illustrative)")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(CM_PLOT, dpi=120)
    print(f"Saved confusion matrix -> {CM_PLOT.name}")


def run(features: Path, folds: int, seed: int) -> None:
    X, y = load(features)
    cross_validate(X, y, folds, seed)
    feature_importance(X, y, seed)
    finalize(X, y, seed)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", type=Path, default=HERE / "features_real.csv")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import traceback
    with tee_output(LOG_OUT):
        try:
            run(args.features, args.folds, args.seed)
            print(f"\nSaved training log -> {LOG_OUT.name}")
        except BaseException:
            traceback.print_exc()
            sys.exit(1)


if __name__ == "__main__":
    main()
