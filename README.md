# Gravitational Microlensing Tool Hub

A web application and machine-learning pipeline for gravitational microlensing
research, developed as part of the TdR_25-27 by Roc Rubió. The project provides:

- **Synthetic dataset generation** — single-lens (Paczyński) and binary-lens
  (image-plane solution, Witt & Mao 1995) light curves sampled from empirical
  distributions (`TdR_RocRC.pdf`, pp. 8–29), with optional OGLE-IV realistic
  imperfections.
- **Parameter validation** — 14 goodness-of-fit checks against reference
  distributions.
- **Distribution logic reference** — interactive cards explaining each
  parameter's sampling formula, KDE curves and literature citations.
- **ML classification** — trained 1D CNNs and a gradient-boosted tree (GBT)
  for single-vs-binary event classification, with live in-app inference.

Optional OGLE-IV realistic imperfections — **photometric noise, cadence gaps and
blending** — can be applied to I(t)-mode datasets. Noise and cadence are derived
empirically from 3 000 real OGLE-IV EWS event light curves; blending is drawn from
the OGLE-IV event catalogue.

## Project structure

```
Microlensing-1/
├── webapp/                   FastAPI web application
│   ├── app/                  Python package
│   │   ├── main.py           FastAPI routes and API endpoints
│   │   ├── dataset.py        Dataset generation orchestrator
│   │   ├── distributions.py  Parameter sampling functions
│   │   ├── lightcurves.py    Single- and binary-lens light curve physics
│   │   ├── ogle_noise.py     OGLE-IV noise, cadence and blending application
│   │   ├── plotting.py       Matplotlib plots (distributions, samples, validation)
│   │   ├── distribution_plots.py  Pre-computed KDE curves for the reference UI
│   │   ├── content.py        Static descriptive content for the UI
│   │   ├── model1.py         Model 1 (Simple) inference wrapper (PyTorch)
│   │   └── model1_real.py    Model 1 (Real) wrapper — 5-channel input, calibration, two stages
│   ├── static/               CSS and JavaScript
│   └── templates/            Jinja2 HTML templates
├── model/                    Trained ML models
│   ├── simple/               Single-vs-binary CNN on PERFECT curves (upper bound)
│   │   ├── train_model_1_simple.py
│   │   └── model_1_simple.pt
│   └── real/                 Same task on noisy / gapped / blended curves
│       ├── CNN/              5-channel CNN — the shipped Real model
│       │   ├── train_model_1_real.py
│       │   └── model_1_real.pt
│       └── GBT/              χ² single-lens-fit feature track (feeds the CNN + tree cross-check)
│           ├── extract_features.py  Paczyński-fit residual (shared with the CNN) + features
│           ├── train_gbt.py         5-fold cross-validated GBT on those features
│           └── model_gbt.joblib     Trained GBT checkpoint
├── noise_analysis/           OGLE-IV empirical imperfection characterisation
│   ├── ogle_event_ids.csv    17 172 OGLE-IV EWS event IDs (years 2011–2025)
│   ├── fetch_phot.py         Downloads 3 000 random phot.dat files in parallel
│   ├── noise_model.py        Fits σ(I) noise model;        → noise_model.npz/.png
│   ├── cadence_model.py      Characterises Δt cadence;     → cadence_model.npz/.png
│   ├── baseline_model.py     Per-event observed baselines; → baseline_model.npz/.png
│   ├── blend_model.py        Paired (I_s, f_s) blending;   → blend_model.npz/.png
│   └── ogle_phot_raw.npz     Pooled photometry (7.6 M obs, 89 MB — not tracked in git)
├── real_data/                Real OGLE-IV planetary microlensing events
│   ├── fetch_ogle_planets.py Downloads and windows real planet light curves onto the
│   │                         model's 400-slot tau grid for positive-label evaluation
│   ├── ogle_planets_dataset.csv  Full fetched dataset (all matched planet events)
│   └── ogle_planets_upload.csv   Upload-ready subset for the Real model
├── distributions/            Standalone scripts — one per parameter distribution
├── requirements.txt
├── run.bat                   Self-bootstrapping Windows launcher
└── TdR_RocRC.pdf             Reference document (parameter distributions, pp. 20–29)
```

### Two OGLE-IV data products (they are not interchangeable)

| Product | What it is | Feeds |
|---|---|---|
| **Photometry** (`phot.dat`, EWS archive) | Every individual measurement: HJD, I magnitude, σ_I. 3 000 events → 7.6 M observations. **Measured.** | noise, cadence, observed baseline |
| **Event catalogue** (`table3.dat`) | One row per event: the *best-fit* parameters (t₀, t_E, u₀, I_s, f_s). 5 760 usable (I_s, f_s) pairs. **Fitted.** | blending |

The distinction matters: because catalogue values are *fitted*, ~8 % of them have
`f_s > 1`, which is physically impossible (it implies negative blend flux). These are
fit scatter past the boundary on nearly-unblended events, and are **kept** — discarding
them would preferentially remove the least-blended events and bias the distribution.

## Features

- Configure total events, single-lens / binary-lens split, and time points per
  light curve (for the available ML models, use 400).
- **Light curve format** — A(t) (dimensionless amplification) or I(t) (I-band
  magnitudes), converted via `I(t) = I_s − 2.5 log₁₀ A(t)`.
- **Event order** — rows are grouped (all single-lens first, then binary-lens) or,
  optionally, shuffled reproducibly from the dataset seed, so positional train/test
  splits keep both classes.
- **OGLE-IV realistic imperfections** *(I(t) mode only)* — three effects, applied in
  order after magnitude conversion and fully reproducible via the dataset seed:
  1. **Blending** — `I(t) = I_base − 2.5 log₁₀(f_s·A(t) + (1 − f_s))`, where the
     observed amplification is a flux-weighted average of the magnified source and the
     never-magnified blend. Dilutes the peaks and *changes the curve's shape*, so it
     cannot be normalised away.
  2. **Photometric noise** — `σ(I) = √(σ_floor² + σ_phot0² × 10^(0.4(I−18)))`, added as
     `N(0, σ(I)²)` per point.
  3. **Cadence gaps** — bootstrap-resampled from the empirical OGLE-IV Δt distribution;
     unobserved time points become NaN. A minimum of **5 % coverage per curve** is
     guaranteed (see Physics notes).
- **Distribution logic reference** — each parameter card shows its sampling formula,
  a KDE curve derived from 100 k sampled points (matching `TdR_RocRC.pdf` histograms),
  and the literature citation. When OGLE noise is enabled, an additional panel shows
  the noise model, the cadence distribution and the blend fraction applied.
- **Visualisations** — generated parameter distributions and sample light curves
  rendered immediately after generation. Sample curves are shown in the dataset's own
  format: A(t) datasets as amplification, I(t) datasets as magnitudes with the axis
  inverted (so the peak still points up), and OGLE datasets as photometry-like points
  whose gaps are the cadence.
- **Validate dataset** — 14 goodness-of-fit checks. Every physical parameter is
  compared against its reference distribution; when OGLE imperfections were applied,
  four more are added: the σ(I) noise level, the cadence coverage fraction, the blend
  fraction `f_s`, and an **independent baseline cross-check** (see below).
- **Upload & validate** — upload an existing `.csv` or `.pkl` dataset for the same
  validation pipeline. OGLE-mode datasets are recognised automatically (by their
  `f_s_blend` column or the NaN cadence gaps in their curves), so a downloaded dataset
  keeps its four OGLE checks when re-uploaded.
- **Download** — export as CSV or Pickle (`.pkl`). Filenames encode the request, e.g.
  `Microlensing_Dataset_1000_5pct_400pts_I_OGLE.csv` (`_A`/`_I` for the format, `_OGLE`
  when imperfections were applied).
- **Model 1 — single vs. binary classifier** — trained 1D CNNs (PyTorch) predicting,
  per event, whether a light curve is single- or binary-lens, in two variants: **Simple**
  (perfect curves, the upper bound) and **Real** (noisy / gapped / blended). Classify the
  dataset you just generated, or upload a *model dataset*. The Real model reports two
  calibrated operating points (a permissive **general** candidate list and a
  high-precision **strict** catalogue) and exports per-event predictions and a
  binaries-only download for each; a single event can also be looked up by id to see its
  curve, parameters and true label. See [Models](#models) for the numbers.

## Validation

All checks are judged on the **KS statistic** (effect size), never on the p-value.
With tens of thousands of events, any negligible difference becomes "statistically
significant" and the p-value collapses to ~0 for *any* real pair of distributions, so a
p-value threshold could never be passed no matter how good the generator is. What
matters is whether a difference is *large enough to matter* — which is what the
statistic measures.

The acceptance threshold is **sample-size-aware**: `KS < max(0.05, 1.63/√n_eff)`
(for two-sample tests, `n_eff = n₁n₂/(n₁+n₂)`). A fixed cutoff alone has the mirror
problem of the p-value: the KS statistic of a perfectly sampled dataset scales as
~1/√n, so for small datasets sampling noise exceeds 0.05 and a fixed rule would cry
wolf. The 1.63/√n term is the α = 0.01 critical value of the KS statistic, so small
datasets are judged fairly while large ones still face the 0.05 effect-size floor.

The **observed-baseline cross-check** is the most valuable of the fourteen. Every other
check compares generated data against the distribution it was *drawn from* — passing
proves the sampling is faithful, not that the model is right. This one compares a
**prediction** against data never used to build it: `I_base = I_s + 2.5 log₁₀ f_s` is
derived from the *fitted catalogue*, while the reference is measured from the *raw
photometry*. Nothing forces them to agree. They match to **0.08 mag** across the whole
distribution (catalogue-implied 18.86 vs photometry-measured 18.94), with a 0.30 mag
acceptance threshold.

## Models

Both models are 1D CNNs (~53 k parameters): three convolutional blocks (kernels
7 → 5 → 3, channels 32 → 64 → 128), concatenated global average **and** max pooling
(max keeps a localized caustic spike that average pooling would dilute), and a small
classifier head. Model selection and early stopping are on validation **AUC** — the
threshold-independent ranking metric — not F1-at-0.5.

| | Data | Test AUC | Notes |
|---|---|---|---|
| **Model 1 — Simple** | perfect I(t) curves | **0.993** | F1 ≈ 0.986 — the upper bound |
| **Model 1 — Real** | noise + cadence + blending | **0.600** | the honest, physics-limited result |

Simple is the deliberate **upper bound**: with noiseless curves a single lens is exactly
time-symmetric, so single-vs-binary is nearly separable and the score says little beyond
"the physics is right". It is a sanity check, not a headline.

**Model 1 — Real is where the science is, and its ~0.60 ceiling is largely a physical
result, not a bug.** On this realistic planet-heavy population (median q ≈ 10⁻³), roughly
**88 % of binaries have an anomaly fainter than the photometric noise** (median binary
anomaly ≈ 0.008 mag vs a fold noise ≈ 0.09 mag), and ~78 % of points are lost to cadence
gaps. Most of the remaining gap to 1.0 is information that is simply not in the curve
(data-processing inequality). The honest result is therefore a low recall at high
precision, not a Simple-like score.

**Input — 5 channels.** (0) per-curve z-scored magnitude, (1) observed mask, (2) fold
residual `R(τ) = I(τ) − I(−τ)` on co-observed points, (3) fold mask, and (4) the residual
of a **best-fit single-lens (Paczyński) model** (see below). Channels 2–3 give the binary
signature — departure from time-symmetry — but the fold is only defined where *both* τ and
−τ were observed, a small fraction of a ~78 %-gap curve. Channel 4 is the key addition:
the single-lens-fit residual uses *every* observed point, and adding it lifted the CNN from
AUC 0.557 (fold only) to **0.600**, tripling recall (see below).

**Calibrated output, two operating points.** The raw sigmoid is not a probability
(`pos_weight` inflates it), so the score is **isotonic-calibrated on validation** —
monotonic, so AUC is unchanged; only the number's *meaning* changes, making a threshold a
precision target. The result is reported at two operating points on the **same** calibrated
score (not two models — one score, two cut-offs):

| Operating point | Threshold | Test TP | Test FP | Precision | Recall |
|---|---|---|---|---|---|
| **general** (candidate list) | 0.50 | 632 | 109 | 0.853 | 0.112 |
| **strict** (clean catalogue) | auto (0.67) | 557 | 55 | 0.910 | 0.099 |

The strict threshold is selected on validation as the *lowest* cut reaching a 0.95
precision target; on the held-out test set it delivered 0.910 (the threshold is chosen from
few validation positives, so it generalises approximately). F1-maximisation is deliberately
**not** used: with this ranker and a 15 % base rate it is maximised by flagging almost
everything binary. Recall correctly *rises* with cadence coverage (0.024 → 0.184 from
< 10 % to > 40 % observed) — the signature of a real detector, not a guesser.

### The χ² single-lens-fit channel and the tree corroboration (`model/real/GBT/`)

Channel 4 comes from fitting a **single-lens (Paczyński) model to each curve and measuring
the residual**. In flux the blended single-lens model `F = a·A(u₀,τ) + b` is linear, so the
fit is a 1-D search over u₀ with a closed-form non-negative least-squares inside, weighted
by the real σ(I) noise model. `extract_features.py` computes this residual (shared with the
CNN so both see an identical signal); `train_gbt.py` trains a gradient-boosted tree on it
as an independent, cheap cross-check.

Result (full 250 k dataset, 5-fold cross-validated):

| Detector | AUC |
|---|---|
| fold residual alone (the old CNN's signal) | 0.527 |
| χ² single-lens-fit residual alone | 0.573 |
| **GBT on the full χ²-fit feature set** | **0.604 ± 0.002** |
| **5-channel CNN (the shipped Real model)** | **0.600** |

The tree and the CNN — different representations (engineered scalars vs the raw residual
curve) — **independently converge on ~0.60**, which triangulates the result. It also
revised the earlier claim of a hard 0.55 ceiling: part of that was the fold *representation*,
not pure physics. The remaining gap to 1.0 is the genuine, physics-limited ceiling.

### Real OGLE-IV planet evaluation (`real_data/`)

`fetch_ogle_planets.py` downloads real OGLE-IV planetary microlensing events from the
OGLE EWS archive, windows each light curve using the published PSPL fit parameters
(t₀, t_E), and maps it onto the model's fixed 400-slot τ grid. These events serve as
confirmed positive labels (every one hosts a discovered planet) for evaluating the Real
model on actual observational data — not synthetic curves.

## Running locally

The simplest way to start the app on Windows is the launcher script:

```bat
REM From the project root (Microlensing-1/)
run.bat
```

`run.bat` is **self-bootstrapping**. On each launch it will, only when needed:

1. create the virtual environment (`venv\`) if it is missing,
2. install / update dependencies from `requirements.txt` (the first run downloads
   PyTorch and can take a few minutes; later runs skip this and start instantly),
3. start the web app at http://127.0.0.1:8000.

Dependency installation is keyed to a hash of `requirements.txt`, so it only re-runs
when that file changes. `requirements.txt` includes `torch` (CPU build on
Windows/macOS) and `scikit-learn`, needed by the Model 1 classifier for in-app
inference and training.

Then open http://127.0.0.1:8000 in a browser.

### Manual / cross-platform

To run without the launcher (e.g. on macOS/Linux), set up the environment once and
start uvicorn with the venv Python:

```bash
python -m venv venv
venv/bin/python -m pip install -r requirements.txt        # Windows: venv\Scripts\python
venv/bin/python -m uvicorn app.main:app --reload --app-dir webapp
```

Then open http://127.0.0.1:8000 in a browser.

### OGLE-IV imperfections pre-requisite

The imperfections feature reads four `.npz` files from `noise_analysis/`. Generate them
once, in this order:

```bash
python noise_analysis/fetch_phot.py       # downloads ~89 MB of OGLE photometry
python noise_analysis/noise_model.py      # -> noise_model.npz      (sigma(I) fit)
python noise_analysis/cadence_model.py    # -> cadence_model.npz    (delta-t distribution)
python noise_analysis/baseline_model.py   # -> baseline_model.npz   (per-event baselines)
python noise_analysis/blend_model.py      # -> blend_model.npz      (paired I_s, f_s)
```

`fetch_phot.py` must run first (the next three read its output, except `blend_model.py`,
which fetches the catalogue directly). The `.npz` files are then reused on every
generation. If any is missing the app degrades gracefully with a `UserWarning` at
startup rather than crashing.

## Physics notes

- **Single-lens** light curves use the Paczyński (1986) point-source point-lens
  amplification formula.
- **Binary-lens** light curves are solved in the **image plane** (Witt & Mao 1995):
  the binary lens equation is recast as a 5th-order complex polynomial whose roots are
  the image positions, and the magnification is `A = Σ 1/|det J|` over the 3 or 5 true
  images. In the limit q → 0 this reproduces Paczyński to ~1e-7. The solve costs
  ~13 ms per 400-point event, so datasets with a high binary-lens fraction take longer
  to generate.
- **Mass ratio `q`** is `m_p / m_star`, the ratio of the two bodies' fractional masses
  `m_i = M_i/(M_p + M_star)` — equivalently `M_planet/M_star`, the same quantity the
  extraction script measures from the NASA Exoplanet Archive. It is used directly as
  the companion mass in units of the primary.
- **`t_E`** is not sampled directly; it is derived as `r_E / v_perp`, where the
  Einstein radius `r_E` follows from lens mass and the three distances.
- **Blending — `(I_s, f_s)` are drawn as a PAIR**, by bootstrap-resampling whole rows of
  the OGLE-IV catalogue. They are *correlated* in reality — brighter sources are
  measurably less blended (median `f_s` falls from 0.88 at `I_s < 17` to 0.66 at
  `I_s ≈ 20`), because a bright star dominates the light in its own aperture. Sampling
  the two independently would reproduce each marginal distribution correctly but
  generate impossible combinations. The observed baseline is then *derived*, not drawn:
  `I_base = I_s + 2.5 log₁₀ f_s`.
- **Blending is not a rescaling.** `A_obs = f_s·A + (1 − f_s)` compresses A non-linearly
  toward 1, and after the log conversion to magnitudes the curve's *proportions* change.
  Per-curve normalisation is affine and therefore cannot undo it — which is precisely
  why blending matters for the classifier: it shrinks the sharp caustic features that
  betray a binary lens, by a random amount per event.
- **Cadence guarantees ≥ 5 % coverage.** The Δt distribution has a tail reaching ~100
  days, so for a short event (`t_E` can be under 2 days, giving an ~11-day window) a
  single unlucky first draw could skip the entire window and return an all-NaN curve.
  Such a curve cannot exist in real data — OGLE only catalogues an event it observed
  often enough to *detect*. The schedule is therefore redrawn (up to 20 attempts) until
  the minimum is met, mirroring that detection requirement. The bulk of the coverage
  distribution is unaffected (mean 0.222, median 0.193 at 400 points); only the bad tail
  is clipped.
- **The observed baseline is measured per event**, as the median magnitude of each
  event's own light curve — not by pooling all observations. Pooling is biased twice
  over: heavily-monitored events would count thousands of times while sparsely-observed
  ones barely count, and the ~9 % of points that are magnified drag the distribution
  bright. The two methods differ by 0.23 mag (18.71 pooled vs **18.94** per-event).
- **Companion geometry** — with the microlensing-only semi-major-axis distribution,
  the projected separation d = a/r_E has median ≈ 1.2 and ~56 % of binary events fall
  in the caustic-rich 0.5 < d < 2 range, exactly where real microlensing-discovered
  planets sit. The generator does not replicate the OGLE detection selection function
  (no alert-pipeline bias is applied to which binaries are "found").

## Limits

| Parameter | Min | Max |
|---|---|---|
| Total events | 10 | 500 000 |
| Time points per curve | 50 | 1 000 |
| Binary-lens fraction | 0 % | 100 % |

Generated datasets are kept in an in-memory LRU cache so Validate and Download can
reuse the same data without regenerating it. Eviction is both count-based (5 most
recent) and size-aware (~2 GiB total budget); the most recent dataset is always kept,
so maximal requests still work — they just evict the older entries.
