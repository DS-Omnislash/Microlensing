# Microlensing Dataset Generator

A web application for generating synthetic gravitational microlensing light-curve
datasets, based on the parameter distributions and physical models from
`TdR_RocRC.pdf` (Roc Rubió, "Gravitational Microlensing", pp. 8–29).

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
│   │   └── model1.py         Model 1 (Simple) inference wrapper (PyTorch)
│   ├── static/               CSS and JavaScript
│   └── templates/            Jinja2 HTML templates
├── models/                   Trained ML models (model_1; model_2/3 are planned placeholders)
│   └── model_1/
│       ├── Simple/           Single-vs-binary CNN on PERFECT curves (upper bound)
│       └── Real/             Same task on noisy / gapped / blended curves
├── noise_analysis/           OGLE-IV empirical imperfection characterisation
│   ├── ogle_event_ids.csv    17 172 OGLE-IV EWS event IDs (years 2011–2025)
│   ├── fetch_phot.py         Downloads 3 000 random phot.dat files in parallel
│   ├── noise_model.py        Fits σ(I) noise model;        → noise_model.npz/.png
│   ├── cadence_model.py      Characterises Δt cadence;     → cadence_model.npz/.png
│   ├── baseline_model.py     Per-event observed baselines; → baseline_model.npz/.png
│   ├── blend_model.py        Paired (I_s, f_s) blending;   → blend_model.npz/.png
│   └── ogle_phot_raw.npz     Pooled photometry (7.6 M obs, 89 MB — not tracked in git)
├── distributions/            Standalone scripts — one per parameter distribution
├── data/                     Local caches (ogle_phot_cache)
├── requirements.txt
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

- Configure total events, single-lens / binary-lens split (recommended 95 % / 5 %,
  matching the reference dataset), and time points per light curve (recommended 400).
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
- **Model 1 — single vs. binary classifier** — a trained 1D CNN (PyTorch) predicting,
  per event, whether a light curve is single- or binary-lens. Classify the dataset you
  just generated with one click, or upload a *model dataset*. Returns a per-event
  `predictions.csv` and a download of only the detected binary events.

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

| | Data | F1 | AUC |
|---|---|---|---|
| **Model 1 — Simple** | perfect I(t) curves | **0.999** | 0.9999 |
| **Model 1 — Real** | noise + cadence + blending | **~0.90** | 0.992 |

Simple is the deliberate **upper bound**: it measures how separable the two classes are
under ideal conditions. The gap to Real is the quantified cost of realism — noise,
blending, and losing ~78 % of the sample points to the observing cadence.

**Decision threshold.** The Real model does *not* use 0.5. Because the loss applies
`pos_weight = 9` to correct the 9:1 class imbalance, the output probabilities are
deliberately shifted upward; cutting at 0.5 corrects for the imbalance a *second* time
and floods the binary class with false positives (precision 0.83 vs recall 0.91). The
threshold is therefore selected by maximising F1 on the **validation** set — never on
the test set, which would leak — and stored in the checkpoint as `decision_threshold`.
Measured effect: **F1 0.872 → 0.903**.

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
