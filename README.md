# Microlensing Dataset Generator

A web application for generating synthetic gravitational microlensing light-curve
datasets, based on the parameter distributions and physical models from
`TDR_ROC.pdf` (Roc Rubió, "Gravitational Microlensing", pp. 8–29).

Optional OGLE-IV realistic imperfections (photometric noise + cadence gaps) can be
applied to I(t)-mode datasets, derived empirically from 3 000 real OGLE-IV EWS event
light curves.

## Project structure

```
Microlensing-1/
├── webapp/                   FastAPI web application
│   ├── app/                  Python package
│   │   ├── main.py           FastAPI routes and API endpoints
│   │   ├── dataset.py        Dataset generation orchestrator
│   │   ├── distributions.py  Parameter sampling functions
│   │   ├── lightcurves.py    Single- and binary-lens light curve physics
│   │   ├── ogle_noise.py     OGLE-IV noise model + cadence gap application
│   │   ├── plotting.py       Matplotlib plots (distributions, samples, validation)
│   │   ├── distribution_plots.py  Pre-computed KDE curves for the reference UI
│   │   ├── content.py        Static descriptive content for the UI
│   │   └── model1.py         Model 1 (Simple) inference wrapper (PyTorch)
│   ├── static/               CSS and JavaScript
│   └── templates/            Jinja2 HTML templates
├── models/                   Trained ML models (Model 1/2/3, each Simple + Real)
│   └── model_1/Simple/       Single-vs-binary CNN: training script + model_1_simple.pt
├── noise_analysis/           OGLE-IV empirical noise and cadence characterisation
│   ├── ogle_event_ids.csv    17 172 OGLE-IV EWS event IDs (years 2011–2025)
│   ├── fetch_phot.py         Downloads 3 000 random phot.dat files in parallel
│   ├── noise_model.py        Fits σ(I) noise model; outputs noise_model.npz/.png
│   ├── cadence_model.py      Characterises Δt cadence; outputs cadence_model.npz/.png
│   ├── ogle_phot_raw.npz     Pooled photometry (7.6 M obs, 89 MB — not tracked in git)
│   ├── noise_model.npz       Fitted noise model parameters + binned reference curve
│   └── cadence_model.npz     Within-season Δt distribution + visibility statistics
├── distributions/            Standalone scripts — one per parameter distribution
├── notebooks/                Exploratory Jupyter notebooks
├── requirements.txt
└── TDR_ROC.pdf               Reference document (parameter distributions, pp. 20–29)
```

## Features

- Configure total events, single-lens / binary-lens split (recommended 95 % / 5 %,
  matching the reference dataset), and time points per light curve (recommended 400).
- **Light curve format** — A(t) (dimensionless amplification) or I(t) (I-band
  magnitudes), converted via `I(t) = I_s − 2.5 log₁₀ A(t)`.
- **OGLE-IV realistic imperfections** *(I(t) mode only)* — adds photometric noise
  `σ(I) = √(σ_floor² + σ_phot0² × 10^(0.4(I−18)))` and cadence gaps (bootstrap-
  resampled from the empirical OGLE-IV Δt distribution). Unobserved time points become
  NaN. Applied immediately after magnitude conversion; fully reproducible via the
  dataset seed.
- **Distribution logic reference** — each parameter card shows its sampling formula,
  a KDE curve derived from 100 k sampled points (matching `TDR_ROC.pdf` histograms),
  and the literature citation. When OGLE noise is enabled, an additional panel shows
  the noise model curve and cadence distribution applied.
- **Visualisations** — generated parameter distributions and sample light curves
  rendered immediately after generation.
- **Validate dataset** — overlays generated histograms with literature reference shapes
  (Images 2–9, pp. 21–29) and runs KS-test goodness-of-fit checks. When OGLE noise
  was applied, two additional validation rows appear in the summary table: the σ(I)
  distribution KS test and the mean cadence coverage fraction.
- **Upload & validate** — upload an existing `.csv` or `.pkl` dataset for the same
  validation pipeline.
- **Download** — export as CSV or Pickle (`.pkl`). Filenames include `_OGLE` when
  imperfections were applied.
- **Model 1 — single vs. binary classifier** *(Simple)* — a trained 1D CNN
  (PyTorch) that predicts, per event, whether a light curve is single-lens or
  binary-lens. Classify the dataset you just generated with one click, or upload a
  *model dataset* (light curves only, exactly 400 points, no gaps). Returns a
  per-event `predictions.csv` and a download containing only the detected binary
  events with their full light curves.

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

### OGLE-IV noise pre-requisite

The imperfections feature reads `noise_analysis/noise_model.npz` and
`noise_analysis/cadence_model.npz`. These are generated by:

```bash
python noise_analysis/fetch_phot.py    # downloads ~89 MB of OGLE photometry
python noise_analysis/noise_model.py
python noise_analysis/cadence_model.py
```

Run once; the `.npz` files are reused every time the app generates a dataset.
If the files are missing, the OGLE noise option silently falls back to no-op
(a `UserWarning` is emitted at server startup).

## Physics notes

- **Single-lens** light curves use the Paczyński (1986) point-source point-lens
  amplification formula.
- **Binary-lens** light curves use the Jacobian inverse ray-shooting method.
  Large datasets with a high binary-lens fraction will take longer to generate.
- **`t_E`** is not sampled directly; it is derived as `r_E / v_perp`, where the
  Einstein radius `r_E` follows from lens mass and the three distances.
- **Binary morphology note** — generated binary curves systematically show double
  caustic crossings because sampled parameters land in the resonant topology regime
  (large diamond caustic). Real OGLE-IV binary detections are biased toward small
  planetary perturbations (the alert system favours single-lens-like curves), so they
  rarely show pronounced double peaks. The generator is physically correct; it does
  not replicate the OGLE detection selection function.

## Limits

| Parameter | Min | Max |
|---|---|---|
| Total events | 10 | 500 000 |
| Time points per curve | 50 | 1 000 |
| Binary-lens fraction | 0 % | 100 % |

Generated datasets are kept in an in-memory LRU cache (5 most recent) so Validate
and Download can reuse the same data without regenerating it.
