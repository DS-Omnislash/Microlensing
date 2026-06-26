# Microlensing Dataset Generator

A web application for generating synthetic gravitational microlensing light-curve
datasets, based on the parameter distributions and physical models from
`notebooks/Dataset1generator.ipynb` and documented in `TDR_ROC.pdf`
(Roc Rubió, "Gravitational Microlensing", pp. 8–29).

## Project structure

```
Microlensing-1/
├── webapp/                   FastAPI web application
│   ├── app/                  Python package
│   │   ├── main.py           FastAPI routes and API endpoints
│   │   ├── dataset.py        Dataset generation orchestrator
│   │   ├── distributions.py  Parameter sampling functions
│   │   ├── lightcurves.py    Single- and binary-lens light curve physics
│   │   ├── plotting.py       Matplotlib plots (distributions, samples, validation)
│   │   ├── distribution_plots.py  Pre-computed KDE curves for the reference UI
│   │   └── content.py        Static descriptive content for the UI
│   ├── static/               CSS and JavaScript
│   └── templates/            Jinja2 HTML templates
├── notebooks/                Exploratory Jupyter notebooks
│   ├── Dataset1generator.ipynb         Reference dataset generation
│   └── 00_OGLEIV_parameter_evaluation.ipynb  OGLE-IV parameter analysis
├── requirements.txt
└── TDR_ROC.pdf               Reference document (parameter distributions, pp. 20–29)
```

## Features

- Configure total events, single-lens / binary-lens split (recommended 95 % / 5 %,
  matching the reference dataset), and time points per light curve (recommended 400).
- **Distribution logic reference** — each parameter card shows its sampling formula,
  a KDE curve derived from 100 k sampled points (matching `TDR_ROC.pdf` histograms),
  and the literature citation.
- **Visualisations** — generated parameter distributions and sample light curves
  rendered immediately after generation.
- **Validate dataset** — overlays generated histograms with literature reference shapes
  (Images 2–9, pp. 21–29) and runs KS-test goodness-of-fit checks.
- **Upload & validate** — upload an existing `.csv` or `.pkl` dataset for the same
  validation pipeline.
- **Download** — export the generated dataset as CSV or Pickle (`.pkl`).

## Running locally

```bash
# From the project root (Microlensing-1/)
pip install -r requirements.txt
uvicorn app.main:app --reload --app-dir webapp
```

Then open http://127.0.0.1:8000 in a browser.

## Physics notes

- **Single-lens** light curves use the Paczyński (1986) point-source point-lens
  amplification formula.
- **Binary-lens** light curves use the Jacobian inverse ray-shooting method, which
  is the computationally expensive step. Large datasets with a high binary-lens
  fraction will take longer to generate.
- **`t_E`** is not sampled directly; it is derived as `r_E / v_perp`, where the
  Einstein radius `r_E` follows from lens mass and the three distances.

## Limits

| Parameter | Min | Max |
|---|---|---|
| Total events | 10 | 500 000 |
| Time points per curve | 50 | 1 000 |
| Binary-lens fraction | 0 % | 100 % |

Generated datasets are kept in an in-memory LRU cache (5 most recent) so Validate
and Download can reuse the same data without regenerating it.