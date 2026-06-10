# Microlensing Dataset Generator (FastAPI)

A small web app for generating synthetic gravitational microlensing
light-curve datasets, based on the parameter distributions and physical
models from `Dataset1generator.ipynb` and documented in `TDR_ROC.pdf`
(Roc Rubió, "Gravitational Microlensing", pp. 8-29).

## Features

- Choose the total number of events, the single-lens / binary-lens split
  (recommended 95% / 5%, matching the reference dataset), and the number of
  time points per light curve (recommended 400).
- Visualizes the generated parameter distributions, sample light curves, and
  a mass-vs-distance coverage plot.
- A "Distribution Logic Reference" section explains how each parameter is
  sampled and cites the corresponding figure/page in `TDR_ROC.pdf`.
- A "Validate Dataset" button overlays the generated histograms with the
  literature reference shapes (Images 2-9, pp. 21-29) and runs simple
  goodness-of-fit checks (medians / KS tests).
- Download the generated dataset as a CSV.

## Running locally

```bash
cd Dataset_generator
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Then open http://127.0.0.1:8000 in a browser.

## Notes

- The single-lens light curves use the Pacynski amplification formula; the
  binary-lens light curves use the Jacobian inverse ray-shooting method
  (per-event computation), so larger datasets with a high binary-lens
  fraction take longer to generate. The app limits requests to 20,000 total
  events, 50% binary fraction, and 1,000 time points per curve to keep
  generation responsive.
- Generated datasets are kept in an in-memory cache (most recent 5) so the
  "Validate" and "Download" actions can reuse the same data without
  regenerating it.
