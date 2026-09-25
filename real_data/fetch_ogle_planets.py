"""
Ingest REAL OGLE-IV planetary microlensing events for the Model 1 (Real) CNN.

Positives-only pipeline: take the known microlensing-discovered planets from the
NASA Exoplanet Archive, keep the OGLE-hosted ones from 2011+ (the range the
OGLE-IV EWS archive covers), and turn each real light curve into a single row on
the model's fixed 400-slot tau grid -- exactly the format
``webapp/app/model1_real.py`` expects.

Why these events are "positives": every one hosts a confirmed planet, so the
ground-truth label is binary = 1. Running the Real model on them tells us how
many of the *detectable* planets it actually flags on real data (most planet
anomalies sit below the OGLE noise -- the ~0.55 AUC ceiling -- so we do NOT
expect to catch them all; we expect to catch the strong ones).

The windowing (the crux)
------------------------
A raw EWS ``phot.dat`` is the FULL time series (whole seasons of baseline), not
just the event. We do not filter the event out of the signal -- we window it
using OGLE's own published PSPL fit in ``params.dat``:
    Tmax = t0,  tau = t_E
    tau_curve = (HJD - Tmax) / tau        (dimensionless event time)
    keep |tau_curve| <= 3                  <- this IS the windowing
    bin the survivors onto the 400-slot grid tau_k = linspace(-3, 3, 400)
Empty slots stay NaN (a cadence gap); the model's observed-mask channel handles
them. Per-curve z-scoring in the model removes the baseline offset, so no flux
calibration is needed here.


Outputs (in this folder)
------------------------
    ogle_planets_dataset.csv   event_id, disc_year, t0, tE, u0, fbl, n_obs_used,
                               coverage, then t_000 ... t_399  (full record)
    ogle_planets_upload.csv    ONLY t_000 ... t_399, same row order -- ready to
                               drop into the Real model's "upload" box in the
                               webapp (row index maps back to the record file)

If PyTorch is importable it also scores the curves inline with the trained Real
model and prints a per-event table + summary (how many flagged general/strict).
"""

import io
import re
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")
requests.packages.urllib3.disable_warnings()

HERE = Path(__file__).parent
REPO = HERE.parent
EVENTS_CSV = REPO / "noise_analysis" / "ogle_event_ids.csv"
OUT_DATASET = HERE / "ogle_planets_dataset.csv"
OUT_UPLOAD = HERE / "ogle_planets_upload.csv"

EWS_BASE = "https://www.astrouw.edu.pl/ogle/ogle4/ews"
NASA_URL = (
    "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query="
    "select+pl_name,hostname,disc_year,discoverymethod+from+ps"
    "+where+default_flag=1+and+discoverymethod=%27Microlensing%27&format=csv"
)

N_POINTS = 400
TAU_MAX = 3.0                       # window half-width in Einstein times
TAU_GRID = np.linspace(-TAU_MAX, TAU_MAX, N_POINTS)
MIN_YEAR = 2011                     # EWS-IV archive coverage starts here
MAX_WORKERS = 10
TIMEOUT = 25                        # seconds per HTTP request

_HOST_RE = re.compile(r"(OGLE-(\d{4})-BLG-(\d+))")


# --------------------------------------------------------------------------- #
# 1. Which events are our positives                                            #
# --------------------------------------------------------------------------- #
def planet_event_ids() -> pd.DataFrame:
    """Known OGLE 2011+ planet events, joined to (year, field) for the URL.

    Returns a DataFrame: event_id, year, field, disc_year (deduplicated by
    event_id -- a multi-planet host is still one light curve).
    """
    print("Fetching microlensing planets from NASA Exoplanet Archive...")
    resp = requests.get(NASA_URL, verify=False, timeout=60)
    ps = pd.read_csv(io.StringIO(resp.text), comment="#")
    print(f"  {len(ps)} microlensing planets in the archive")

    rows = []
    for _, r in ps.iterrows():
        m = _HOST_RE.match(str(r["hostname"]))
        if not m:
            continue                                  # MOA / KMT / non-OGLE host
        year = int(m.group(2))
        if year < MIN_YEAR:
            continue                                  # OGLE-III, not in EWS-IV
        event_id = f"OGLE-{year}-BLG-{int(m.group(3)):04d}"
        rows.append((event_id, year, r.get("disc_year")))

    planets = (
        pd.DataFrame(rows, columns=["event_id", "year", "disc_year"])
        .drop_duplicates("event_id")
        .reset_index(drop=True)
    )
    print(f"  {len(planets)} unique OGLE {MIN_YEAR}+ planet events")

    # Join to the field ('blg' etc.) needed to build the EWS URL.
    ids = pd.read_csv(EVENTS_CSV)
    merged = planets.merge(ids[["id", "field"]], left_on="event_id",
                           right_on="id", how="inner")
    missing = len(planets) - len(merged)
    if missing:
        absent = sorted(set(planets["event_id"]) - set(merged["event_id"]))
        print(f"  {missing} not in {EVENTS_CSV.name} (skipped): {absent}")
    print(f"  {len(merged)} events fetchable from EWS-IV")
    return merged[["event_id", "year", "field", "disc_year"]].reset_index(drop=True)


# --------------------------------------------------------------------------- #
# 2. Fetch + window one event                                                  #
# --------------------------------------------------------------------------- #
def _event_dir(event_id: str, year: int, field: str) -> str:
    num = event_id.rsplit("-", 1)[-1]
    return f"{EWS_BASE}/{year}/{field.lower()}-{num}"


def _parse_params(text: str) -> dict:
    """Pull Tmax, tau, umin, fbl, I0 from a params.dat body."""
    out = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0] in ("Tmax", "tau", "umin", "fbl", "I0"):
            try:
                out[parts[0]] = float(parts[1])
            except ValueError:
                pass
    return out


def _parse_phot(text: str) -> tuple[np.ndarray, np.ndarray]:
    """HJD and I magnitude columns from a phot.dat body."""
    hjd, imag = [], []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        cols = line.split()
        if len(cols) < 3:
            continue
        try:
            hjd.append(float(cols[0]))
            imag.append(float(cols[1]))
        except ValueError:
            continue
    return np.asarray(hjd), np.asarray(imag)


def _bin_to_grid(tau: np.ndarray, imag: np.ndarray) -> np.ndarray:
    """Average magnitudes into the 400 fixed tau slots; empty slots -> NaN."""
    curve = np.full(N_POINTS, np.nan, dtype=np.float64)
    inwin = np.abs(tau) <= TAU_MAX
    if not inwin.any():
        return curve
    tau, imag = tau[inwin], imag[inwin]
    # slot index on the uniform grid tau_k = -3 + 6k/399
    slot = np.rint((tau + TAU_MAX) / (2 * TAU_MAX) * (N_POINTS - 1)).astype(int)
    slot = np.clip(slot, 0, N_POINTS - 1)
    ssum = np.bincount(slot, weights=imag, minlength=N_POINTS)
    scnt = np.bincount(slot, minlength=N_POINTS)
    seen = scnt > 0
    curve[seen] = ssum[seen] / scnt[seen]
    return curve


def fetch_one(event_id: str, year: int, field: str):
    """Return a record dict for one event, or None on failure."""
    base = _event_dir(event_id, year, field)
    try:
        pr = requests.get(f"{base}/params.dat", verify=False, timeout=TIMEOUT)
        ph = requests.get(f"{base}/phot.dat", verify=False, timeout=TIMEOUT)
        if pr.status_code != 200 or ph.status_code != 200:
            return None
        p = _parse_params(pr.text)
        if "Tmax" not in p or "tau" not in p or p["tau"] <= 0:
            return None
        hjd, imag = _parse_phot(ph.text)
        if len(hjd) < 5:
            return None

        tau_curve = (hjd - p["Tmax"]) / p["tau"]
        curve = _bin_to_grid(tau_curve, imag)
        n_used = int(np.isfinite(curve).sum())
        if n_used == 0:
            return None                               # no in-window observations

        rec = {
            "event_id": event_id,
            "disc_year": year,
            "t0": p["Tmax"],
            "tE": p["tau"],
            "u0": p.get("umin", np.nan),
            "fbl": p.get("fbl", np.nan),
            "n_obs_used": n_used,
            "coverage": n_used / N_POINTS,
        }
        for k in range(N_POINTS):
            rec[f"t_{k:03d}"] = curve[k]
        return rec
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# 3. Optional inline scoring with the trained Real model                       #
# --------------------------------------------------------------------------- #
def score_inline(df: pd.DataFrame) -> None:
    try:
        sys.path.insert(0, str(REPO / "webapp"))
        # pyrefly: ignore [missing-import]
        from app import model1_real as m1r          # noqa: E402
    except Exception as e:
        print(f"\n(Skipping inline scoring -- could not import the Real model: {e})")
        print("Upload ogle_planets_upload.csv to the Real model in the webapp instead.")
        return

    time_cols = [f"t_{k:03d}" for k in range(N_POINTS)]
    X = df[time_cols].to_numpy(dtype=np.float32)
    try:
        prob = m1r.predict_proba(X)
        gen_thr, strict_thr = m1r.thresholds()
    except m1r.ModelDatasetError as e:
        print(f"\n(Skipping inline scoring -- {e})")
        return

    gen = prob >= gen_thr
    strict = prob >= strict_thr
    print("\n--- Real model on real OGLE planets "
          f"(general>={gen_thr:.3f}, strict>={strict_thr:.3f}) ---")
    for eid, pr, cov in sorted(zip(df["event_id"], prob, df["coverage"]),
                               key=lambda t: -t[1]):
        flag = "STRICT" if pr >= strict_thr else ("general" if pr >= gen_thr else "")
        print(f"  {eid:<20} P(binary)={pr:.3f}  coverage={cov:5.1%}  {flag}")
    n = len(prob)
    print(f"\n  recall @ general = {gen.sum()}/{n} = {gen.mean():.1%}")
    print(f"  recall @ strict  = {strict.sum()}/{n} = {strict.mean():.1%}")
    print("  (most planet anomalies sit below OGLE noise -- catching the strong "
          "ones is the expected, honest result.)")


# --------------------------------------------------------------------------- #
def main() -> None:
    targets = planet_event_ids()
    if targets.empty:
        print("No target events -- nothing to do.")
        return

    print(f"\nDownloading + windowing {len(targets)} events "
          f"({MAX_WORKERS} parallel workers)...")
    records, fail = [], 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {ex.submit(fetch_one, r.event_id, int(r.year), r.field): r.event_id
                for r in targets.itertuples()}
        for i, fut in enumerate(as_completed(futs), 1):
            rec = fut.result()
            if rec is not None:
                records.append(rec)
            else:
                fail += 1
            if i % 20 == 0 or i == len(futs):
                print(f"  [{i:>3}/{len(futs)}]  ok={len(records):>3}  failed={fail:>3}")

    if not records:
        print("No events windowed successfully -- check connectivity / URL pattern.")
        return

    df = pd.DataFrame(records)
    meta_cols = ["event_id", "disc_year", "t0", "tE", "u0", "fbl",
                 "n_obs_used", "coverage"]
    time_cols = [f"t_{k:03d}" for k in range(N_POINTS)]
    df = df[meta_cols + time_cols]
    df.to_csv(OUT_DATASET, index=False)
    df[time_cols].to_csv(OUT_UPLOAD, index=False)

    print(f"\nSaved {len(df)} windowed light curves:")
    print(f"  full record -> {OUT_DATASET}")
    print(f"  upload-ready -> {OUT_UPLOAD}")
    print(f"  median coverage = {df['coverage'].median():.1%}  "
          f"(fraction of the 400 slots observed)")

    score_inline(df)


if __name__ == "__main__":
    main()
