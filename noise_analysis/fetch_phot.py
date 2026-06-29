"""
Download OGLE-IV EWS photometry for cadence and noise analysis.

Selects 3000 random events from ogle_event_ids.csv, fetches their .phot.dat
files from the Warsaw Observatory EWS archive, and saves the pooled photometry
to ogle_phot_raw.npz for downstream analysis (cadence, noise modelling).

Each .phot.dat file is expected to contain whitespace-separated rows:
    HJD-2450000    I_mag    sigma_I    [optional extra columns]
Lines starting with '#' are skipped.

Output — ogle_phot_raw.npz:
    hjd          float64 (N_obs,)   heliocentric JD - 2450000 for every observation
    imag         float64 (N_obs,)   I-band magnitude
    sigma        float64 (N_obs,)   photometric uncertainty in mag
    event_index  int32   (N_obs,)   which event each observation belongs to
    dt           float64 (N_dt,)    consecutive Δt within each event [days]
    n_events     int32   (1,)       number of successfully downloaded events
"""

import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")
requests.packages.urllib3.disable_warnings()

HERE = Path(__file__).parent
EVENTS_CSV = HERE / "ogle_event_ids.csv"
OUTPUT_NPZ = HERE / "ogle_phot_raw.npz"
OUTPUT_IDS = HERE / "ogle_phot_fetched_ids.csv"

EWS_BASE    = "https://www.astrouw.edu.pl/ogle/ogle4/ews"
N_TARGET    = 3000
MAX_WORKERS = 15
TIMEOUT     = 20   # seconds per HTTP request
SEED        = 42


def _build_url(event_id: str, year: str, field: str) -> str:
    # "OGLE-2011-BLG-0002" -> number = "0002"
    # URL pattern: {base}/{year}/{field}-{num}/phot.dat
    # Confirmed from: https://ogle.astrouw.edu.pl/ogle4/ews/2024/gd-0004.html
    num = event_id.rsplit("-", 1)[-1]
    return f"{EWS_BASE}/{year}/{field}-{num}/phot.dat"


def _fetch_event(event_id: str, year: str, field: str):
    """
    Download one .phot.dat file and parse it.
    Returns (event_id, hjd, imag, sigma) on success, or None on failure.
    """
    url = _build_url(event_id, year, field)
    try:
        resp = requests.get(url, verify=False, timeout=TIMEOUT)
        if resp.status_code != 200:
            return None

        hjd_list, imag_list, sigma_list = [], [], []
        for line in resp.text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            cols = line.split()
            if len(cols) < 3:
                continue
            try:
                hjd_list.append(float(cols[0]))
                imag_list.append(float(cols[1]))
                sigma_list.append(float(cols[2]))
            except ValueError:
                continue

        if len(hjd_list) < 5:
            return None

        return (
            event_id,
            np.array(hjd_list,  dtype=np.float64),
            np.array(imag_list,  dtype=np.float64),
            np.array(sigma_list, dtype=np.float64),
        )

    except Exception:
        return None


def main():
    # -- 1. Load event list and draw random sample -------------------------
    events = pd.read_csv(EVENTS_CSV)
    print(f"Event catalogue: {len(events):,} events loaded from {EVENTS_CSV.name}")

    sample = events.sample(n=N_TARGET, random_state=SEED).reset_index(drop=True)
    print(f"Random sample  : {N_TARGET} events selected (seed={SEED})")
    print(f"Year range     : {sample['year'].min()}–{sample['year'].max()}")
    print(f"Field split    : {dict(sample['field'].value_counts())}")
    print()

    # -- 2. Build download tasks -------------------------------------------
    tasks = [
        (row["id"], str(int(row["year"])), row["field"].lower())
        for _, row in sample.iterrows()
    ]

    # -- 3. Parallel download ----------------------------------------------
    all_event_ids  = []
    all_hjd        = []
    all_imag       = []
    all_sigma      = []
    event_index    = []    # per-observation event index (0-based)

    success, fail = 0, 0

    print(f"Downloading {N_TARGET} phot.dat files ({MAX_WORKERS} parallel workers)...")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(_fetch_event, *task): task
            for task in tasks
        }

        for future in as_completed(futures):
            result = future.result()

            if result is not None:
                event_id, hjd, imag, sigma = result
                event_idx = success               # 0-based index of this event
                all_event_ids.append(event_id)
                all_hjd.append(hjd)
                all_imag.append(imag)
                all_sigma.append(sigma)
                event_index.extend([event_idx] * len(hjd))
                success += 1
            else:
                fail += 1

            done = success + fail
            if done % 300 == 0 or done == N_TARGET:
                print(f"  [{done:>4}/{N_TARGET}]  ok={success:>4}  failed={fail:>4}")

    print(f"\nFinished: {success} events downloaded successfully ({fail} failed/not found).")

    if not all_hjd:
        print("No data retrieved — check network connectivity and URL pattern.")
        return

    # -- 4. Compute per-event cadence (Δt between consecutive observations) -
    dt_list = []
    for hjd in all_hjd:
        order = np.argsort(hjd)
        dt = np.diff(hjd[order])
        if len(dt) > 0:
            dt_list.append(dt)

    # -- 5. Pool arrays ----------------------------------------------------
    hjd_pooled    = np.concatenate(all_hjd)
    imag_pooled   = np.concatenate(all_imag)
    sigma_pooled  = np.concatenate(all_sigma)
    ei_arr        = np.array(event_index, dtype=np.int32)
    dt_pooled     = np.concatenate(dt_list) if dt_list else np.array([], dtype=np.float64)

    # -- 6. Save -----------------------------------------------------------
    np.savez_compressed(
        OUTPUT_NPZ,
        hjd         = hjd_pooled,
        imag        = imag_pooled,
        sigma       = sigma_pooled,
        event_index = ei_arr,
        dt          = dt_pooled,
        n_events    = np.array([success], dtype=np.int32),
    )
    print(f"Saved photometry -> {OUTPUT_NPZ}")

    pd.DataFrame({"event_id": all_event_ids}).to_csv(OUTPUT_IDS, index=False)
    print(f"Saved event IDs  -> {OUTPUT_IDS}")

    # -- 7. Summary --------------------------------------------------------
    within_season = dt_pooled[(dt_pooled > 0) & (dt_pooled < 100)]

    print()
    print("--- Cadence summary -------------------------------------")
    print(f"  Total Δt values           : {len(dt_pooled):>10,}")
    print(f"  Within-season (<100 d)    : {len(within_season):>10,}")
    if len(within_season):
        print(f"  Median Δt                 : {np.median(within_season):.4f} days  "
              f"({np.median(within_season)*24*60:.1f} min)")
        print(f"  10th pct Δt               : {np.percentile(within_season, 10):.4f} days")
        print(f"  90th pct Δt               : {np.percentile(within_season, 90):.4f} days")

    print()
    print("--- Photometry summary ----------------------------------")
    print(f"  Total observations        : {len(hjd_pooled):>10,}")
    sane = (imag_pooled > 10) & (imag_pooled < 25) & (sigma_pooled > 0) & (sigma_pooled < 2)
    im_s = imag_pooled[sane]
    si_s = sigma_pooled[sane]
    if len(im_s):
        print(f"  Magnitude range           : [{im_s.min():.2f}, {im_s.max():.2f}] mag")
        print(f"  Median I_mag              : {np.median(im_s):.2f} mag")
        print(f"  Median σ_I                : {np.median(si_s):.4f} mag")
    print("---------------------------------------------------------")


main()