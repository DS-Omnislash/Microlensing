"""FastAPI application for generating synthetic microlensing datasets."""

import io
import uuid
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd

# pyrefly: ignore [missing-import]
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
# pyrefly: ignore [missing-import]
from fastapi.responses import StreamingResponse
# pyrefly: ignore [missing-import]
from fastapi.staticfiles import StaticFiles
# pyrefly: ignore [missing-import]
from fastapi.templating import Jinja2Templates
# pyrefly: ignore [missing-import]
from pydantic import BaseModel, Field

from . import content, distribution_plots, plotting
from .dataset import generate_dataset

BASE_DIR = Path(__file__).resolve().parent.parent

app = FastAPI(title="Microlensing Dataset Generator")
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# Generation limits.
N_TOTAL_MIN, N_TOTAL_MAX = 10, 500_000
N_TIME_MIN, N_TIME_MAX = 50, 1_000
BINARY_PCT_MIN, BINARY_PCT_MAX = 0.0, 100.0

# Simple in-memory LRU cache of generated datasets. Eviction is BOTH
# count-based and size-aware: a request at the configured maximums can weigh
# several GB, so the cache also enforces a total-bytes budget. The most recent
# entry is always kept regardless of its size (so maximal requests still work;
# they just evict everything else).
MAX_CACHED_DATASETS = 5
MAX_CACHE_BYTES = 2 * 1024**3  # ~2 GiB across all cached datasets
_DATASET_CACHE: "OrderedDict[str, dict]" = OrderedDict()


def _estimate_entry_bytes(data: dict) -> int:
    """Approximate memory held by a cache entry (DataFrames + numpy arrays)."""
    total = 0
    for value in data.values():
        if isinstance(value, pd.DataFrame):
            total += int(value.memory_usage(index=True).sum())
        elif isinstance(value, np.ndarray):
            total += int(value.nbytes)
    return total


def _store_dataset(data: dict) -> str:
    dataset_id = uuid.uuid4().hex
    data["_cache_bytes"] = _estimate_entry_bytes(data)
    _DATASET_CACHE[dataset_id] = data
    _DATASET_CACHE.move_to_end(dataset_id)

    def _total_bytes() -> int:
        return sum(d.get("_cache_bytes", 0) for d in _DATASET_CACHE.values())

    while len(_DATASET_CACHE) > 1 and (
        len(_DATASET_CACHE) > MAX_CACHED_DATASETS or _total_bytes() > MAX_CACHE_BYTES
    ):
        _DATASET_CACHE.popitem(last=False)
    return dataset_id


def _get_dataset(dataset_id: str) -> dict:
    data = _DATASET_CACHE.get(dataset_id)
    if data is None:
        raise HTTPException(status_code=404, detail="Dataset not found or expired. Please generate it again.")
    _DATASET_CACHE.move_to_end(dataset_id)
    return data


def _select_columns(df, data: dict):
    """Return df filtered to the columns chosen by the user, or the full df if none specified."""
    selected = data.get("selected_params", [])
    if not selected:
        return df
    selected_set = set(selected)
    time_cols = [c for c in df.columns if c.startswith("t_") and c[2:].isdigit()]
    keep = set()
    for s in selected_set:
        if s == "__lightcurves__":
            keep.update(time_cols)
        elif s in df.columns:
            keep.add(s)
    return df[[c for c in df.columns if c in keep]]


def _build_filename(data: dict, ext: str) -> str:
    # Token-safe characters only: "%" and parentheses are not valid unquoted in
    # a Content-Disposition header and some browsers mangle them.
    pct_str = f"{data['binary_percent']:g}pct"
    fmt_part = "_I" if data.get("use_magnitudes") else "_A"
    noise_part = "_OGLE" if data.get("ogle_noise") else ""
    preset_part = f"_{data['preset']}" if data.get("preset") else ""
    return f"Microlensing_Dataset_{data['n_total']}_{pct_str}_{data['n_time']}pts{fmt_part}{noise_part}{preset_part}.{ext}"


def _data_from_df(df: pd.DataFrame) -> dict:
    """Build a data dict suitable for validation from an uploaded DataFrame."""
    if "event_lenses" in df.columns:
        n_binary = int((df["event_lenses"] == 2).sum())
        n_single = int((df["event_lenses"] == 1).sum())
    else:
        n_binary = 0
        n_single = len(df)

    time_cols = [c for c in df.columns if c.startswith("t_") and c[2:].isdigit()]

    data: dict = {
        "df": df,
        "n_total": len(df),
        "n_single": n_single,
        "n_binary": n_binary,
        "n_time": len(time_cols),
    }

    col_map = [
        ("M_star_solar", "M_star_solar"),
        ("D_l_pc",       "D_l_pc"),
        ("D_ls_pc",      "D_ls_pc"),
        ("v_perp_kms",   "v_perp_kms"),
        ("u0",           "u0_all"),
        ("t_E_days",     "t_E_days"),
    ]
    for df_col, key in col_map:
        if df_col in df.columns:
            data[key] = df[df_col].dropna().to_numpy(dtype=float)

    # (I_s, f_s) must stay PAIRED (they come from the same catalogue event), so
    # when both columns exist they are extracted with a joint dropna.
    if "I_s_mag" in df.columns and "f_s_blend" in df.columns:
        pair = df[["I_s_mag", "f_s_blend"]].dropna()
        if len(pair) > 0:
            data["I_s_mag"] = pair["I_s_mag"].to_numpy(dtype=float)
            data["f_s_blend"] = pair["f_s_blend"].to_numpy(dtype=float)
    elif "I_s_mag" in df.columns:
        arr = df["I_s_mag"].dropna().to_numpy(dtype=float)
        if len(arr) > 0:
            data["I_s_mag"] = arr

    # Infer whether this dataset carries OGLE imperfections: either it kept its
    # blend-fraction column, or its light curves contain cadence gaps (NaN).
    # Without this the four OGLE checks would silently be skipped on re-upload.
    has_nan_curves = bool(df[time_cols].isna().any().any()) if time_cols else False
    data["ogle_noise"] = bool("f_s_blend" in df.columns or has_nan_curves)

    if n_binary > 0 and "event_lenses" in df.columns:
        binary_mask = df["event_lenses"] == 2
        for df_col, key in [
            ("q",            "q_binary"),
            ("a_pc",         "a_pc_binary"),
            ("eccentricity", "e_binary"),
            ("alpha_ref_rad","alpha_ref_binary"),
        ]:
            if df_col in df.columns:
                arr = df.loc[binary_mask, df_col].dropna().to_numpy(dtype=float)
                if len(arr) > 0:
                    data[key] = arr

    return data


class GenerateRequest(BaseModel):
    n_total: int = Field(..., ge=N_TOTAL_MIN, le=N_TOTAL_MAX)
    binary_percent: float = Field(..., ge=BINARY_PCT_MIN, le=BINARY_PCT_MAX)
    n_time: int = Field(..., ge=N_TIME_MIN, le=N_TIME_MAX)
    selected_params: list[str] = Field(default_factory=list)
    preset: str = Field(default="")
    use_magnitudes: bool = Field(default=False)
    ogle_noise: bool = Field(default=False)
    shuffle: bool = Field(default=False)


@app.get("/")
def index(request: Request):
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "parameter_info": content.PARAMETER_INFO,
            "event_ratio_explanation": content.EVENT_RATIO_EXPLANATION,
            "n_time_explanation": content.N_TIME_EXPLANATION,
            "recommended_binary_percent": content.RECOMMENDED_BINARY_PERCENT,
            "recommended_n_time": content.RECOMMENDED_N_TIME,
            "n_total_min": N_TOTAL_MIN,
            "n_total_max": N_TOTAL_MAX,
            "n_time_min": N_TIME_MIN,
            "n_time_max": N_TIME_MAX,
            "binary_pct_min": BINARY_PCT_MIN,
            "binary_pct_max": BINARY_PCT_MAX,
            "distribution_plots_json": distribution_plots.DISTRIBUTION_PLOTS_JSON,
        },
    )


@app.post("/api/generate")
def api_generate(req: GenerateRequest):
    data = generate_dataset(
        n_total=req.n_total,
        binary_fraction=req.binary_percent / 100.0,
        n_time=req.n_time,
        use_magnitudes=req.use_magnitudes,
        ogle_noise=req.ogle_noise,
        shuffle=req.shuffle,
    )
    data["selected_params"] = req.selected_params
    data["binary_percent"] = req.binary_percent
    data["preset"] = req.preset
    dataset_id = _store_dataset(data)

    return {
        "dataset_id": dataset_id,
        "n_total": data["n_total"],
        "n_single": data["n_single"],
        "n_binary": data["n_binary"],
        "n_time": data["n_time"],
        "plots": {
            "distributions_common": plotting.plot_distributions_common(data),
            "distributions_binary": plotting.plot_distributions_binary(data),
            "distributions_ogle": plotting.plot_distributions_ogle(data) if data.get("ogle_noise") else None,
            "sample_single_lightcurves": plotting.plot_sample_single_lightcurves(data),
            "sample_binary_lightcurves": plotting.plot_sample_binary_lightcurves(data),
        },
    }


@app.get("/api/sample-single/{dataset_id}")
def api_sample_single(dataset_id: str, seed: int = 42):
    data = _get_dataset(dataset_id)
    # Only datasets generated this session carry the raw arrays the plots need.
    if "single_lightcurves" not in data:
        raise HTTPException(status_code=400, detail="Sample plots are only available for datasets generated in this session.")
    plot_b64 = plotting.plot_sample_single_lightcurves(data, seed=seed)
    if plot_b64 is None:
        raise HTTPException(status_code=400, detail="No single-lens events in this dataset")
    return {"plot": plot_b64}


@app.get("/api/sample-binary/{dataset_id}")
def api_sample_binary(dataset_id: str, seed: int = 42):
    data = _get_dataset(dataset_id)
    if "binary_lightcurves" not in data:
        raise HTTPException(status_code=400, detail="Sample plots are only available for datasets generated in this session.")
    plot_b64 = plotting.plot_sample_binary_lightcurves(data, seed=seed)
    if plot_b64 is None:
        raise HTTPException(status_code=400, detail="No binary-lens events in this dataset")
    return {"plot": plot_b64}


@app.post("/api/validate/{dataset_id}")
def api_validate(dataset_id: str):
    data = _get_dataset(dataset_id)

    common_img, velocity_img, binary_img, stats_list = plotting.plot_validation_available(data)

    ogle_img, ogle_stats = None, []
    if data.get("ogle_noise"):
        result = plotting.plot_ogle_validation(data)
        if result is not None:
            ogle_img, ogle_stats = result

    return {
        "dataset_id": dataset_id,
        "plots": {
            "validation_common": common_img,
            "validation_velocity": velocity_img,
            "validation_binary": binary_img,
            "validation_ogle": ogle_img,
        },
        "stats": stats_list + ogle_stats,
    }


@app.post("/api/upload-validate")
async def api_upload_validate(file: UploadFile = File(...)):
    content_bytes = await file.read()
    fname = (file.filename or "").lower()
    try:
        if fname.endswith(".pkl"):
            df = pd.read_pickle(io.BytesIO(content_bytes))
        else:
            df = pd.read_csv(io.StringIO(content_bytes.decode("utf-8", errors="replace")))
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not parse file: {exc}")

    if not isinstance(df, pd.DataFrame):
        raise HTTPException(status_code=422, detail="File does not contain a tabular dataset.")

    data = _data_from_df(df)
    dataset_id = _store_dataset(data)

    time_cols = [c for c in df.columns if c.startswith("t_") and c[2:].isdigit()]
    param_cols = [c for c in df.columns if not (c.startswith("t_") and c[2:].isdigit())]
    binary_pct = round(100.0 * data["n_binary"] / data["n_total"], 1) if data["n_total"] > 0 else 0.0

    return {
        "dataset_id": dataset_id,
        "n_total": data["n_total"],
        "n_single": data["n_single"],
        "n_binary": data["n_binary"],
        "binary_percent": binary_pct,
        "n_time": data["n_time"],
        "param_columns": param_cols,
        "has_lightcurves": len(time_cols) > 0,
    }


@app.post("/api/model1/predict")
async def api_model1_predict(file: UploadFile = File(...)):
    """Run Model 1 (Simple) on an uploaded model dataset (light curves only)."""
    content_bytes = await file.read()
    fname = (file.filename or "").lower()
    try:
        if fname.endswith(".pkl"):
            df = pd.read_pickle(io.BytesIO(content_bytes))
        else:
            df = pd.read_csv(io.StringIO(content_bytes.decode("utf-8", errors="replace")))
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not parse file: {exc}")

    if not isinstance(df, pd.DataFrame):
        raise HTTPException(status_code=422, detail="File does not contain a tabular dataset.")

    # Import lazily so the webapp starts even without torch installed.
    try:
        from . import model1
    except ImportError:
        raise HTTPException(
            status_code=503,
            detail="The model runtime (PyTorch) is not installed on the server.",
        )

    try:
        result = model1.classify_dataframe(df)
    except model1.ModelDatasetError as exc:
        # Dataset rejected before inference -- 422 so the UI can show why.
        raise HTTPException(status_code=422, detail=str(exc))

    dataset_id = _store_dataset(
        {
            "model1_df": df,
            "model1_pred": result["pred"],
            "model1_prob_binary": result["prob_binary"],
            "source_filename": file.filename or "dataset",
        }
    )

    return {
        "dataset_id": dataset_id,
        "n_total": result["n_total"],
        "n_single": result["n_single"],
        "n_binary": result["n_binary"],
    }


@app.post("/api/model1/predict-generated/{dataset_id}")
def api_model1_predict_generated(dataset_id: str):
    """Run Model 1 (Simple) on a dataset generated in-app this session."""
    data = _get_dataset(dataset_id)

    if "df" not in data:
        raise HTTPException(status_code=404, detail="No generated dataset for this id.")

    # The Simple model was trained on I(t) magnitude curves; A(t) is a
    # different (sign-flipped) domain and would not classify reliably.
    if not data.get("use_magnitudes"):
        raise HTTPException(
            status_code=422,
            detail="The model expects I(t) magnitude light curves. This dataset "
            "is in A(t) (amplification) mode. Regenerate it in I(t) mode to classify it.",
        )

    try:
        from . import model1
    except ImportError:
        raise HTTPException(
            status_code=503,
            detail="The model runtime (PyTorch) is not installed on the server.",
        )

    try:
        result = model1.classify_generated(data["df"])
    except model1.ModelDatasetError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    # Attach predictions to the existing cache entry so the download endpoints
    # (which look up model1_pred / model1_df) work on the same id.
    data["model1_pred"] = result["pred"]
    data["model1_prob_binary"] = result["prob_binary"]
    data["model1_df"] = data["df"]

    return {
        "dataset_id": dataset_id,
        "n_total": result["n_total"],
        "n_single": result["n_single"],
        "n_binary": result["n_binary"],
    }


@app.get("/api/model1/download-predictions/{dataset_id}")
def api_model1_download_predictions(dataset_id: str):
    """Stream the per-event predictions.csv for a Model 1 run."""
    data = _get_dataset(dataset_id)
    if "model1_pred" not in data:
        raise HTTPException(status_code=404, detail="No Model 1 predictions for this id.")

    pred = data["model1_pred"]
    prob = data["model1_prob_binary"]
    out = pd.DataFrame(
        {
            "row_index": range(len(pred)),
            "pred_label": ["binary" if p == 1 else "single" for p in pred],
            "prob_binary": prob,
        }
    )

    buf = io.StringIO()
    out.to_csv(buf, index=False, float_format="%.6g")
    buf.seek(0)
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=predictions.csv"},
    )


@app.get("/api/model1/download-binaries/{dataset_id}")
def api_model1_download_binaries(dataset_id: str):
    """Stream a CSV of only the events predicted to be binary-lens (full curves)."""
    data = _get_dataset(dataset_id)
    if "model1_pred" not in data:
        raise HTTPException(status_code=404, detail="No Model 1 predictions for this id.")

    df = data["model1_df"]
    pred = data["model1_pred"]
    binary_df = df.loc[pred == 1]

    buf = io.StringIO()
    binary_df.to_csv(buf, index=False, float_format="%.10g")
    buf.seek(0)
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=detected_binary_events.csv"},
    )


# --------------------------------------------------------------------------- #
# Model 1 (Real) -- noisy / gapped OGLE-like curves, two-stage output
# --------------------------------------------------------------------------- #
def _import_model1_real():
    try:
        from . import model1_real
    except ImportError:
        raise HTTPException(
            status_code=503,
            detail="The model runtime (PyTorch) is not installed on the server.",
        )
    return model1_real


def _store_real_result(store: dict, df, result: dict) -> None:
    """Attach a Real-model result to a cache entry (shared by both entry points)."""
    store["model1r_df"] = df
    store["model1r_prob"] = result["prob_binary"]
    store["model1r_general_pred"] = result["general_pred"]
    store["model1r_strict_pred"] = result["strict_pred"]
    store["model1r_general_threshold"] = result["general_threshold"]
    store["model1r_strict_threshold"] = result["strict_threshold"]
    store["model1r_calibrated"] = result["calibrated"]


def _real_summary(dataset_id: str, result: dict) -> dict:
    return {
        "dataset_id": dataset_id,
        "n_total": result["n_total"],
        "n_general_binary": result["n_general_binary"],
        "n_strict_binary": result["n_strict_binary"],
        "general_threshold": result["general_threshold"],
        "strict_threshold": result["strict_threshold"],
        "calibrated": result["calibrated"],
    }


@app.post("/api/model1-real/predict")
async def api_model1_real_predict(file: UploadFile = File(...)):
    """Run Model 1 (Real) on an uploaded model dataset (light curves only).

    Unlike the Simple model this accepts cadence gaps (NaN) as well as clean
    curves -- the Real model takes an observed-mask channel.
    """
    content_bytes = await file.read()
    fname = (file.filename or "").lower()
    try:
        if fname.endswith(".pkl"):
            df = pd.read_pickle(io.BytesIO(content_bytes))
        else:
            df = pd.read_csv(io.StringIO(content_bytes.decode("utf-8", errors="replace")))
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not parse file: {exc}")

    if not isinstance(df, pd.DataFrame):
        raise HTTPException(status_code=422, detail="File does not contain a tabular dataset.")

    model1_real = _import_model1_real()
    try:
        result = model1_real.classify_dataframe(df)
    except model1_real.ModelDatasetError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    store = {"source_filename": file.filename or "dataset"}
    _store_real_result(store, df, result)
    return _real_summary(_store_dataset(store), result)


@app.post("/api/model1-real/predict-generated/{dataset_id}")
def api_model1_real_predict_generated(dataset_id: str):
    """Run Model 1 (Real) on a dataset generated in-app this session."""
    data = _get_dataset(dataset_id)
    if "df" not in data:
        raise HTTPException(status_code=404, detail="No generated dataset for this id.")

    # Trained on I(t) magnitudes; A(t) is a different (sign-flipped) domain.
    if not data.get("use_magnitudes"):
        raise HTTPException(
            status_code=422,
            detail="The model expects I(t) magnitude light curves. This dataset "
            "is in A(t) (amplification) mode. Regenerate it in I(t) mode to classify it.",
        )

    model1_real = _import_model1_real()
    try:
        result = model1_real.classify_generated(data["df"])
    except model1_real.ModelDatasetError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    _store_real_result(data, data["df"], result)
    return _real_summary(dataset_id, result)


def _require_real(dataset_id: str) -> dict:
    data = _get_dataset(dataset_id)
    if "model1r_prob" not in data:
        raise HTTPException(
            status_code=404, detail="No Model 1 (Real) predictions for this id.")
    return data


def _csv_response(frame, filename: str, float_format: str = "%.6g"):
    buf = io.StringIO()
    frame.to_csv(buf, index=False, float_format=float_format)
    buf.seek(0)
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@app.get("/api/model1-real/download-predictions/{dataset_id}")
def api_model1_real_download_predictions(
    dataset_id: str, stage: str = "general", with_prob: bool = True
):
    """Per-event predictions for one stage.

    stage=general -> permissive candidate list (hand-off for review)
    stage=strict  -> clean, high-confidence catalogue
    with_prob     -> include the calibrated P(binary) column
    """
    data = _require_real(dataset_id)
    if stage not in ("general", "strict"):
        raise HTTPException(status_code=422, detail="stage must be 'general' or 'strict'.")

    pred = data[f"model1r_{stage}_pred"]
    out = pd.DataFrame({
        "row_index": range(len(pred)),
        "pred_label": ["binary" if p == 1 else "single" for p in pred],
    })
    if with_prob:
        out["prob_binary"] = data["model1r_prob"]
    return _csv_response(out, f"predictions_{stage}.csv")


@app.get("/api/model1-real/download-binaries/{dataset_id}")
def api_model1_real_download_binaries(
    dataset_id: str, stage: str = "general", with_prob: bool = True
):
    """Full curves of the events predicted binary at one stage."""
    data = _require_real(dataset_id)
    if stage not in ("general", "strict"):
        raise HTTPException(status_code=422, detail="stage must be 'general' or 'strict'.")

    df = data["model1r_df"]
    pred = data[f"model1r_{stage}_pred"]
    out = df.loc[pred == 1].copy()
    if with_prob:
        out.insert(0, "prob_binary", np.asarray(data["model1r_prob"])[pred == 1])
    return _csv_response(out, f"detected_binaries_{stage}.csv", float_format="%.10g")


@app.get("/api/model1-real/download-cascade/{dataset_id}")
def api_model1_real_download_cascade(dataset_id: str, with_prob: bool = True):
    """The strict stage applied to the GENERAL stage's candidates.

    This is the review product: the candidate list from the general stage, each
    row carrying the strict stage's verdict and calibrated probability. For a
    single score the kept set equals the strict stage by construction
    ({p>=general} AND {p>=strict} == {p>=strict}); it is exported separately so a
    different second-opinion model can be substituted later without reworking the
    pipeline or the UI.
    """
    data = _require_real(dataset_id)
    prob = np.asarray(data["model1r_prob"])
    general_pred = data["model1r_general_pred"]
    strict_pred = data["model1r_strict_pred"]

    cand = np.where(general_pred == 1)[0]
    out = pd.DataFrame({
        "row_index": cand,
        "general_pred": "binary",
        "strict_pred": ["binary" if strict_pred[i] == 1 else "single" for i in cand],
        "kept_by_strict": [bool(strict_pred[i] == 1) for i in cand],
    })
    if with_prob:
        out["prob_binary"] = prob[cand]
    return _csv_response(out, "predictions_cascade_strict_over_general.csv")


@app.get("/api/download/{dataset_id}")
def api_download(dataset_id: str):
    data = _get_dataset(dataset_id)
    # Uploaded / model-run cache entries lack the generation metadata the
    # filename needs (and the user already has the file).
    if "df" not in data or "binary_percent" not in data:
        raise HTTPException(status_code=400, detail="Download is only available for datasets generated in this session.")
    df = _select_columns(data["df"], data)

    buf = io.StringIO()
    df.to_csv(buf, index=False, float_format="%.10g")
    buf.seek(0)

    filename = _build_filename(data, "csv")
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@app.get("/api/download-pkl/{dataset_id}")
def api_download_pkl(dataset_id: str):
    data = _get_dataset(dataset_id)
    if "df" not in data or "binary_percent" not in data:
        raise HTTPException(status_code=400, detail="Download is only available for datasets generated in this session.")
    df = _select_columns(data["df"], data)

    buf = io.BytesIO()
    df.to_pickle(buf)
    buf.seek(0)

    filename = _build_filename(data, "pkl")
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="application/octet-stream",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )