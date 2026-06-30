"""FastAPI application for generating synthetic microlensing datasets."""

import io
import uuid
from collections import OrderedDict
from pathlib import Path

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

# Simple in-memory LRU cache of generated datasets.
MAX_CACHED_DATASETS = 5
_DATASET_CACHE: "OrderedDict[str, dict]" = OrderedDict()


def _store_dataset(data: dict) -> str:
    dataset_id = uuid.uuid4().hex
    _DATASET_CACHE[dataset_id] = data
    _DATASET_CACHE.move_to_end(dataset_id)
    while len(_DATASET_CACHE) > MAX_CACHED_DATASETS:
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
    pct_str = f"{data['binary_percent']:g}%"
    fmt_part = "_I(t)" if data.get("use_magnitudes") else "_A(t)"
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

    if "I_s_mag" in df.columns:
        arr = df["I_s_mag"].dropna().to_numpy(dtype=float)
        if len(arr) > 0:
            data["I_s_mag"] = arr

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
    plot_b64 = plotting.plot_sample_single_lightcurves(data, seed=seed)
    if plot_b64 is None:
        raise HTTPException(status_code=400, detail="No single-lens events in this dataset")
    return {"plot": plot_b64}


@app.get("/api/sample-binary/{dataset_id}")
def api_sample_binary(dataset_id: str, seed: int = 42):
    data = _get_dataset(dataset_id)
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


@app.get("/api/download/{dataset_id}")
def api_download(dataset_id: str):
    data = _get_dataset(dataset_id)
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