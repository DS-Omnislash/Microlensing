const form = document.getElementById("generate-form");
const generateBtn = document.getElementById("generate-btn");
const generateStatus = document.getElementById("generate-status");
const resultsSection = document.getElementById("results-section");
const validateBtn = document.getElementById("validate-btn");
const validateStatus = document.getElementById("validate-status");
const downloadLink = document.getElementById("download-link");
const downloadPklLink = document.getElementById("download-pkl-link");
const splitPreview = document.getElementById("split-preview");
const validationResults = document.getElementById("validation-results");

const nTotalInput = document.getElementById("n_total");
const binaryPercentInput = document.getElementById("binary_percent");
const binaryPercentRange = document.getElementById("binary_percent_range");
const nTimeInput = document.getElementById("n_time");

let currentDatasetId = null;
let uploadedDatasetId = null;
let singleSeed = 42;
let binarySeed = 42;
let currentPreset = null;

const refreshSingleBtn = document.getElementById("refresh-single-btn");
const refreshBinaryBtn = document.getElementById("refresh-binary-btn");
const colCheckboxes = document.querySelectorAll(".col-check");
const isMagCheckItem = document.getElementById("Is-mag-check-item");
const isMagCheck = document.getElementById("Is-mag-check");
const fsBlendCheckItem = document.getElementById("fs-blend-check-item");
const fsBlendCheck = document.getElementById("fs-blend-check");
const formatRadios = document.querySelectorAll('input[name="use_magnitudes"]');
const ogleRadios = document.querySelectorAll('input[name="ogle_noise"]');
const imperfectionsField = document.getElementById("imperfections-field");
const colsAllBtn = document.getElementById("cols-all-btn");
const colsNoneBtn = document.getElementById("cols-none-btn");

// Upload elements
const uploadFile = document.getElementById("upload-file");
const uploadFilename = document.getElementById("upload-filename");
const uploadAnalyzeBtn = document.getElementById("upload-analyze-btn");
const uploadStatus = document.getElementById("upload-status");
const uploadInfo = document.getElementById("upload-info");
const validateUploadedBtn = document.getElementById("validate-uploaded-btn");
const validateUploadedStatus = document.getElementById("validate-uploaded-status");

function updateSplitPreview() {
    const nTotal = parseInt(nTotalInput.value, 10) || 0;
    const pct = parseFloat(binaryPercentInput.value) || 0;
    const nBinary = Math.round(nTotal * (pct / 100));
    const nSingle = nTotal - nBinary;
    splitPreview.textContent =
        `This will generate ${nSingle.toLocaleString()} single-lens events ` +
        `and ${nBinary.toLocaleString()} binary-lens events ` +
        `(${nTotal.toLocaleString()} total).`;
}

binaryPercentRange.addEventListener("input", () => {
    binaryPercentInput.value = binaryPercentRange.value;
    updateSplitPreview();
});
binaryPercentInput.addEventListener("input", () => {
    binaryPercentRange.value = binaryPercentInput.value;
    updateSplitPreview();
});
nTotalInput.addEventListener("input", updateSplitPreview);
updateSplitPreview();


// ── Format toggle: A(t) vs I(t), and the mode-dependent columns ───────────
function setColToggle(item, check, enabled) {
    check.disabled = !enabled;
    check.checked = enabled;
    item.style.opacity = enabled ? "1" : "0.4";
    item.style.pointerEvents = enabled ? "auto" : "none";
}

function updateFormatToggle() {
    const useMag = document.querySelector('input[name="use_magnitudes"]:checked').value === "1";
    if (useMag) {
        imperfectionsField.style.display = "";
    } else {
        imperfectionsField.style.display = "none";
        document.getElementById("imperfections-none").checked = true;
    }
    const useOgle = useMag &&
        document.querySelector('input[name="ogle_noise"]:checked')?.value === "ogle";

    // I_s_mag exists only in I(t) mode; f_s_blend only when OGLE blending is applied.
    setColToggle(isMagCheckItem, isMagCheck, useMag);
    setColToggle(fsBlendCheckItem, fsBlendCheck, useOgle);
    updateGenerateEnabled();
}
formatRadios.forEach(r => r.addEventListener("change", updateFormatToggle));
ogleRadios.forEach(r => r.addEventListener("change", updateFormatToggle));

// ── Generate gating: with no output columns there is nothing to generate ──
function updateGenerateEnabled() {
    const anyChecked = Array.from(colCheckboxes).some(cb => cb.checked);
    generateBtn.disabled = !anyChecked;
    generateBtn.title = anyChecked ? "" : "Select at least one output column first.";
}

const PRESET_CLASSIFICATION  = new Set(["event_lenses", "__lightcurves__"]);
const PRESET_SINGLE_LENS     = new Set(["M_star_solar", "D_l_pc", "D_ls_pc", "D_s_pc", "v_perp_kms", "u0", "r_E_m", "t_E_days", "__lightcurves__"]);
const PRESET_MODEL           = new Set(["__lightcurves__"]);
const PRESET_DISTRIBUTIONS   = new Set(["event_lenses", "M_star_solar", "D_l_pc", "D_ls_pc", "D_s_pc", "v_perp_kms", "u0", "r_E_m", "t_E_days", "q", "a_pc", "eccentricity", "alpha_ref_rad", "I_s_mag", "f_s_blend"]);

function applyPreset(preset, name) {
    // Mode-gated columns (I_s_mag, f_s_blend) stay unchecked while disabled.
    colCheckboxes.forEach(cb => { cb.checked = preset.has(cb.value) && !cb.disabled; });
    currentPreset = name;
    updateGenerateEnabled();
}

colsAllBtn.addEventListener("click", () => { colCheckboxes.forEach(cb => { if (!cb.disabled) cb.checked = true; }); currentPreset = null; updateGenerateEnabled(); });
colsNoneBtn.addEventListener("click", () => { colCheckboxes.forEach(cb => cb.checked = false); currentPreset = null; updateGenerateEnabled(); });
document.getElementById("cols-classification-btn").addEventListener("click", () => applyPreset(PRESET_CLASSIFICATION, "Classification"));
document.getElementById("cols-single-btn").addEventListener("click",         () => applyPreset(PRESET_SINGLE_LENS,    "Single_Lens"));
document.getElementById("cols-model-btn").addEventListener("click",          () => applyPreset(PRESET_MODEL,          "Model"));
document.getElementById("cols-distributions-btn").addEventListener("click",  () => applyPreset(PRESET_DISTRIBUTIONS,  "Distributions"));

colCheckboxes.forEach(cb => cb.addEventListener("change", () => { currentPreset = null; updateGenerateEnabled(); }));
updateGenerateEnabled();

function setImage(id, b64) {
    const img = document.getElementById(id);
    if (!img) {
        console.warn(`Element with id "${id}" not found.`);
        return;
    }
    if (b64) {
        img.src = `data:image/png;base64,${b64}`;
        const block = img.closest(".plot-block");
        if (block) block.hidden = false;
    } else {
        const block = img.closest(".plot-block");
        if (block) block.hidden = true;
    }
}

// ── Shared validation runner ──────────────────────────────────────────────
async function runValidation(datasetId, btn, statusEl) {
    btn.disabled = true;
    statusEl.classList.remove("error");
    statusEl.textContent = "Running validation against TdR_RocRC.pdf reference distributions...";
    validationResults.hidden = true;

    try {
        const resp = await fetch(`/api/validate/${datasetId}`, { method: "POST" });

        if (!resp.ok) {
            const err = await resp.json().catch(() => ({}));
            throw new Error(err.detail || `Request failed with status ${resp.status}`);
        }

        const data = await resp.json();

        setImage("plot-validation-common", data.plots.validation_common);
        setImage("plot-validation-velocity", data.plots.validation_velocity);
        setImage("plot-validation-binary", data.plots.validation_binary);
        setImage("plot-validation-ogle", data.plots.validation_ogle);

        const noDataMsg = document.getElementById("no-validation-data");
        const tableHeading = document.getElementById("validation-table-heading");
        const table = document.getElementById("validation-table");
        const tbody = table.querySelector("tbody");

        tbody.innerHTML = "";
        if (data.stats.length > 0) {
            for (const row of data.stats) {
                const tr = document.createElement("tr");
                const statusClass = row.status === "OK" ? "status-ok" : "status-check";
                tr.innerHTML = `
                    <td>${row.parameter}</td>
                    <td>${row.reference}</td>
                    <td>${row.observed}</td>
                    <td>${row.expected}</td>
                    <td class="${statusClass}">${row.status}</td>
                `;
                tbody.appendChild(tr);
            }
            noDataMsg.hidden = true;
            tableHeading.hidden = false;
            table.hidden = false;
        } else {
            noDataMsg.hidden = false;
            tableHeading.hidden = true;
            table.hidden = true;
        }

        validationResults.hidden = false;
        statusEl.textContent = "Done.";
        validationResults.scrollIntoView({ behavior: "smooth" });
    } catch (err) {
        statusEl.classList.add("error");
        statusEl.textContent = `Error: ${err.message}`;
    } finally {
        btn.disabled = false;
    }
}

// ── Validate recently generated ───────────────────────────────────────────
validateBtn.addEventListener("click", async () => {
    if (!currentDatasetId) return;
    await runValidation(currentDatasetId, validateBtn, validateStatus);
});

// ── File upload: selection ────────────────────────────────────────────────
uploadFile.addEventListener("change", () => {
    const file = uploadFile.files[0];
    if (file) {
        uploadFilename.textContent = file.name;
        uploadAnalyzeBtn.disabled = false;
    } else {
        uploadFilename.textContent = "No file selected";
        uploadAnalyzeBtn.disabled = true;
    }
    uploadInfo.hidden = true;
    validationResults.hidden = true;
    uploadStatus.classList.remove("error");
    uploadStatus.textContent = "";
});

// ── File upload: analyze ──────────────────────────────────────────────────
uploadAnalyzeBtn.addEventListener("click", async () => {
    const file = uploadFile.files[0];
    if (!file) return;

    uploadAnalyzeBtn.disabled = true;
    uploadStatus.classList.remove("error");
    uploadStatus.textContent = "Analyzing file...";
    uploadInfo.hidden = true;
    validationResults.hidden = true;

    try {
        const formData = new FormData();
        formData.append("file", file);

        const resp = await fetch("/api/upload-validate", {
            method: "POST",
            body: formData,
        });

        if (!resp.ok) {
            const err = await resp.json().catch(() => ({}));
            throw new Error(err.detail || `Request failed with status ${resp.status}`);
        }

        const data = await resp.json();
        uploadedDatasetId = data.dataset_id;

        const infoGrid = document.getElementById("upload-info-grid");
        const paramList = data.param_columns.length > 0
            ? data.param_columns.join(", ")
            : "none";
        const lcLabel = data.has_lightcurves
            ? `yes (${data.n_time} pts/curve)`
            : "no";

        infoGrid.innerHTML = `
            <div class="info-cell">
                <span class="info-label">Total events</span>
                <strong>${data.n_total.toLocaleString()}</strong>
            </div>
            <div class="info-cell">
                <span class="info-label">Single-lens</span>
                <strong>${data.n_single.toLocaleString()}</strong>
            </div>
            <div class="info-cell">
                <span class="info-label">Binary-lens</span>
                <strong>${data.n_binary.toLocaleString()} (${data.binary_percent}%)</strong>
            </div>
            <div class="info-cell">
                <span class="info-label">Light curves</span>
                <strong>${lcLabel}</strong>
            </div>
            <div class="info-cell info-cell--wide">
                <span class="info-label">Parameter columns</span>
                <span class="info-cols">${paramList}</span>
            </div>
        `;

        uploadInfo.hidden = false;
        uploadStatus.textContent = "Analysis complete.";
    } catch (err) {
        uploadStatus.classList.add("error");
        uploadStatus.textContent = `Error: ${err.message}`;
    } finally {
        uploadAnalyzeBtn.disabled = false;
    }
});

// ── Validate uploaded dataset ─────────────────────────────────────────────
validateUploadedBtn.addEventListener("click", async () => {
    if (!uploadedDatasetId) return;
    await runValidation(uploadedDatasetId, validateUploadedBtn, validateUploadedStatus);
});

// ── Generate form ─────────────────────────────────────────────────────────
form.addEventListener("submit", async (e) => {
    e.preventDefault();

    const selectedParams = Array.from(colCheckboxes)
        .filter(cb => cb.checked)
        .map(cb => cb.value);

    const useMagnitudes = document.querySelector('input[name="use_magnitudes"]:checked').value === "1";
    const ogleNoise = document.querySelector('input[name="ogle_noise"]:checked')?.value === "ogle";
    const shuffleRows = document.querySelector('input[name="shuffle"]:checked')?.value === "1";
    const payload = {
        n_total: parseInt(nTotalInput.value, 10),
        binary_percent: parseFloat(binaryPercentInput.value),
        n_time: parseInt(nTimeInput.value, 10),
        selected_params: selectedParams,
        preset: currentPreset || "",
        use_magnitudes: useMagnitudes,
        ogle_noise: ogleNoise,
        shuffle: shuffleRows,
    };

    generateBtn.disabled = true;
    generateStatus.classList.remove("error");
    generateStatus.textContent = "Generating dataset and rendering plots... this may take a moment for larger datasets.";
    resultsSection.hidden = true;
    validationResults.hidden = true;
    validateBtn.disabled = true;
    document.getElementById("gen-model1-results").hidden = true;
    document.getElementById("gen-classify-status").textContent = "";

    try {
        const resp = await fetch("/api/generate", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
        });

        if (!resp.ok) {
            const err = await resp.json().catch(() => ({}));
            throw new Error(err.detail || `Request failed with status ${resp.status}`);
        }

        const data = await resp.json();
        currentDatasetId = data.dataset_id;
        singleSeed = 42;
        binarySeed = 42;

        document.getElementById("results-summary").innerHTML = `
            <p>
                Generated <strong>${data.n_total.toLocaleString()}</strong> events:
                <strong>${data.n_single.toLocaleString()}</strong> single-lens and
                <strong>${data.n_binary.toLocaleString()}</strong> binary-lens,
                each with <strong>${data.n_time}</strong> time points.
            </p>
        `;

        setImage("plot-distributions-common", data.plots.distributions_common);
        setImage("plot-sample-single-lightcurves", data.plots.sample_single_lightcurves);

        const binaryBlock = document.getElementById("binary-distributions-block");
        if (data.plots.distributions_binary) {
            setImage("plot-distributions-binary", data.plots.distributions_binary);
            binaryBlock.hidden = false;
        } else {
            binaryBlock.hidden = true;
        }

        const ogleDistBlock = document.getElementById("ogle-distributions-block");
        if (data.plots.distributions_ogle) {
            setImage("plot-distributions-ogle", data.plots.distributions_ogle);
            ogleDistBlock.hidden = false;
        } else {
            ogleDistBlock.hidden = true;
        }

        const binarySamplesBlock = document.getElementById("binary-samples-block");
        if (data.plots.sample_binary_lightcurves) {
            setImage("plot-sample-binary-lightcurves", data.plots.sample_binary_lightcurves);
            binarySamplesBlock.hidden = false;
        } else {
            binarySamplesBlock.hidden = true;
        }

        downloadLink.href = `/api/download/${currentDatasetId}`;
        downloadPklLink.href = `/api/download-pkl/${currentDatasetId}`;
        resultsSection.hidden = false;
        validateBtn.disabled = false;
        generateStatus.textContent = "Done.";
    } catch (err) {
        generateStatus.classList.add("error");
        generateStatus.textContent = `Error: ${err.message}`;
    } finally {
        updateGenerateEnabled();
    }
});

// ── Refresh light curve samples ───────────────────────────────────────────
refreshSingleBtn.addEventListener("click", async () => {
    if (!currentDatasetId) return;
    refreshSingleBtn.disabled = true;
    singleSeed = Math.floor(Math.random() * 1000000);
    try {
        const resp = await fetch(`/api/sample-single/${currentDatasetId}?seed=${singleSeed}`);
        if (!resp.ok) {
            const err = await resp.json().catch(() => ({}));
            throw new Error(err.detail || `Request failed with status ${resp.status}`);
        }
        const resData = await resp.json();
        setImage("plot-sample-single-lightcurves", resData.plot);
    } catch (err) {
        alert(`Error refreshing single-lens curves: ${err.message}`);
    } finally {
        refreshSingleBtn.disabled = false;
    }
});

refreshBinaryBtn.addEventListener("click", async () => {
    if (!currentDatasetId) return;
    refreshBinaryBtn.disabled = true;
    binarySeed = Math.floor(Math.random() * 1000000);
    try {
        const resp = await fetch(`/api/sample-binary/${currentDatasetId}?seed=${binarySeed}`);
        if (!resp.ok) {
            const err = await resp.json().catch(() => ({}));
            throw new Error(err.detail || `Request failed with status ${resp.status}`);
        }
        const resData = await resp.json();
        setImage("plot-sample-binary-lightcurves", resData.plot);
    } catch (err) {
        alert(`Error refreshing binary-lens curves: ${err.message}`);
    } finally {
        refreshBinaryBtn.disabled = false;
    }
});
// ── Model 1: single vs. binary classifier ────────────────────────────────
const model1File = document.getElementById("model1-file");
const model1Filename = document.getElementById("model1-filename");
const model1ClassifyBtn = document.getElementById("model1-classify-btn");
const model1Status = document.getElementById("model1-status");
const model1Results = document.getElementById("model1-results");
const model1Summary = document.getElementById("model1-summary");
const model1DownloadPredictions = document.getElementById("model1-download-predictions");
const model1DownloadBinaries = document.getElementById("model1-download-binaries");

model1File.addEventListener("change", () => {
    const file = model1File.files[0];
    if (file) {
        model1Filename.textContent = file.name;
        model1ClassifyBtn.disabled = false;
    } else {
        model1Filename.textContent = "No file selected";
        model1ClassifyBtn.disabled = true;
    }
    model1Results.hidden = true;
    model1Status.classList.remove("error");
    model1Status.textContent = "";
});

model1ClassifyBtn.addEventListener("click", async () => {
    const file = model1File.files[0];
    if (!file) return;

    model1ClassifyBtn.disabled = true;
    model1Status.classList.remove("error");
    model1Status.textContent = "Checking dataset and running the model...";
    model1Results.hidden = true;

    try {
        const formData = new FormData();
        formData.append("file", file);

        const resp = await fetch("/api/model1/predict", {
            method: "POST",
            body: formData,
        });

        if (!resp.ok) {
            const err = await resp.json().catch(() => ({}));
            throw new Error(err.detail || `Request failed with status ${resp.status}`);
        }

        const data = await resp.json();
        const pct = data.n_total > 0
            ? ((100 * data.n_binary) / data.n_total).toFixed(1)
            : "0.0";

        model1Summary.innerHTML = `
            <p>
                Classified <strong>${data.n_total.toLocaleString()}</strong> events:
                <strong>${data.n_single.toLocaleString()}</strong> single-lens and
                <strong>${data.n_binary.toLocaleString()}</strong> binary-lens
                (${pct}% binary).
            </p>
        `;

        model1DownloadPredictions.href = `/api/model1/download-predictions/${data.dataset_id}`;
        model1DownloadBinaries.href = `/api/model1/download-binaries/${data.dataset_id}`;

        // No detected binaries -> nothing to download in that file.
        if (data.n_binary === 0) {
            model1DownloadBinaries.classList.add("button--disabled");
        } else {
            model1DownloadBinaries.classList.remove("button--disabled");
        }

        model1Results.hidden = false;
        model1Status.textContent = "Done.";
    } catch (err) {
        model1Status.classList.add("error");
        model1Status.textContent = `Error: ${err.message}`;
    } finally {
        model1ClassifyBtn.disabled = false;
    }
});

// ── Model 1 (Real) — two-stage classification of noisy / gapped curves ────
const m1rFile = document.getElementById("model1real-file");
const m1rFilename = document.getElementById("model1real-filename");
const m1rBtn = document.getElementById("model1real-classify-btn");
const m1rStatus = document.getElementById("model1real-status");
const m1rResults = document.getElementById("model1real-results");
const m1rSummary = document.getElementById("model1real-summary");
const m1rWithProb = document.getElementById("model1real-with-prob");
const m1rLinks = {
    predGeneral: document.getElementById("m1r-dl-pred-general"),
    binGeneral: document.getElementById("m1r-dl-bin-general"),
    predStrict: document.getElementById("m1r-dl-pred-strict"),
    binStrict: document.getElementById("m1r-dl-bin-strict"),
    cascade: document.getElementById("m1r-dl-cascade"),
};

let m1rDatasetId = null;
let m1rCounts = { general: 0, strict: 0 };

// Rebuild every download URL: the probability column is a query flag, so the
// links must follow the checkbox without re-running the model.
function m1rRefreshLinks() {
    if (!m1rDatasetId) return;
    const wp = m1rWithProb.checked ? "true" : "false";
    const base = "/api/model1-real";
    m1rLinks.predGeneral.href = `${base}/download-predictions/${m1rDatasetId}?stage=general&with_prob=${wp}`;
    m1rLinks.binGeneral.href = `${base}/download-binaries/${m1rDatasetId}?stage=general&with_prob=${wp}`;
    m1rLinks.predStrict.href = `${base}/download-predictions/${m1rDatasetId}?stage=strict&with_prob=${wp}`;
    m1rLinks.binStrict.href = `${base}/download-binaries/${m1rDatasetId}?stage=strict&with_prob=${wp}`;
    m1rLinks.cascade.href = `${base}/download-cascade/${m1rDatasetId}?with_prob=${wp}`;

    m1rLinks.binGeneral.classList.toggle("button--disabled", m1rCounts.general === 0);
    m1rLinks.binStrict.classList.toggle("button--disabled", m1rCounts.strict === 0);
    m1rLinks.cascade.classList.toggle("button--disabled", m1rCounts.general === 0);
}

m1rWithProb.addEventListener("change", m1rRefreshLinks);

m1rFile.addEventListener("change", () => {
    const file = m1rFile.files[0];
    m1rFilename.textContent = file ? file.name : "No file selected";
    m1rBtn.disabled = !file;
    m1rResults.hidden = true;
    m1rStatus.classList.remove("error");
    m1rStatus.textContent = "";
});

m1rBtn.addEventListener("click", async () => {
    const file = m1rFile.files[0];
    if (!file) return;

    m1rBtn.disabled = true;
    m1rStatus.classList.remove("error");
    m1rStatus.textContent = "Checking dataset and running the model...";
    m1rResults.hidden = true;

    try {
        const formData = new FormData();
        formData.append("file", file);
        const resp = await fetch("/api/model1-real/predict", { method: "POST", body: formData });
        if (!resp.ok) {
            const err = await resp.json().catch(() => ({}));
            throw new Error(err.detail || `Request failed with status ${resp.status}`);
        }
        const data = await resp.json();
        m1rDatasetId = data.dataset_id;
        m1rCounts = { general: data.n_general_binary, strict: data.n_strict_binary };

        const pctG = data.n_total ? ((100 * data.n_general_binary) / data.n_total).toFixed(1) : "0.0";
        const pctS = data.n_total ? ((100 * data.n_strict_binary) / data.n_total).toFixed(1) : "0.0";
        const calNote = data.calibrated
            ? "Probabilities are Platt-calibrated, so a threshold is a precision target."
            : "This checkpoint is uncalibrated — probabilities are raw scores, not true probabilities.";

        m1rSummary.innerHTML = `
            <p>
                Scored <strong>${data.n_total.toLocaleString()}</strong> events.
                <strong>General</strong> (P ≥ ${data.general_threshold.toFixed(3)}) flagged
                <strong>${data.n_general_binary.toLocaleString()}</strong> candidates (${pctG}%).
                <strong>Strict</strong> (P ≥ ${data.strict_threshold.toFixed(3)}) kept
                <strong>${data.n_strict_binary.toLocaleString()}</strong> (${pctS}%).
            </p>
            <p class="hint">${calNote}</p>
        `;

        m1rRefreshLinks();
        m1rResults.hidden = false;
        m1rStatus.textContent = "Done.";
    } catch (err) {
        m1rStatus.classList.add("error");
        m1rStatus.textContent = `Error: ${err.message}`;
    } finally {
        m1rBtn.disabled = false;
    }
});

// ── Classify the recently generated dataset with Model 1 ──────────────────
const genClassifyBtn = document.getElementById("gen-classify-btn");
const genClassifyStatus = document.getElementById("gen-classify-status");
const genModel1Results = document.getElementById("gen-model1-results");
const genModel1Summary = document.getElementById("gen-model1-summary");
const genModel1DownloadPredictions = document.getElementById("gen-model1-download-predictions");
const genModel1DownloadBinaries = document.getElementById("gen-model1-download-binaries");

genClassifyBtn.addEventListener("click", async () => {
    if (!currentDatasetId) return;

    genClassifyBtn.disabled = true;
    genClassifyStatus.classList.remove("error");
    genClassifyStatus.textContent = "Running the recently generated dataset through Model 1...";
    genModel1Results.hidden = true;

    try {
        const resp = await fetch(`/api/model1/predict-generated/${currentDatasetId}`, {
            method: "POST",
        });

        if (!resp.ok) {
            const err = await resp.json().catch(() => ({}));
            throw new Error(err.detail || `Request failed with status ${resp.status}`);
        }

        const data = await resp.json();
        const pct = data.n_total > 0
            ? ((100 * data.n_binary) / data.n_total).toFixed(1)
            : "0.0";

        genModel1Summary.innerHTML = `
            <p>
                Classified <strong>${data.n_total.toLocaleString()}</strong> events:
                <strong>${data.n_single.toLocaleString()}</strong> single-lens and
                <strong>${data.n_binary.toLocaleString()}</strong> binary-lens
                (${pct}% binary).
            </p>
        `;

        genModel1DownloadPredictions.href = `/api/model1/download-predictions/${currentDatasetId}`;
        genModel1DownloadBinaries.href = `/api/model1/download-binaries/${currentDatasetId}`;
        if (data.n_binary === 0) {
            genModel1DownloadBinaries.classList.add("button--disabled");
        } else {
            genModel1DownloadBinaries.classList.remove("button--disabled");
        }

        genModel1Results.hidden = false;
        genClassifyStatus.textContent = "Done.";
    } catch (err) {
        genClassifyStatus.classList.add("error");
        genClassifyStatus.textContent = `Error: ${err.message}`;
    } finally {
        genClassifyBtn.disabled = false;
    }
});

// ── Reference distribution plots (Plotly.js) ─────────────────────────────
(function () {
    const dataEl = document.getElementById("dist-plots-data");
    if (!dataEl || !window.Plotly) return;
    const plots = JSON.parse(dataEl.textContent);
    const config = { displayModeBar: false, responsive: true };
    for (const [key, fig] of Object.entries(plots)) {
        if (key.endsWith("_ogle")) continue;   // mode variant, not its own panel
        const el = document.getElementById(`dist-plot-${key}`);
        if (el) Plotly.newPlot(el, fig.data, fig.layout, config);
    }

    // The baseline magnitude reference depends on the imperfections mode: in OGLE
    // mode the baseline is drawn from the real OGLE-IV observed distribution
    // (blended, median ~18.7 mag), not the theoretical source one (~19.8 mag).
    // Swap the curve so the reference always matches what is actually generated.
    const isEl = document.getElementById("dist-plot-I_s_mag");
    const ogleFig = plots["I_s_mag_ogle"];
    const baseFig = plots["I_s_mag"];
    if (isEl && ogleFig && baseFig) {
        const renderBaseline = () => {
            const useOgle = document.querySelector('input[name="ogle_noise"]:checked')?.value === "ogle";
            const fig = useOgle ? ogleFig : baseFig;
            Plotly.react(isEl, fig.data, fig.layout, config);
        };
        document.querySelectorAll('input[name="ogle_noise"]').forEach((radio) =>
            radio.addEventListener("change", renderBaseline));
        renderBaseline();
    }
})();