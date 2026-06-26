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

const PRESET_CLASSIFICATION  = new Set(["event_lenses", "__lightcurves__"]);
const PRESET_SINGLE_LENS     = new Set(["M_star_solar", "D_l_pc", "D_ls_pc", "D_s_pc", "v_perp_kms", "u0", "r_E_m", "t_E_days", "__lightcurves__"]);
const PRESET_MODEL           = new Set(["__lightcurves__"]);
const PRESET_DISTRIBUTIONS   = new Set(["event_lenses", "M_star_solar", "D_l_pc", "D_ls_pc", "D_s_pc", "v_perp_kms", "u0", "r_E_m", "t_E_days", "q", "a_pc", "eccentricity", "alpha_ref_rad"]);

function applyPreset(preset, name) {
    colCheckboxes.forEach(cb => { cb.checked = preset.has(cb.value); });
    currentPreset = name;
}

colsAllBtn.addEventListener("click", () => { colCheckboxes.forEach(cb => cb.checked = true);  currentPreset = null; });
colsNoneBtn.addEventListener("click", () => { colCheckboxes.forEach(cb => cb.checked = false); currentPreset = null; });
document.getElementById("cols-classification-btn").addEventListener("click", () => applyPreset(PRESET_CLASSIFICATION, "Classification"));
document.getElementById("cols-single-btn").addEventListener("click",         () => applyPreset(PRESET_SINGLE_LENS,    "Single_Lens"));
document.getElementById("cols-model-btn").addEventListener("click",          () => applyPreset(PRESET_MODEL,          "Model"));
document.getElementById("cols-distributions-btn").addEventListener("click",  () => applyPreset(PRESET_DISTRIBUTIONS,  "Distributions"));

colCheckboxes.forEach(cb => cb.addEventListener("change", () => { currentPreset = null; }));

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
    statusEl.textContent = "Running validation against TDR_ROC.pdf reference distributions...";
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

    const payload = {
        n_total: parseInt(nTotalInput.value, 10),
        binary_percent: parseFloat(binaryPercentInput.value),
        n_time: parseInt(nTimeInput.value, 10),
        selected_params: selectedParams,
        preset: currentPreset || "",
    };

    generateBtn.disabled = true;
    generateStatus.classList.remove("error");
    generateStatus.textContent = "Generating dataset and rendering plots... this may take a moment for larger datasets.";
    resultsSection.hidden = true;
    validationResults.hidden = true;
    validateBtn.disabled = true;

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
        generateBtn.disabled = false;
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
// ── Reference distribution plots (Plotly.js) ─────────────────────────────
(function () {
    const dataEl = document.getElementById("dist-plots-data");
    if (!dataEl || !window.Plotly) return;
    const plots = JSON.parse(dataEl.textContent);
    const config = { displayModeBar: false, responsive: true };
    for (const [key, fig] of Object.entries(plots)) {
        const el = document.getElementById(`dist-plot-${key}`);
        if (el) Plotly.newPlot(el, fig.data, fig.layout, config);
    }
})();