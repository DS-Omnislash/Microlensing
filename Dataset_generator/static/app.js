const form = document.getElementById("generate-form");
const generateBtn = document.getElementById("generate-btn");
const generateStatus = document.getElementById("generate-status");
const resultsSection = document.getElementById("results-section");
const validationSection = document.getElementById("validation-section");
const validateBtn = document.getElementById("validate-btn");
const validateStatus = document.getElementById("validate-status");
const downloadLink = document.getElementById("download-link");
const splitPreview = document.getElementById("split-preview");

const nTotalInput = document.getElementById("n_total");
const binaryPercentInput = document.getElementById("binary_percent");
const binaryPercentRange = document.getElementById("binary_percent_range");
const nTimeInput = document.getElementById("n_time");

let currentDatasetId = null;
let singleSeed = 42;
let binarySeed = 42;

const refreshSingleBtn = document.getElementById("refresh-single-btn");
const refreshBinaryBtn = document.getElementById("refresh-binary-btn");

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

form.addEventListener("submit", async (e) => {
    e.preventDefault();

    const payload = {
        n_total: parseInt(nTotalInput.value, 10),
        binary_percent: parseFloat(binaryPercentInput.value),
        n_time: parseInt(nTimeInput.value, 10),
    };

    generateBtn.disabled = true;
    generateStatus.classList.remove("error");
    generateStatus.textContent = "Generating dataset and rendering plots... this may take a moment for larger datasets.";
    resultsSection.hidden = true;
    validationSection.hidden = true;

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
        setImage("plot-coverage", data.plots.coverage);

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
        resultsSection.hidden = false;
        generateStatus.textContent = "Done.";
    } catch (err) {
        generateStatus.classList.add("error");
        generateStatus.textContent = `Error: ${err.message}`;
    } finally {
        generateBtn.disabled = false;
    }
});

validateBtn.addEventListener("click", async () => {
    if (!currentDatasetId) {
        return;
    }

    validateBtn.disabled = true;
    validateStatus.classList.remove("error");
    validateStatus.textContent = "Running validation against TDR_ROC.pdf reference distributions...";

    try {
        const resp = await fetch(`/api/validate/${currentDatasetId}`, { method: "POST" });

        if (!resp.ok) {
            const err = await resp.json().catch(() => ({}));
            throw new Error(err.detail || `Request failed with status ${resp.status}`);
        }

        const data = await resp.json();

        setImage("plot-validation-common", data.plots.validation_common);
        setImage("plot-validation-velocity", data.plots.validation_velocity);

        const binaryBlock = document.getElementById("binary-validation-block");
        if (data.plots.validation_binary) {
            setImage("plot-validation-binary", data.plots.validation_binary);
            binaryBlock.hidden = false;
        } else {
            binaryBlock.hidden = true;
        }

        const tbody = document.querySelector("#validation-table tbody");
        tbody.innerHTML = "";
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

        validationSection.hidden = false;
        validateStatus.textContent = "Done.";
        validationSection.scrollIntoView({ behavior: "smooth" });
    } catch (err) {
        validateStatus.classList.add("error");
        validateStatus.textContent = `Error: ${err.message}`;
    } finally {
        validateBtn.disabled = false;
    }
});

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
