// SlothML Frontend Application Logic

let currentDatasetMeta = null;
let currentBenchmarkResults = null;
let featureChartInstance = null;
let debounceTimer = null;

// DOM Elements
const demosContainer = document.getElementById("demos-container");
const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("file-input");
const inspectionSection = document.getElementById("inspection-section");
const benchmarkSection = document.getElementById("benchmark-section");
const targetSelect = document.getElementById("target-select");
const btnRunBenchmark = document.getElementById("btn-run-benchmark");
const previewTable = document.getElementById("preview-table");
const loadingOverlay = document.getElementById("loading-overlay");
const loadingTitle = document.getElementById("loading-title");
const loadingDesc = document.getElementById("loading-desc");
const slothStatusText = document.getElementById("sloth-status-text");

// Modal Elements
const codeModal = document.getElementById("code-modal");
const btnOpenCode = document.getElementById("btn-open-code");
const btnCloseModal = document.getElementById("btn-close-modal");
const btnCopyCode = document.getElementById("btn-copy-code");
const codeBlock = document.getElementById("code-block");
const copyText = document.getElementById("copy-text");

// Initialize application
document.addEventListener("DOMContentLoaded", () => {
  loadDemos();
  setupEventListeners();
});

function setupEventListeners() {
  // File Dropzone
  dropzone.addEventListener("click", () => fileInput.click());
  fileInput.addEventListener("change", handleFileUpload);

  dropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropzone.classList.add("border-emerald-500", "bg-emerald-950/10");
  });

  dropzone.addEventListener("dragleave", () => {
    dropzone.classList.remove("border-emerald-500", "bg-emerald-950/10");
  });

  dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzone.classList.remove("border-emerald-500", "bg-emerald-950/10");
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      uploadFile(e.dataTransfer.files[0]);
    }
  });

  // Benchmark Trigger
  btnRunBenchmark.addEventListener("click", runBenchmark);

  // Modal Triggers
  btnOpenCode.addEventListener("click", openCheatCodeModal);
  btnCloseModal.addEventListener("click", () => codeModal.classList.add("hidden"));
  btnCopyCode.addEventListener("click", copyCodeToClipboard);
}

// 1. Load Preloaded Demos
async function loadDemos() {
  try {
    const res = await fetch("/api/demos");
    const data = await res.json();
    demosContainer.innerHTML = "";

    const icons = {
      saas_churn: "users",
      icu_readmission: "heart-pulse",
      house_prices: "home"
    };

    data.demos.forEach((demo) => {
      const card = document.createElement("div");
      card.className = "glass-card rounded-xl p-5 cursor-pointer flex flex-col justify-between group";
      card.onclick = () => selectDemo(demo.id);

      const iconName = icons[demo.id] || "file-text";

      card.innerHTML = `
        <div class="space-y-3">
          <div class="flex items-center justify-between">
            <div class="w-9 h-9 rounded-lg bg-emerald-500/15 border border-emerald-500/30 flex items-center justify-center text-emerald-400 group-hover:bg-emerald-500 group-hover:text-slate-950 transition">
              <i data-lucide="${iconName}" class="w-5 h-5"></i>
            </div>
            <span class="px-2 py-0.5 text-[10px] font-bold rounded-full bg-slate-800 text-slate-400 border border-slate-700">${demo.badge}</span>
          </div>
          <div>
            <h4 class="font-bold text-sm text-slate-100 group-hover:text-emerald-400 transition">${demo.name}</h4>
            <p class="text-xs text-slate-400 mt-1 line-clamp-2">${demo.description}</p>
          </div>
        </div>
        <div class="pt-4 flex items-center justify-between border-t border-slate-800/80 mt-4 text-[11px] text-slate-400">
          <span class="font-mono text-emerald-400/90">${demo.task}</span>
          <span class="flex items-center gap-1 group-hover:translate-x-1 transition text-emerald-400 font-semibold">
            Load Demo <i data-lucide="arrow-right" class="w-3.5 h-3.5"></i>
          </span>
        </div>
      `;
      demosContainer.appendChild(card);
    });

    lucide.createIcons();
  } catch (err) {
    console.error("Failed to load demos:", err);
  }
}

// 2. Select Demo Dataset
async function selectDemo(demoId) {
  showLoading("Sloth is stretching...", "Ingesting demo dataset into memory");
  try {
    const formData = new FormData();
    formData.append("demo_id", demoId);

    const res = await fetch("/api/load-demo", {
      method: "POST",
      body: formData
    });
    const data = await res.json();
    handleDataLoaded(data);
  } catch (err) {
    alert("Error loading demo: " + err);
  } finally {
    hideLoading();
  }
}

// 3. Upload Custom File
function handleFileUpload(e) {
  if (e.target.files && e.target.files[0]) {
    uploadFile(e.target.files[0]);
  }
}

async function uploadFile(file) {
  showLoading("Sloth is reading file...", `Parsing ${file.name}`);
  const formData = new FormData();
  formData.append("file", file);

  try {
    const res = await fetch("/api/upload", {
      method: "POST",
      body: formData
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Upload failed");
    handleDataLoaded(data);
  } catch (err) {
    alert("Upload error: " + err.message);
  } finally {
    hideLoading();
  }
}

// 4. Handle Loaded Data
function handleDataLoaded(data) {
  currentDatasetMeta = data;

  // Update UI Elements
  document.getElementById("dataset-filename").textContent = data.filename;
  document.getElementById("dataset-dims").textContent = `${data.rows_count} rows × ${data.cols_count} features`;
  
  const taskBadge = document.getElementById("dataset-task-badge");
  taskBadge.textContent = data.detected_task.toUpperCase();
  if (data.detected_task === "classification") {
    taskBadge.className = "px-2 py-0.5 text-xs font-medium rounded-full bg-emerald-500/20 text-emerald-400 border border-emerald-500/30";
  } else {
    taskBadge.className = "px-2 py-0.5 text-xs font-medium rounded-full bg-cyan-500/20 text-cyan-400 border border-cyan-500/30";
  }

  // Populate Target Column Select
  targetSelect.innerHTML = "";
  data.columns.forEach((col) => {
    const opt = document.createElement("option");
    opt.value = col.name;
    opt.textContent = `${col.name} (${col.dtype})`;
    if (col.name === data.detected_target) {
      opt.selected = true;
    }
    targetSelect.appendChild(opt);
  });

  // Render Table Preview
  renderPreviewTable(data.columns, data.preview);

  // Reveal Inspection Section
  inspectionSection.classList.remove("hidden");
  benchmarkSection.classList.add("hidden");

  // Scroll into view
  inspectionSection.scrollIntoView({ behavior: "smooth", block: "start" });
  slothStatusText.textContent = `Glanced at ${data.filename}`;
}

function renderPreviewTable(columns, rows) {
  let html = `<thead class="bg-slate-900 text-slate-400 font-semibold border-b border-slate-800"><tr>`;
  columns.forEach((col) => {
    html += `<th class="px-4 py-2.5 whitespace-nowrap">${col.name}</th>`;
  });
  html += `</tr></thead><tbody class="divide-y divide-slate-800/60 font-mono text-slate-300">`;

  rows.forEach((row) => {
    html += `<tr class="hover:bg-slate-800/40 transition">`;
    columns.forEach((col) => {
      const val = row[col.name] !== undefined ? row[col.name] : "-";
      html += `<td class="px-4 py-2 whitespace-nowrap">${val}</td>`;
    });
    html += `</tr>`;
  });
  html += `</tbody>`;
  previewTable.innerHTML = html;
}

// 5. Run Benchmark (TabPFN vs Baseline)
async function runBenchmark() {
  const chosenTarget = targetSelect.value;
  showLoading(
    "Sloth is doing in-context inference...",
    "Zero tuning. Single forward pass transformer matrix multiplication.",
    true
  );
  slothStatusText.textContent = "Crunching tabular foundation vectors...";

  try {
    const res = await fetch("/api/benchmark", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ target_col: chosenTarget })
    });
    const payload = await res.json();
    if (!res.ok) throw new Error(payload.detail || "Modeling failed");

    currentBenchmarkResults = payload.results;
    displayBenchmarkResults(payload.results);

    // Confetti celebration for Sloth speed!
    confetti({
      particleCount: 70,
      spread: 60,
      origin: { y: 0.7 }
    });

    slothStatusText.textContent = "Done in seconds! Sloth is going back to sleep 🦥💤";
  } catch (err) {
    alert("Benchmark failed: " + err.message);
    slothStatusText.textContent = "Sloth encountered an error.";
  } finally {
    hideLoading();
  }
}

// 6. Display Benchmark Results & Charts
function displayBenchmarkResults(results) {
  benchmarkSection.classList.remove("hidden");

  // Multiplier Badge
  document.getElementById("speedup-badge").textContent = `${results.speedup_ratio} FASTER`;

  // TabPFN Metrics
  document.getElementById("tabpfn-time").textContent = results.tabpfn.training_time;
  if (results.task_type === "classification") {
    document.getElementById("tabpfn-metric-label").textContent = "Accuracy Score";
    document.getElementById("tabpfn-score").textContent = `${results.tabpfn.metrics.accuracy}%`;
    const auc = results.tabpfn.metrics.roc_auc ? `ROC-AUC: ${results.tabpfn.metrics.roc_auc}%` : `F1: ${results.tabpfn.metrics.f1_score}`;
    document.getElementById("tabpfn-submetric").textContent = auc;

    document.getElementById("baseline-metric-label").textContent = "Accuracy Score";
    document.getElementById("baseline-score").textContent = `${results.baseline.metrics.accuracy}%`;
    const b_auc = results.baseline.metrics.roc_auc ? `ROC-AUC: ${results.baseline.metrics.roc_auc}%` : `F1: ${results.baseline.metrics.f1_score}`;
    document.getElementById("baseline-submetric").textContent = b_auc;
  } else {
    document.getElementById("tabpfn-metric-label").textContent = "R² Variance Score";
    document.getElementById("tabpfn-score").textContent = `${results.tabpfn.metrics.r2_score}%`;
    document.getElementById("tabpfn-submetric").textContent = `RMSE: ${results.tabpfn.metrics.rmse}`;

    document.getElementById("baseline-metric-label").textContent = "R² Variance Score";
    document.getElementById("baseline-score").textContent = `${results.baseline.metrics.r2_score}%`;
    document.getElementById("baseline-submetric").textContent = `RMSE: ${results.baseline.metrics.rmse}`;
  }

  document.getElementById("baseline-time").textContent = results.baseline.training_time;

  // Render Feature Importances Chart
  renderFeaturesChart(results.feature_importances);

  // Render Executive Briefing
  renderExecutiveBrief(results.executive_brief);

  // Setup What-If Simulator Controls
  setupSimulator(results.feature_stats, results.feature_importances);

  // Smooth scroll
  benchmarkSection.scrollIntoView({ behavior: "smooth", block: "start" });
  lucide.createIcons();
}

function renderFeaturesChart(importances) {
  const ctx = document.getElementById("features-chart").getContext("2d");
  if (featureChartInstance) {
    featureChartInstance.destroy();
  }

  const labels = Object.keys(importances);
  const data = Object.values(importances);

  featureChartInstance = new Chart(ctx, {
    type: "bar",
    data: {
      labels: labels,
      datasets: [
        {
          label: "Impact %",
          data: data,
          backgroundColor: "rgba(16, 185, 129, 0.75)",
          borderColor: "#10B981",
          borderWidth: 1,
          borderRadius: 6
        }
      ]
    },
    options: {
      indexAxis: "y",
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: "#0F172A",
          titleColor: "#10B981",
          bodyColor: "#F1F5F9",
          borderColor: "#334155",
          borderWidth: 1
        }
      },
      scales: {
        x: {
          grid: { color: "rgba(51, 65, 85, 0.3)" },
          ticks: { color: "#94A3B8", callback: (val) => val + "%" }
        },
        y: {
          grid: { display: false },
          ticks: { color: "#CBD5E1", font: { weight: "600" } }
        }
      }
    }
  });
}

function renderExecutiveBrief(briefs) {
  const container = document.getElementById("executive-brief-container");
  container.innerHTML = "";

  briefs.forEach((item) => {
    const card = document.createElement("div");
    card.className = "p-3.5 rounded-xl bg-slate-900/60 border border-slate-800/80 flex items-start gap-3";
    card.innerHTML = `
      <div class="w-8 h-8 rounded-lg bg-amber-500/15 border border-amber-500/30 flex items-center justify-center text-amber-400 shrink-0 mt-0.5">
        <i data-lucide="${item.icon || 'info'}" class="w-4 h-4"></i>
      </div>
      <div>
        <h5 class="text-xs font-bold text-slate-200">${item.title}</h5>
        <p class="text-xs text-slate-400 mt-0.5 leading-relaxed">${item.text}</p>
      </div>
    `;
    container.appendChild(card);
  });
  lucide.createIcons();
}

// 7. Interactive What-If Simulator Setup
function setupSimulator(stats, importances) {
  const container = document.getElementById("simulator-controls");
  container.innerHTML = "";

  // Pick top features according to importance
  const topFeatures = Object.keys(importances).slice(0, 5);
  // If fewer than 4, add others from stats
  Object.keys(stats).forEach((f) => {
    if (topFeatures.length < 5 && !topFeatures.includes(f)) {
      topFeatures.push(f);
    }
  });

  const currentValues = {};

  topFeatures.forEach((feat) => {
    const stat = stats[feat];
    if (!stat) return;

    const row = document.createElement("div");
    row.className = "bg-slate-900/60 p-3.5 rounded-xl border border-slate-800 space-y-2";

    if (stat.type === "numeric") {
      const minVal = stat.min;
      const maxVal = stat.max;
      const currentVal = stat.median;
      currentValues[feat] = currentVal;

      const step = (maxVal - minVal) > 10 ? 1 : 0.1;

      row.innerHTML = `
        <div class="flex items-center justify-between">
          <span class="text-xs font-bold text-slate-200">${feat}</span>
          <span class="font-mono text-xs text-cyan-400 font-semibold" id="val-${feat}">${currentVal}</span>
        </div>
        <input type="range" id="slider-${feat}" min="${minVal}" max="${maxVal}" step="${step}" value="${currentVal}" class="cursor-pointer">
      `;
      container.appendChild(row);

      const slider = row.querySelector(`#slider-${feat}`);
      slider.addEventListener("input", (e) => {
        document.getElementById(`val-${feat}`).textContent = e.target.value;
        currentValues[feat] = parseFloat(e.target.value);
        triggerSimulation(currentValues);
      });
    } else if (stat.type === "categorical") {
      const currentVal = stat.default;
      currentValues[feat] = currentVal;

      let optionsHtml = "";
      stat.classes.forEach((cls) => {
        optionsHtml += `<option value="${cls}" ${cls === currentVal ? "selected" : ""}>${cls}</option>`;
      });

      row.innerHTML = `
        <div class="flex items-center justify-between">
          <span class="text-xs font-bold text-slate-200">${feat}</span>
        </div>
        <select id="select-${feat}" class="w-full bg-slate-950 border border-slate-700 rounded-lg px-2.5 py-1.5 text-xs text-cyan-300 outline-none">
          ${optionsHtml}
        </select>
      `;
      container.appendChild(row);

      const select = row.querySelector(`#select-${feat}`);
      select.addEventListener("change", (e) => {
        currentValues[feat] = e.target.value;
        triggerSimulation(currentValues);
      });
    }
  });

  // Run initial simulation
  triggerSimulation(currentValues);
}

function triggerSimulation(features) {
  clearTimeout(debounceTimer);
  debounceTimer = setTimeout(async () => {
    try {
      const res = await fetch("/api/simulate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ features })
      });
      const data = await res.json();
      if (data.success) {
        updateSimulationOutput(data.simulation);
      }
    } catch (err) {
      console.error("Simulation error:", err);
    }
  }, 40); // 40ms debounce for ultra-responsive live dragging!
}

function updateSimulationOutput(sim) {
  const labelEl = document.getElementById("sim-prediction-label");
  const meterEl = document.getElementById("sim-confidence-meter");
  const probsContainer = document.getElementById("sim-probabilities-container");

  if (sim.task_type === "classification") {
    labelEl.textContent = sim.prediction;
    
    // Style color dynamically
    const predLower = String(sim.prediction).toLowerCase();
    if (predLower.includes("churn") || predLower.includes("high risk") || predLower.includes("fail")) {
      labelEl.className = "text-3xl font-black text-rose-400 tracking-tight transition";
    } else {
      labelEl.className = "text-3xl font-black text-emerald-400 tracking-tight transition";
    }

    meterEl.textContent = `${sim.confidence}% Model Confidence`;

    // Probability Breakdown Bars
    probsContainer.innerHTML = "";
    if (sim.probabilities) {
      Object.entries(sim.probabilities).forEach(([cls, prob]) => {
        const bar = document.createElement("div");
        bar.className = "space-y-1";
        bar.innerHTML = `
          <div class="flex justify-between text-xs text-slate-400">
            <span>${cls}</span>
            <span class="font-mono font-bold text-slate-200">${prob}%</span>
          </div>
          <div class="w-full bg-slate-800 rounded-full h-2 overflow-hidden">
            <div class="bg-gradient-to-r from-emerald-500 to-teal-400 h-2 rounded-full transition-all duration-300" style="width: ${prob}%"></div>
          </div>
        `;
        probsContainer.appendChild(bar);
      });
    }
  } else {
    // Regression
    labelEl.textContent = `$${Number(sim.prediction).toLocaleString()}`;
    labelEl.className = "text-3xl font-black text-cyan-400 tracking-tight transition";
    meterEl.textContent = "Estimated Value Point";
    probsContainer.innerHTML = `
      <div class="text-xs text-slate-400 bg-slate-950 p-2.5 rounded-lg border border-slate-800">
        In-context TabPFN continuous distribution estimate computed instantly.
      </div>
    `;
  }
}

// 8. Cheat Code Modal
async function openCheatCodeModal() {
  codeModal.classList.remove("hidden");
  try {
    const res = await fetch("/api/export-code");
    const code = await res.text();
    codeBlock.textContent = code;
  } catch (err) {
    codeBlock.textContent = "# Error loading code: " + err;
  }
}

function copyCodeToClipboard() {
  navigator.clipboard.writeText(codeBlock.textContent).then(() => {
    copyText.textContent = "Copied to Clipboard!";
    setTimeout(() => {
      copyText.textContent = "Copy to Clipboard";
    }, 2000);
  });
}

let loadingInterval = null;

// Helpers
function showLoading(title, desc, isBenchmark = false) {
  loadingTitle.textContent = title;
  loadingDesc.textContent = desc;
  const stepsContainer = document.getElementById("loading-steps");
  
  if (isBenchmark && stepsContainer) {
    stepsContainer.classList.remove("hidden");
    stepsContainer.innerHTML = `
      <div class="flex items-center gap-2 text-emerald-400">
        <span class="w-2 h-2 rounded-full bg-emerald-400 animate-ping"></span>
        <span>🧹 Auto-sanitizing feature columns & missing values...</span>
      </div>
    `;

    clearInterval(loadingInterval);
    const steps = [
      { delay: 350, text: "🦥 In-Context Tabular Transformer forward pass running..." },
      { delay: 900, text: "⚡ Evaluating predictions vs. Random Forest pipeline..." },
      { delay: 1500, text: "🎛️ Distilling real-time scenario simulator (<5ms)..." }
    ];

    steps.forEach((step) => {
      setTimeout(() => {
        if (!loadingOverlay.classList.contains("hidden")) {
          const item = document.createElement("div");
          item.className = "flex items-center gap-2 text-slate-300 transition-all";
          item.innerHTML = `<span class="text-emerald-400 font-bold">✓</span> <span>${step.text}</span>`;
          stepsContainer.appendChild(item);
        }
      }, step.delay);
    });
  } else if (stepsContainer) {
    stepsContainer.classList.add("hidden");
  }

  loadingOverlay.classList.remove("hidden");
}

function hideLoading() {
  clearInterval(loadingInterval);
  loadingOverlay.classList.add("hidden");
}
