"use strict";

(() => {
  const selector = document.getElementById("training-run-select");
  if (!selector) return;
  const statusNode = document.getElementById("report-status");
  const runSummary = document.getElementById("run-summary");
  const modelList = document.getElementById("model-report-list");
  const failedRuns = document.getElementById("failed-runs");
  const chartWrap = document.getElementById("comparison-chart-wrap");
  let models = [];
  let runs = [];
  let chart = null;

  function el(tag, text, className) {
    const item = document.createElement(tag);
    if (text !== undefined && text !== null) item.textContent = String(text);
    if (className) item.className = className;
    return item;
  }

  async function api(url) {
    const response = await fetch(url);
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(data.error || `Request failed (${response.status}).`);
    return data;
  }

  function showStatus(message, kind = "info") {
    statusNode.hidden = !message;
    statusNode.className = `ml-status ${kind}`;
    statusNode.textContent = message || "";
  }

  function percent(value) {
    return value === null || value === undefined ? "Unavailable" : `${(Number(value) * 100).toFixed(1)}%`;
  }

  function renderRuns() {
    selector.replaceChildren();
    if (!runs.length) {
      const option = el("option", "No training runs yet");
      option.value = "";
      selector.append(option);
      selector.disabled = true;
      return;
    }
    selector.disabled = false;
    runs.forEach((run) => {
      const date = run.started_at ? new Date(run.started_at).toLocaleString() : "";
      const option = el("option", `Run ${run.id} · ${run.dataset_name} · ${run.status} · ${date}`);
      option.value = String(run.id);
      selector.append(option);
    });
    const latestCompleted = runs.find((run) => run.status === "completed");
    if (latestCompleted) selector.value = String(latestCompleted.id);
  }

  function renderConfusion(model, parent) {
    const metrics = model.test_metrics;
    const matrix = metrics.confusion_matrix;
    const caption = el("p", `Held-out test confusion matrix · rows = actual (${matrix.labels[0]} then ${matrix.labels[1]}), columns = predicted.`, "confusion-caption");
    const table = el("table", null, "confusion-table");
    const body = el("tbody");
    const heading = el("tr");
    ["Actual \\ Predicted", matrix.labels[0], matrix.labels[1]].forEach((value) => heading.append(el("th", value)));
    body.append(heading);
    const negative = el("tr");
    [matrix.labels[0], matrix.negative_as_negative, matrix.negative_as_positive].forEach((value, index) => negative.append(el(index === 0 ? "th" : "td", value)));
    body.append(negative);
    const positive = el("tr");
    [matrix.labels[1], matrix.positive_as_negative, matrix.positive_as_positive].forEach((value, index) => positive.append(el(index === 0 ? "th" : "td", value)));
    body.append(positive);
    table.append(body);
    parent.append(caption, table);
  }

  function renderModel(model) {
    const card = el("article", null, "model-report-card");
    card.append(el("h3", `${model.algorithm} · model v${model.id}`));
    card.append(el("p", `${model.dataset_name} · scikit-learn ${model.algorithm_version} · target ${model.target_column} (${model.positive_class} = churn)`, "model-caption"));
    const validation = el("p", `Validation F1: ${percent(model.validation_metrics.f1_score)} · validation sample: ${model.validation_metrics.sample_count}.`, "report-selection");
    card.append(validation);
    const metrics = model.test_metrics;
    const metricGrid = el("div", null, "metric-grid");
    [[metrics.accuracy, "Accuracy"], [metrics.precision, "Positive precision"], [metrics.recall, "Positive recall"], [metrics.f1_score, "Positive F1"], [metrics.roc_auc, "ROC-AUC"], [metrics.sample_count, "Test rows"]].forEach(([value, label]) => {
      const item = el("div", null, "metric-value");
      item.append(el("strong", label === "Test rows" ? Number(value).toLocaleString() : percent(value)), el("span", label));
      metricGrid.append(item);
    });
    card.append(metricGrid);
    const distribution = metrics.class_distribution;
    card.append(el("p", `Test class distribution: ${distribution.negative_class} ${distribution.negative_count} · ${distribution.positive_class} ${distribution.positive_count}.`, "model-caption"));
    if (metrics.roc_auc_note) card.append(el("p", metrics.roc_auc_note, "model-caption"));
    renderConfusion(model, card);
    card.append(el("p", `Feature columns: ${model.feature_columns.join(", ")}`, "model-caption"));
    modelList.append(card);
  }

  function renderChart(runModels) {
    const canvas = document.getElementById("comparison-chart");
    if (chart) {
      chart.destroy();
      chart = null;
    }
    chartWrap.hidden = !runModels.length;
    if (!runModels.length) return;
    if (typeof window.Chart !== "function") {
      chartWrap.replaceChildren(el("p", "Chart.js could not be loaded. The model metric tables remain available below.", "chart-unavailable"));
      chartWrap.hidden = false;
      return;
    }
    chartWrap.replaceChildren(canvas);
    chart = new window.Chart(canvas, {
      type: "bar",
      data: {
        labels: runModels.map((model) => model.algorithm),
        datasets: [
          { label: "Accuracy", data: runModels.map((model) => model.test_metrics.accuracy), backgroundColor: "#9bcbbd" },
          { label: "Precision", data: runModels.map((model) => model.test_metrics.precision), backgroundColor: "#68aa96" },
          { label: "Recall", data: runModels.map((model) => model.test_metrics.recall), backgroundColor: "#168c7a" },
          { label: "F1", data: runModels.map((model) => model.test_metrics.f1_score), backgroundColor: "#365d5b" },
        ],
      },
      options: { responsive: true, maintainAspectRatio: false, scales: { y: { min: 0, max: 1, ticks: { callback: (value) => `${Math.round(value * 100)}%` } } } },
    });
  }

  function renderSelectedRun() {
    const runId = Number(selector.value);
    const run = runs.find((item) => item.id === runId);
    modelList.replaceChildren();
    runSummary.replaceChildren();
    if (!run) {
      renderChart([]);
      return;
    }
    const runModels = models.filter((model) => model.training_run_id === runId);
    [[`Run ${run.id}`, "run-chip"], [run.dataset_name, "run-chip"], [run.status, "run-chip"], [`${runModels.length} model versions`, "run-chip"]].forEach(([value, kind]) => runSummary.append(el("span", value, kind)));
    if (run.summary?.split) {
      const split = run.summary.split;
      runSummary.append(el("p", `Split: ${split.strategy} · train ${split.train_count}, validation ${split.validation_count}, held-out test ${split.test_count}.`, "split-note"));
      (split.warnings || []).forEach((warning) => runSummary.append(el("p", warning, "model-warning")));
    }
    runSummary.append(el("p", run.selection_procedure, "report-selection"));
    runModels.forEach(renderModel);
    renderChart(runModels);
  }

  async function load() {
    showStatus("Loading saved model versions and training history…", "loading");
    try {
      const result = await api("/api/models");
      models = result.models;
      runs = result.training_runs;
      renderRuns();
      renderSelectedRun();
      const failed = runs.filter((run) => run.status === "failed");
      failedRuns.replaceChildren();
      failed.forEach((run) => failedRuns.append(el("p", `Run ${run.id} · ${run.dataset_name}: ${run.error_message || "Training failed."}`, "failed-run")));
      showStatus(runs.length ? "Metrics shown here were produced from recorded validation and held-out test splits." : "No model training runs have been recorded yet.", runs.length ? "success" : "info");
    } catch (error) {
      showStatus(error.message, "error");
    }
  }

  selector.addEventListener("change", renderSelectedRun);
  load();
})();
