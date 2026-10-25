"use strict";

(() => {
  const trainForm = document.getElementById("train-form");
  if (!trainForm) return;

  const trainingDataset = document.getElementById("training-dataset");
  const modelSelect = document.getElementById("model-version");
  const scoringDataset = document.getElementById("scoring-dataset");
  const trainButton = document.getElementById("train-button");
  const scoreButton = document.getElementById("score-button");
  const trainingStatus = document.getElementById("training-status");
  const predictionStatus = document.getElementById("prediction-status");
  const trainingResults = document.getElementById("training-results");
  const predictionSummary = document.getElementById("prediction-summary");
  const resultsSection = document.getElementById("prediction-results");
  let datasets = [];
  let models = [];
  let activeRunId = null;
  let currentPage = 1;
  let pageCount = 1;
  let searchTimer = null;

  function el(tag, text, className) {
    const item = document.createElement(tag);
    if (text !== undefined && text !== null) item.textContent = String(text);
    if (className) item.className = className;
    return item;
  }

  async function api(url, options = {}) {
    const response = await fetch(url, options);
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(data.error || `Request failed (${response.status}).`);
    return data;
  }

  function status(target, message, kind = "info") {
    target.hidden = !message;
    target.className = `ml-status ${kind}`;
    target.replaceChildren();
    if (kind === "loading") target.append(el("span", null, "spinner"));
    if (message) target.append(document.createTextNode(message));
  }

  function metric(value) {
    return value === null || value === undefined ? "Unavailable" : `${(Number(value) * 100).toFixed(1)}%`;
  }

  function fillSelect(select, emptyLabel, items, labelFor) {
    select.replaceChildren();
    const empty = el("option", emptyLabel);
    empty.value = "";
    select.append(empty);
    items.forEach((item) => {
      const option = el("option", labelFor(item));
      option.value = String(item.id);
      select.append(option);
    });
    select.disabled = items.length === 0;
  }

  function renderModelList() {
    const ready = models.filter((model) => model.status === "ready");
    fillSelect(modelSelect, ready.length ? "Choose a saved model" : "Train a model first", ready,
      (model) => `${model.algorithm} · v${model.id} · ${model.dataset_name}`);
  }

  function renderDatasetOptions() {
    const trainable = datasets.filter((dataset) => {
      const config = dataset.configuration;
      return config && config.dataset_type === "labeled_historical" && config.target && config.positive_class && config.negative_class && config.features?.length;
    });
    const scoreable = datasets.filter((dataset) => dataset.configuration?.features?.length);
    fillSelect(trainingDataset, trainable.length ? "Choose labeled historical data" : "No configured labeled datasets", trainable,
      (dataset) => `${dataset.original_filename} · ${dataset.row_count.toLocaleString()} rows`);
    fillSelect(scoringDataset, scoreable.length ? "Choose a configured dataset" : "Configure dataset features first", scoreable,
      (dataset) => `${dataset.original_filename} · ${dataset.dataset_type}`);
  }

  async function loadReferences() {
    try {
      const [datasetResult, modelResult] = await Promise.all([api("/api/datasets"), api("/api/models")]);
      datasets = datasetResult.datasets;
      models = modelResult.models;
      renderDatasetOptions();
      renderModelList();
    } catch (error) {
      status(trainingStatus, error.message, "error");
      status(predictionStatus, error.message, "error");
    }
  }

  function renderTrainingResult(result) {
    trainingResults.replaceChildren();
    const modelsResult = result.models || [];
    if (!modelsResult.length) return;
    const description = el("p", "Both candidates were fit using training data, compared on validation positive-class F1, then refit on train + validation. Test metrics below use a separate held-out split.", "split-note");
    trainingResults.append(description);
    const warning = el("p", "Accuracy alone can be misleading for imbalanced outcomes. Review precision, recall, F1, class distribution, and confusion matrices in Reports.", "model-warning");
    trainingResults.append(warning);
    const table = el("table", null, "comparison-table");
    const head = el("thead");
    const row = el("tr");
    ["Baseline", "Validation F1", "Test accuracy", "Test precision", "Test recall", "Test F1", "Test ROC-AUC"].forEach((name) => row.append(el("th", name)));
    head.append(row);
    const body = el("tbody");
    modelsResult.forEach((model) => {
      const line = el("tr");
      [model.algorithm, metric(model.validation_metrics.f1_score), metric(model.test_metrics.accuracy),
        metric(model.test_metrics.precision), metric(model.test_metrics.recall), metric(model.test_metrics.f1_score),
        metric(model.test_metrics.roc_auc)].forEach((value) => line.append(el("td", value)));
      body.append(line);
    });
    table.append(head, body);
    trainingResults.append(table);
    const split = result.training_run?.summary?.split;
    if (split) {
      trainingResults.append(el("p", `Evaluation split: ${split.strategy}. Training ${split.train_count}, validation ${split.validation_count}, test ${split.test_count}; ${split.dropped_missing_target_rows} rows with missing target were excluded.`, "split-note"));
      (split.warnings || []).forEach((message) => trainingResults.append(el("p", message, "model-warning")));
    }
  }

  trainForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    if (!trainingDataset.value) return;
    trainButton.disabled = true;
    trainButton.textContent = "Training…";
    trainingResults.replaceChildren();
    status(trainingStatus, "Training Logistic Regression and Random Forest; evaluating on validation and held-out test splits…", "loading");
    try {
      const result = await api(`/api/datasets/${trainingDataset.value}/train`, { method: "POST" });
      status(trainingStatus, `Training run ${result.training_run.id} completed. Both baseline pipelines were saved.`, "success");
      renderTrainingResult(result);
      await loadReferences();
    } catch (error) {
      status(trainingStatus, error.message, "error");
    } finally {
      trainButton.disabled = false;
      trainButton.replaceChildren(document.createTextNode("Train both models "), el("span", "→"));
    }
  });

  const resultSearch = document.getElementById("result-search");
  const riskFilter = document.getElementById("risk-filter");
  const resultSort = document.getElementById("result-sort");
  const sortDirection = document.getElementById("sort-direction");

  async function loadPredictionPage() {
    if (!activeRunId) return;
    const query = new URLSearchParams({
      page: String(currentPage), per_page: "50", q: resultSearch.value,
      risk: riskFilter.value, sort: resultSort.value, direction: sortDirection.value,
    });
    const body = document.getElementById("result-body");
    body.replaceChildren(el("tr", null));
    body.firstElementChild.append(el("td", "Loading prediction rows…"));
    try {
      const result = await api(`/api/predictions/${activeRunId}/results?${query}`);
      pageCount = Math.max(result.pages, 1);
      currentPage = Math.min(currentPage, pageCount);
      const head = document.getElementById("result-head");
      head.replaceChildren();
      const header = el("tr");
      const columns = ["Customer / source row", "Churn probability", "Predicted class", "Risk category"];
      if (result.actual_outcomes_included) columns.push("Actual historical label");
      columns.forEach((column) => header.append(el("th", column)));
      head.append(header);
      body.replaceChildren();
      if (!result.results.length) {
        const emptyRow = el("tr");
        const emptyCell = el("td", "No predictions match these filters.");
        emptyCell.colSpan = columns.length;
        emptyRow.append(emptyCell);
        body.append(emptyRow);
      } else {
        result.results.forEach((record) => {
          const row = el("tr");
          row.append(el("td", record.customer_id || `Row ${record.row_number}`));
          row.append(el("td", record.churn_probability === null ? "Unavailable" : `${(record.churn_probability * 100).toFixed(1)}%`));
          row.append(el("td", record.predicted_class));
          row.append(el("td", record.risk_category, `risk-badge ${record.risk_category}`));
          if (result.actual_outcomes_included) row.append(el("td", record.actual_outcome ?? "Missing"));
          body.append(row);
        });
      }
      document.getElementById("results-count").textContent = `${result.total.toLocaleString()} matching rows · page ${result.page} of ${result.pages}`;
      document.getElementById("page-indicator").textContent = `Page ${result.page} of ${pageCount}`;
      document.getElementById("previous-page").disabled = currentPage <= 1;
      document.getElementById("next-page").disabled = currentPage >= pageCount;
    } catch (error) {
      body.replaceChildren();
      const row = el("tr");
      const cell = el("td", error.message);
      cell.colSpan = 5;
      row.append(cell);
      body.append(row);
    }
  }

  document.getElementById("prediction-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    if (!modelSelect.value || !scoringDataset.value) return;
    scoreButton.disabled = true;
    scoreButton.textContent = "Scoring…";
    status(predictionStatus, "Checking feature compatibility and scoring the selected dataset…", "loading");
    predictionSummary.replaceChildren();
    resultsSection.hidden = true;
    try {
      const result = await api("/api/predictions", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model_id: Number(modelSelect.value),
          dataset_id: Number(scoringDataset.value),
          high_risk_threshold: Number(document.getElementById("high-threshold").value),
          medium_risk_threshold: Number(document.getElementById("medium-threshold").value),
        }),
      });
      activeRunId = result.prediction_run.id;
      currentPage = 1;
      const summary = result.prediction_run.summary;
      [[summary.row_count, "Rows scored"], [summary.risk_counts.high, "High risk"], [summary.risk_counts.medium, "Medium risk"], [summary.risk_counts.low, "Low risk"]].forEach(([value, label]) => {
        const chip = el("div", null, "summary-chip");
        chip.append(el("strong", Number(value).toLocaleString()), el("span", label));
        predictionSummary.append(chip);
      });
      const info = summary.actual_outcomes_included
        ? "Historical labels are displayed separately from predictions. These rows are a scoring output, not an unbiased evaluation."
        : "This dataset has no configured historical target; no actual outcomes are shown.";
      status(predictionStatus, `${info} High-risk customers are not guaranteed to leave.`, "success");
      const exportLink = document.getElementById("export-results");
      exportLink.href = `/api/predictions/${activeRunId}/export`;
      exportLink.hidden = false;
      resultsSection.hidden = false;
      await loadPredictionPage();
    } catch (error) {
      status(predictionStatus, error.message, "error");
    } finally {
      scoreButton.disabled = false;
      scoreButton.replaceChildren(document.createTextNode("Generate predictions "), el("span", "→"));
    }
  });

  document.getElementById("previous-page").addEventListener("click", () => { currentPage = Math.max(1, currentPage - 1); loadPredictionPage(); });
  document.getElementById("next-page").addEventListener("click", () => { currentPage = Math.min(pageCount, currentPage + 1); loadPredictionPage(); });
  [riskFilter, resultSort, sortDirection].forEach((control) => control.addEventListener("change", () => { currentPage = 1; loadPredictionPage(); }));
  resultSearch.addEventListener("input", () => {
    window.clearTimeout(searchTimer);
    searchTimer = window.setTimeout(() => { currentPage = 1; loadPredictionPage(); }, 250);
  });

  loadReferences();
})();
