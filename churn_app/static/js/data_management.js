"use strict";

(() => {
  const uploadForm = document.getElementById("upload-form");
  if (!uploadForm) return;

  const listNode = document.getElementById("dataset-list");
  const countNode = document.getElementById("dataset-count");
  const detailNode = document.getElementById("dataset-detail");
  const statusNode = document.getElementById("upload-status");
  const fileInput = document.getElementById("csv-file");
  const fileName = document.getElementById("file-name");
  const uploadButton = document.getElementById("upload-button");
  let datasets = [];
  let selectedId = null;
  let selectedDetail = null;

  function node(tag, text, className) {
    const element = document.createElement(tag);
    if (text !== undefined && text !== null) element.textContent = String(text);
    if (className) element.className = className;
    return element;
  }

  async function request(url, options = {}) {
    const response = await fetch(url, options);
    const body = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(body.error || `Request failed (${response.status}).`);
    return body;
  }

  function showStatus(message, kind = "info") {
    statusNode.hidden = !message;
    statusNode.textContent = message || "";
    statusNode.className = `upload-status ${kind}`;
  }

  function datasetTypeLabel(type) {
    return ({
      labeled_historical: "Labeled historical",
      unlabeled: "Unlabeled",
      descriptive_only: "Descriptive only",
      unconfigured: "Not configured",
    })[type] || "Not configured";
  }

  function renderList() {
    listNode.replaceChildren();
    countNode.textContent = `${datasets.length} ${datasets.length === 1 ? "dataset" : "datasets"}`;
    if (!datasets.length) {
      listNode.append(node("div", "No datasets yet. Upload a CSV to get started.", "list-empty"));
      return;
    }
    for (const dataset of datasets) {
      const button = node("button", null, `dataset-item${dataset.id === selectedId ? " selected" : ""}`);
      button.type = "button";
      button.setAttribute("aria-pressed", String(dataset.id === selectedId));
      const glyph = node("span", "▤", "dataset-glyph");
      const text = node("span", null, "dataset-item-copy");
      text.append(node("strong", dataset.original_filename), node("small", `${dataset.row_count.toLocaleString()} rows · ${dataset.column_count} columns`));
      const chevron = node("span", "›", "dataset-chevron");
      button.append(glyph, text, chevron);
      button.addEventListener("click", () => selectDataset(dataset.id));
      listNode.append(button);
    }
  }

  async function refreshList() {
    listNode.replaceChildren(node("div", "Loading saved datasets…", "inline-loading"));
    try {
      const result = await request("/api/datasets");
      datasets = result.datasets;
      renderList();
      if (selectedId && datasets.some((item) => item.id === selectedId)) await selectDataset(selectedId);
      else if (!datasets.length) renderEmptyDetail();
    } catch (error) {
      listNode.replaceChildren(node("div", error.message, "inline-error"));
      countNode.textContent = "Could not load datasets";
    }
  }

  function renderEmptyDetail() {
    selectedId = null;
    detailNode.replaceChildren();
    const empty = node("div", null, "detail-empty");
    empty.append(node("div", "▤", "detail-empty-icon"), node("h2", "Select a dataset to inspect it"), node("p", "Row previews, quality findings, and column configuration will appear here."));
    detailNode.append(empty);
    renderList();
  }

  async function selectDataset(id) {
    selectedId = id;
    renderList();
    detailNode.replaceChildren(node("div", null, "inline-loading detail-loading").appendChild(node("span", null, "spinner")).parentElement);
    try {
      const [preview, qualityResult] = await Promise.all([
        request(`/api/datasets/${id}/preview`),
        request(`/api/datasets/${id}/quality`),
      ]);
      selectedDetail = { ...preview, quality: qualityResult.quality };
      renderDetail(selectedDetail);
    } catch (error) {
      detailNode.replaceChildren(node("div", error.message, "inline-error"));
    }
  }

  function addQuality(report, parent) {
    const metrics = node("div", null, "quality-metrics");
    for (const [label, value] of [
      ["Rows", report.row_count], ["Columns", report.column_count],
      ["Missing cells", report.missing_cells], ["Duplicate rows", report.duplicate_rows],
    ]) {
      const metric = node("div", null, "quality-metric");
      metric.append(node("span", Number(value).toLocaleString(), "quality-number"), node("span", label, "quality-label"));
      metrics.append(metric);
    }
    parent.append(metrics);
    const findings = [];
    if (report.missing_cells) findings.push(`${report.missing_cells} missing cells found`);
    if (report.duplicate_rows) findings.push(`${report.duplicate_rows} duplicate rows found`);
    if (report.high_cardinality_columns.length) findings.push(`High-cardinality columns: ${report.high_cardinality_columns.join(", ")}`);
    if (report.header_changes.length) findings.push("Some headers were trimmed or made unique; review the normalized names below.");
    const box = node("div", null, `quality-findings ${findings.length ? "has-findings" : "clean"}`);
    box.textContent = findings.length ? findings.join(" · ") : "No missing values, duplicate rows, or header changes were detected in these checks.";
    parent.append(box);

    const table = node("table", null, "schema-table");
    const head = node("thead");
    const headerRow = node("tr");
    ["Column", "Type", "Missing", "Distinct"].forEach((label) => headerRow.append(node("th", label)));
    head.append(headerRow);
    const body = node("tbody");
    for (const column of report.columns) {
      const row = node("tr");
      [column.name, column.dtype, column.missing_count.toLocaleString(), column.distinct_count.toLocaleString()].forEach((value) => row.append(node("td", value)));
      body.append(row);
    }
    table.append(head, body);
    const wrapper = node("div", null, "table-scroll");
    wrapper.append(table);
    parent.append(wrapper);
  }

  function addPreview(dataset, parent) {
    const wrap = node("div", null, "table-scroll preview-scroll");
    const table = node("table", null, "preview-table");
    const head = node("thead");
    const headerRow = node("tr");
    dataset.columns.forEach((column) => headerRow.append(node("th", column)));
    head.append(headerRow);
    const body = node("tbody");
    for (const record of dataset.rows) {
      const row = node("tr");
      dataset.columns.forEach((column) => row.append(node("td", record[column] === null ? "—" : record[column])));
      body.append(row);
    }
    table.append(head, body);
    wrap.append(table);
    parent.append(wrap);
    parent.append(node("p", `Showing ${dataset.rows.length} of ${dataset.dataset.row_count.toLocaleString()} rows (maximum preview: 15).`, "table-caption"));
  }

  function addOption(select, value, label) {
    const option = node("option", label);
    option.value = value;
    select.append(option);
  }

  function makeColumnSelect(label, columns, value, options = {}) {
    const wrapper = node("label", null, "mapping-field");
    wrapper.append(node("span", label, "mapping-label"));
    const select = node("select");
    select.name = options.name;
    if (options.multiple) select.multiple = true;
    else addOption(select, "", "Not selected");
    columns.forEach((column) => addOption(select, column, column));
    if (options.multiple) {
      const selectedValues = Array.isArray(value) ? value : [];
      Array.from(select.options).forEach((option) => { option.selected = selectedValues.includes(option.value); });
      select.size = Math.min(Math.max(columns.length, 3), 6);
    } else {
      select.value = value || "";
    }
    wrapper.append(select);
    if (options.hint) wrapper.append(node("small", options.hint, "mapping-hint"));
    return { wrapper, select };
  }

  function classSelect(label, values, value, name) {
    const wrapper = node("label", null, "mapping-field");
    wrapper.append(node("span", label, "mapping-label"));
    const select = node("select");
    select.name = name;
    addOption(select, "", "Choose an observed class");
    values.forEach((item) => addOption(select, item, item));
    select.value = value || "";
    wrapper.append(select);
    return { wrapper, select };
  }

  async function loadTargetValues(target, positive, negative) {
    positive.select.replaceChildren();
    negative.select.replaceChildren();
    addOption(positive.select, "", "Choose an observed class");
    addOption(negative.select, "", "Choose an observed class");
    if (!target.value) return;
    try {
      const query = new URLSearchParams({ target_column: target.value });
      const result = await request(`/api/datasets/${selectedId}/quality?${query}`);
      const values = result.quality.target_values.values;
      values.forEach((value) => {
        addOption(positive.select, value, value);
        addOption(negative.select, value, value);
      });
      if (result.quality.target_values.has_more) {
        showStatus("This target has more than 100 distinct values. A binary churn target is expected; review the source data.", "warning");
      }
    } catch (error) {
      showStatus(error.message, "error");
    }
  }

  function renderMapping(dataset) {
    const section = node("section", null, "mapping-section");
    const heading = node("div", null, "mapping-heading");
    heading.append(node("h3", "Column configuration"), node("p", "Choose what each field means. The target and class values are never guessed."));
    section.append(heading);
    const stored = dataset.dataset.configuration;
    const suggested = dataset.suggested_configuration;
    const mapping = stored || suggested;
    const form = node("form", null, "mapping-form");
    form.noValidate = true;
    const columns = dataset.columns;
    const idField = makeColumnSelect("Customer identifier (optional)", columns, mapping.customer_id, { name: "customer_id" });
    const targetField = makeColumnSelect("Historical churn target (optional)", columns, mapping.target, { name: "target" });
    const posField = classSelect("Positive churn class", [], mapping.positive_class, "positive_class");
    const negField = classSelect("Negative / non-churn class", [], mapping.negative_class, "negative_class");
    const featuresField = makeColumnSelect("Feature columns", columns, mapping.features, { name: "features", multiple: true, hint: "Use Ctrl or Cmd to select multiple columns." });
    const dateField = makeColumnSelect("Date column (optional)", columns, mapping.date, { name: "date" });
    const revenueField = makeColumnSelect("Revenue or customer-value column (optional)", columns, mapping.revenue, { name: "revenue" });
    const excludedField = makeColumnSelect("Exclude from analysis or training", columns, mapping.excluded, { name: "excluded", multiple: true, hint: "Excluded columns cannot also be selected as features." });
    const grid = node("div", null, "mapping-grid");
    [idField.wrapper, targetField.wrapper, posField.wrapper, negField.wrapper, featuresField.wrapper, dateField.wrapper, revenueField.wrapper, excludedField.wrapper].forEach((field) => grid.append(field));
    form.append(grid);
    const actions = node("div", null, "mapping-actions");
    const save = node("button", "Save column configuration", "primary-button");
    save.type = "submit";
    const type = node("span", datasetTypeLabel(dataset.dataset.dataset_type), "dataset-type-chip");
    actions.append(save, type);
    form.append(actions);
    const explanation = node("p", stored?.capability || "Suggested features are preselected for review. No target or churn class is assumed.", "mapping-explanation");
    form.append(explanation);
    const formError = node("p", "", "form-error");
    formError.setAttribute("role", "alert");
    form.append(formError);
    section.append(form);

    targetField.select.addEventListener("change", () => loadTargetValues(targetField.select, posField, negField));
    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      formError.textContent = "";
      save.disabled = true;
      save.textContent = "Saving configuration…";
      const selected = (select) => Array.from(select.selectedOptions).map((option) => option.value);
      const payload = {
        customer_id: idField.select.value || null,
        target: targetField.select.value || null,
        positive_class: posField.select.value || null,
        negative_class: negField.select.value || null,
        features: selected(featuresField.select),
        date: dateField.select.value || null,
        revenue: revenueField.select.value || null,
        excluded: selected(excludedField.select),
      };
      try {
        const result = await request(`/api/datasets/${dataset.dataset.id}/configure`, {
          method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload),
        });
        dataset.dataset = result.dataset;
        selectedDetail = dataset;
        showStatus("Column configuration saved.", "success");
        await refreshList();
      } catch (error) {
        formError.textContent = error.message;
      } finally {
        save.disabled = false;
        save.textContent = "Save column configuration";
      }
    });
    loadTargetValues(targetField.select, posField, negField).then(() => {
      posField.select.value = mapping.positive_class || "";
      negField.select.value = mapping.negative_class || "";
    });
    return section;
  }

  function renderDetail(dataset) {
    detailNode.replaceChildren();
    const top = node("div", null, "detail-top");
    const title = node("div");
    title.append(node("div", "SELECTED DATASET", "eyebrow"), node("h2", dataset.dataset.original_filename, "detail-title"));
    const remove = node("button", "Delete dataset", "danger-button");
    remove.type = "button";
    remove.addEventListener("click", async () => {
      if (!window.confirm(`Delete ${dataset.dataset.original_filename} and its stored CSV? This cannot be undone.`)) return;
      remove.disabled = true;
      try {
        await request(`/api/datasets/${dataset.dataset.id}`, { method: "DELETE" });
        showStatus("Dataset and stored file deleted.", "success");
        selectedId = null;
        selectedDetail = null;
        await refreshList();
        renderEmptyDetail();
      } catch (error) {
        showStatus(error.message, "error");
        remove.disabled = false;
      }
    });
    top.append(title, remove);
    detailNode.append(top);
    const meta = node("div", null, "dataset-meta");
    meta.append(node("span", `${dataset.dataset.row_count.toLocaleString()} rows`), node("span", `${dataset.dataset.column_count} columns`), node("span", `Uploaded ${new Date(dataset.dataset.uploaded_at).toLocaleString()}`), node("span", datasetTypeLabel(dataset.dataset.dataset_type), "dataset-type-chip"));
    detailNode.append(meta);

    const qualitySection = node("section", null, "detail-section");
    qualitySection.append(node("h3", "Data quality"));
    addQuality(dataset.quality, qualitySection);
    detailNode.append(qualitySection);

    const previewSection = node("section", null, "detail-section");
    previewSection.append(node("h3", "Preview"));
    addPreview(dataset, previewSection);
    detailNode.append(previewSection);
    detailNode.append(renderMapping(dataset));
  }

  fileInput.addEventListener("change", () => {
    fileName.textContent = fileInput.files.length ? fileInput.files[0].name : "No file selected";
    showStatus("");
  });

  uploadForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    if (!fileInput.files.length) return;
    showStatus("Uploading and validating your CSV…", "loading");
    uploadButton.disabled = true;
    uploadButton.textContent = "Validating…";
    try {
      const result = await request("/api/datasets/upload", { method: "POST", body: new FormData(uploadForm) });
      showStatus("Upload complete. Review the preview, quality findings, and suggested configuration.", "success");
      fileInput.value = "";
      fileName.textContent = "No file selected";
      await refreshList();
      await selectDataset(result.dataset.id);
    } catch (error) {
      showStatus(error.message, "error");
    } finally {
      uploadButton.disabled = false;
      uploadButton.replaceChildren(document.createTextNode("Upload dataset "), node("span", "→"));
    }
  });

  document.getElementById("refresh-datasets").addEventListener("click", refreshList);
  refreshList();
})();
