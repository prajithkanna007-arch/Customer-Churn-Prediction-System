# Configurable Customer Churn Prediction and Retention System

Phases 1–3 provide the Flask application shell, SQLite persistence, configurable CSV upload and quality review, supervised baseline training, evaluation, saved model pipelines, and schema-checked prediction runs. Churn analysis, retention workflows, and action tracking are not implemented. The application never fabricates customers, labels, predictions, or metrics.

## Requirements

- Windows 10/11
- Python 3.11 (recommended project environment)
- VS Code (recommended)

## Setup (PowerShell)

```powershell
py -3.11 -m venv .venv311
.\.venv311\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, use `\.venv311\Scripts\python.exe` in place of `python` in the commands below.

## Run the application

```powershell
python run.py
```

Open http://127.0.0.1:5000. The local SQLite database is created under `instance/churn.sqlite3` on first run. Check application liveness at http://127.0.0.1:5000/health.

Uploaded CSV files are stored under `instance/uploads/` and model pipeline artifacts under `instance/models/`, both with generated filenames. Keep these directories private; they and the local database are excluded from Git.

## Run tests

```powershell
python -m pytest -q
```

## CSV workflow

1. Open **Data Management** and choose a `.csv` file up to 20 MB.
2. Review the bounded row preview, inferred data types, missing values, duplicate rows, high-cardinality columns, and any normalized headers.
3. Select a dataset to review or change its column mappings. Customer ID suggestions use only generic identifier names. No churn target or target classes are inferred.
4. Select and save a target plus observed positive and negative classes only if the dataset has historical labels. Without a target, the dataset is classified as unlabeled (features selected) or descriptive-only (no features selected).
5. Delete a dataset from its detail panel; its metadata and generated CSV file are removed.

Upload, preview, quality, configuration, and delete operations are available through `/api/datasets` endpoints. The API returns aggregate quality findings and at most 15 preview rows, not the entire dataset.

## Training and prediction

1. Configure a labeled historical dataset with a target, two observed positive/negative classes, and one or more feature columns.
2. Open **Churn Prediction**, choose the labeled dataset, and train both Logistic Regression and Random Forest baselines.
3. Evaluation uses separate train, validation, and held-out test splits. If a date column is configured, the split is chronological; otherwise, it is stratified and reproducible. The baselines are compared on validation positive-class F1 at a 0.5 threshold. The held-out test split is reserved for final reporting.
4. Review precision, recall, F1, accuracy, ROC-AUC where valid, class distributions, confusion matrices, and split details under **Reports & Model Performance**. Accuracy alone may mislead on imbalanced data.
5. Select a saved model and a dataset with the same feature columns, order, and compatible feature types to generate predictions. Unseen categorical values are handled by the fitted one-hot encoder. Unlabeled datasets do not show actual outcomes.
6. Prediction runs are stored as individual database records and can be filtered, sorted, paginated, and exported to CSV. High and medium risk thresholds default to 0.70 and 0.40; they are operational defaults, not guarantees. A high-risk score does not mean a customer will definitely leave.

Model and prediction APIs are under `/api/models` and `/api/predictions`. Model artifacts are loaded only from the configured application model directory; arbitrary uploaded joblib files are never accepted.

## Pages

Dashboard, Data Management, Customer Directory, Churn Prediction, Churn Analysis, Retention Center, Reports & Model Performance, and Settings share one responsive application layout. Pages without implemented functionality retain honest empty states.

## Planned architecture

Later phases can add model explanations, churn analytics, retention recommendations and action tracking, and broader reports/exports. Historical labels remain separate from model predictions, and supervised metrics are only calculated from eligible labeled data.
