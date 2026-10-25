"""Safe CSV parsing and data-quality inspection."""

from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


class CSVValidationError(ValueError):
    """A user-correctable CSV validation error."""


@dataclass
class ParsedCSV:
    """A validated in-memory dataset and non-fatal data-quality findings."""

    frame: pd.DataFrame
    encoding: str
    header_changes: list[dict[str, str]]


def normalize_headers(headers: list[str]) -> tuple[list[str], list[dict[str, str]]]:
    """Trim and uniquely normalize headers, returning an audit of any changes."""
    if not headers:
        raise CSVValidationError("The CSV must contain a header row with at least one column.")

    normalized: list[str] = []
    used: set[str] = set()
    changes: list[dict[str, str]] = []
    for index, raw_name in enumerate(headers, start=1):
        base = raw_name.strip()
        if not base:
            base = f"column_{index}"
        candidate = base
        suffix = 2
        while candidate.casefold() in used:
            candidate = f"{base}__{suffix}"
            suffix += 1
        used.add(candidate.casefold())
        normalized.append(candidate)
        if candidate != raw_name:
            changes.append({"original": raw_name, "normalized": candidate})
    return normalized, changes


def parse_csv_bytes(content: bytes) -> ParsedCSV:
    """Parse CSV bytes without executing content and reject malformed/empty files."""
    if not content or not content.strip():
        raise CSVValidationError("The uploaded CSV is empty.")

    decoded: str | None = None
    encoding_used: str | None = None
    for encoding in ("utf-8-sig", "utf-8", "utf-16", "cp1252"):
        try:
            decoded = content.decode(encoding)
            encoding_used = encoding
            break
        except (UnicodeDecodeError, LookupError):
            continue
    if decoded is None or encoding_used is None:
        raise CSVValidationError("The file encoding could not be read. Save it as UTF-8 CSV and try again.")

    try:
        reader = csv.reader(io.StringIO(decoded, newline=""), strict=True)
        headers = next(reader, None)
        if headers is not None:
            for row_number, row in enumerate(reader, start=2):
                if row and len(row) != len(headers):
                    raise CSVValidationError(
                        f"Malformed CSV row {row_number}: expected {len(headers)} fields but found {len(row)}."
                    )
    except csv.Error as exc:
        raise CSVValidationError(f"The CSV header is malformed: {exc}.") from exc
    if headers is None:
        raise CSVValidationError("The uploaded CSV has no header row.")
    names, changes = normalize_headers(headers)

    try:
        frame = pd.read_csv(
            io.BytesIO(content),
            encoding=encoding_used,
            header=0,
            names=names,
            on_bad_lines="error",
            low_memory=False,
        )
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeError, ValueError) as exc:
        raise CSVValidationError(f"The CSV could not be parsed. Check its rows and delimiters. ({exc})") from exc
    if frame.empty:
        raise CSVValidationError("The CSV contains headers but no data rows.")
    if frame.shape[1] == 0:
        raise CSVValidationError("The CSV must contain at least one data column.")
    return ParsedCSV(frame=frame, encoding=encoding_used, header_changes=changes)


def quality_report(frame: pd.DataFrame, header_changes: list[dict[str, str]] | None = None) -> dict[str, Any]:
    """Summarize schema and non-fatal data-quality findings without exposing rows."""
    columns = []
    high_cardinality = []
    for name in frame.columns:
        series = frame[name]
        distinct = int(series.nunique(dropna=True))
        dtype = str(series.dtype)
        columns.append({
            "name": str(name),
            "dtype": dtype,
            "missing_count": int(series.isna().sum()),
            "distinct_count": distinct,
        })
        if distinct > max(20, int(len(frame) * 0.8)):
            high_cardinality.append(str(name))

    return {
        "row_count": int(frame.shape[0]),
        "column_count": int(frame.shape[1]),
        "columns": columns,
        "missing_cells": int(frame.isna().sum().sum()),
        "duplicate_rows": int(frame.duplicated().sum()),
        "high_cardinality_columns": high_cardinality,
        "header_changes": header_changes or [],
    }


def safe_preview(frame: pd.DataFrame, limit: int = 15) -> list[dict[str, Any]]:
    """Convert a small preview to JSON-safe values; never serialize the full dataset."""
    preview = frame.head(limit).astype(object).where(pd.notna(frame.head(limit)), None)
    records: list[dict[str, Any]] = []
    for row in preview.to_dict(orient="records"):
        safe_row = {}
        for key, value in row.items():
            if isinstance(value, (np.integer, np.floating, np.bool_)):
                value = value.item()
            if isinstance(value, float) and not np.isfinite(value):
                value = None
            if hasattr(value, "isoformat"):
                value = value.isoformat()
            safe_row[str(key)] = value
        records.append(safe_row)
    return records


def suggested_mapping(frame: pd.DataFrame) -> dict[str, Any]:
    """Suggest only generic, reviewable mappings; never guess a churn target."""
    columns = [str(column) for column in frame.columns]
    identifier = next((column for column in columns if re.fullmatch(r"(?i:id|[a-z0-9]+_id)", column)), None)
    features = [column for column in columns if column != identifier]
    return {
        "customer_id": identifier,
        "target": None,
        "positive_class": None,
        "negative_class": None,
        "features": features,
        "date": None,
        "revenue": None,
        "excluded": [],
    }


def target_values(frame: pd.DataFrame, column: str, limit: int = 100) -> dict[str, Any]:
    """Return bounded observed target classes for explicit user selection."""
    if column not in frame.columns:
        raise CSVValidationError("The selected target column does not exist in this dataset.")
    values = frame[column].dropna().drop_duplicates()
    return {"values": [str(value) for value in values.iloc[:limit].tolist()], "has_more": len(values) > limit}
