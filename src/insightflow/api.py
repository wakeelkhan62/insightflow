import io
import json
import os

import pandas as pd
from fastapi import FastAPI, Form, HTTPException, UploadFile

from insightflow.ingestion import load_csv
from insightflow.mapping import (
    STANDARD_FIELDS,
    apply_column_mapping,
    suggest_column_mapping,
)
from insightflow.pipeline import PipelineOptions, run_pipeline
from insightflow.quality import build_quality_report

MAX_UPLOAD_MB = int(os.getenv("INSIGHTFLOW_MAX_UPLOAD_MB", "50"))
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024

app = FastAPI(title="InsightFlow AI", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    """Report that the API is running. Used by hosting platforms and monitors."""
    return {"status": "ok"}


async def _read_csv_upload(file: UploadFile) -> pd.DataFrame:
    """Validate an uploaded CSV and parse it, raising HTTP 400/413 on bad input."""
    if not (file.filename or "").lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only .csv files are accepted")
    content = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        limit_mb = MAX_UPLOAD_BYTES // (1024 * 1024)
        raise HTTPException(
            status_code=413, detail=f"File is larger than the {limit_mb} MB limit"
        )
    if not content.strip():
        raise HTTPException(status_code=400, detail="The file is empty")
    try:
        return load_csv(io.BytesIO(content))
    except (pd.errors.ParserError, UnicodeDecodeError, ValueError) as error:
        raise HTTPException(
            status_code=400, detail="The file could not be read as CSV"
        ) from error


@app.post("/datasets/quality-report")
async def quality_report(file: UploadFile, key_column: str | None = None) -> dict:
    """Upload a CSV and get its data quality report. Nothing is stored or changed."""
    df = await _read_csv_upload(file)
    try:
        return build_quality_report(df, key_column=key_column)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


def _records(df: pd.DataFrame) -> list[dict]:
    """Convert a DataFrame to JSON-safe records (missing values become null)."""
    return json.loads(df.to_json(orient="records"))


def _parse_mapping(mapping: str | None) -> dict[str, str] | None:
    """Parse the user's confirmed column mapping from a JSON text form field."""
    if mapping is None:
        return None
    try:
        parsed = json.loads(mapping)
    except json.JSONDecodeError as error:
        raise HTTPException(
            status_code=400, detail="mapping must be valid JSON"
        ) from error
    if not isinstance(parsed, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in parsed.items()
    ):
        raise HTTPException(
            status_code=400,
            detail='mapping must be an object like {"quantity": "Qty"}',
        )
    return parsed


def _parse_status_list(value: str | None) -> list[str] | None:
    """Parse the confirmed statuses to leave out, sent as a JSON list of text."""
    if value is None:
        return None
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as error:
        raise HTTPException(
            status_code=400, detail="excluded_statuses must be valid JSON"
        ) from error
    if not isinstance(parsed, list) or not all(isinstance(s, str) for s in parsed):
        raise HTTPException(
            status_code=400,
            detail='excluded_statuses must be a list like ["Cancelled"]',
        )
    return parsed


@app.post("/datasets/revenue")
async def revenue(
    file: UploadFile,
    mapping: str | None = Form(None),
    excluded_statuses: str | None = Form(None),
    remove_duplicates: bool = True,
    fix_text: bool = True,
    leave_out_future: bool = True,
    discount_type: str | None = None,
) -> dict:
    """Upload a CSV and get revenue with evidence, using the shared pipeline.

    mapping: optional JSON object (standard field -> column name) confirmed by
    the user, e.g. {"quantity": "Qty"}. Without it, columns that already have
    standard names are used. excluded_statuses: optional JSON list; without it
    the suggested non-revenue statuses (cancelled, returned...) are left out.
    discount_type: fraction, percent or amount; guessed when not given.
    If quantity or unit_price cannot be found, the error includes a suggested
    column mapping.
    """
    df = await _read_csv_upload(file)
    confirmed = _parse_mapping(mapping)
    if confirmed is not None:
        try:
            apply_column_mapping(df, confirmed)  # report a wrong mapping first
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
    used_mapping = confirmed or {
        f: f for f in STANDARD_FIELDS if f in map(str, df.columns)
    }
    if "quantity" not in used_mapping or "unit_price" not in used_mapping:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Missing required columns: quantity and unit_price. "
                "Apply the column mapping first.",
                "suggested_mapping": suggest_column_mapping(df),
            },
        )
    options = PipelineOptions(
        mapping=used_mapping,
        fix_text=fix_text,
        remove_duplicates=remove_duplicates,
        excluded_statuses=_parse_status_list(excluded_statuses),
        leave_out_future=leave_out_future,
        discount_type=discount_type,
    )
    try:
        result = run_pipeline(df, options)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    by_month = None
    if result.monthly is not None:
        by_month = {
            "months": _records(result.monthly),
            "undated_rows": result.undated_rows,
            "yearly": _records(result.yearly),
            "forecast": result.forecast,
        }
    return {
        "mapping_applied": confirmed or {},
        "duplicates_removed": result.duplicates_removed,
        "text_values_fixed": (
            int(result.text_merges["rows_changed"].sum())
            if not result.text_merges.empty
            else 0
        ),
        "excluded_statuses": result.excluded_statuses,
        "status_excluded": result.status_excluded,
        "future_left_out": result.future_left_out,
        "gross_total": result.gross_total,
        "discount": result.discount,
        "summary": result.summary,
        "kpis": result.kpis,
        "reconciliation": result.reconciliation,
        "checks": result.checks,
        "by_product": (
            _records(result.by_product) if result.by_product is not None else None
        ),
        "by_month": by_month,
    }
