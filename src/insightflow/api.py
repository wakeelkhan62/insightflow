import io
import json
import os

import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile

from insightflow.analytics import revenue_by_product, revenue_summary
from insightflow.cleaning import remove_duplicate_rows
from insightflow.ingestion import load_csv
from insightflow.mapping import suggest_column_mapping
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


@app.post("/datasets/revenue")
async def revenue(file: UploadFile, remove_duplicates: bool = True) -> dict:
    """Upload a CSV with standard column names and get revenue with evidence.

    Exact duplicate rows are removed only when remove_duplicates is true, and
    the number removed is reported. If standard columns are missing, the error
    includes a suggested column mapping.
    """
    df = await _read_csv_upload(file)
    removed_count = 0
    if remove_duplicates:
        df, removed = remove_duplicate_rows(df)
        removed_count = len(removed)
    try:
        summary = revenue_summary(df)
        by_product = revenue_by_product(df)
    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail={
                "message": str(error),
                "suggested_mapping": suggest_column_mapping(df),
            },
        ) from error
    return {
        "duplicates_removed": removed_count,
        "summary": summary,
        "by_product": _records(by_product),
    }
