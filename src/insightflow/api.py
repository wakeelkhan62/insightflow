import io

import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile

from insightflow.ingestion import load_csv
from insightflow.quality import build_quality_report

MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB

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
        raise HTTPException(status_code=413, detail="File is larger than 10 MB")
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
