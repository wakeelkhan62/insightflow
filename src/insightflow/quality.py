import pandas as pd
from pandas.api import types as ptypes


def find_missing_values(df: pd.DataFrame) -> dict[str, int]:
    """Return the number of missing cells in every column (0 if none)."""
    missing_counts = df.isna().sum()
    # Convert numpy ints to plain Python ints so the result is JSON-safe.
    return {str(column): int(count) for column, count in missing_counts.items()}


def count_duplicate_rows(df: pd.DataFrame) -> int:
    """Return how many rows are exact copies of an earlier row.

    The first copy counts as the original, so 3 identical rows give 2.
    """
    return int(df.duplicated().sum())


def count_duplicate_keys(df: pd.DataFrame, key_column: str) -> int:
    """Return how many rows reuse a key value that appeared earlier.

    Raises ValueError if key_column is not in the DataFrame.
    """
    if key_column not in df.columns:
        raise ValueError(f"Column '{key_column}' not found in dataset")
    return int(df[key_column].duplicated().sum())


def _type_label(series: pd.Series) -> str:
    """Map a pandas dtype to a simple, business-friendly label."""
    if ptypes.is_bool_dtype(series):
        return "boolean"
    if ptypes.is_integer_dtype(series):
        return "integer"
    if ptypes.is_float_dtype(series):
        return "decimal"
    if ptypes.is_datetime64_any_dtype(series):
        return "datetime"
    return "text"


def profile_column_types(df: pd.DataFrame) -> dict[str, str]:
    """Return a simple type label for every column."""
    return {str(column): _type_label(df[column]) for column in df.columns}


def detect_date_columns(df: pd.DataFrame, min_share: float = 0.9) -> list[str]:
    """Return text columns whose values look like dates (YYYY-MM-DD).

    Only suggests columns; the data itself is not changed.
    """
    date_columns = []
    for column in df.columns:
        series = df[column]
        if _type_label(series) != "text":
            continue
        values = series.dropna().astype("string")
        if values.empty:
            continue
        parsed = pd.to_datetime(values, errors="coerce", format="ISO8601")
        if parsed.notna().mean() >= min_share:
            date_columns.append(str(column))
    return date_columns


def build_quality_report(df: pd.DataFrame, key_column: str | None = None) -> dict:
    """Run every quality check and return the results in one report.

    Exact duplicate rows are removed before counting conflicting keys,
    so the same problem is never counted twice. The input is not changed.
    """
    report = {
        "row_count": len(df),
        "column_count": len(df.columns),
        "missing_values": find_missing_values(df),
        "duplicate_rows": count_duplicate_rows(df),
        "column_types": profile_column_types(df),
        "date_columns": detect_date_columns(df),
    }
    if key_column is not None:
        unique_rows = df.drop_duplicates()
        report["conflicting_keys"] = count_duplicate_keys(unique_rows, key_column)
    return report
