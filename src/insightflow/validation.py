"""Business validation rules for sales data.

Every rule runs only if the columns it needs are present, so any file can be
checked. Rows are flagged, never changed or removed: the user decides.
"""

from datetime import date

import pandas as pd
from pandas.api import types as ptypes

MAX_EXAMPLES = 10
OUTLIER_FACTOR = 3.0  # IQR multiplier; 3.0 flags only extreme values
MIN_ROWS_FOR_OUTLIERS = 10


def _file_lines(mask: pd.Series) -> list[int]:
    """Return CSV line numbers (header is line 1) of the first flagged rows."""
    return [int(i) + 2 for i in mask[mask].index[:MAX_EXAMPLES]]


def _result(name: str, rule: str, mask: pd.Series) -> dict:
    return {
        "name": name,
        "rule": rule,
        "rows_flagged": int(mask.sum()),
        "example_file_lines": _file_lines(mask),
    }


def _skipped(name: str, reason: str) -> dict:
    return {"name": name, "reason": reason}


def _check_not_positive(df: pd.DataFrame, column: str) -> dict:
    name = f"{column}_not_positive"
    if column not in df.columns:
        return _skipped(name, f"No {column} column")
    if not ptypes.is_numeric_dtype(df[column]):
        return _skipped(name, f"{column} is not numeric")
    mask = (df[column] <= 0).fillna(False).astype(bool)
    return _result(name, f"{column} <= 0", mask)


def _check_future_dates(df: pd.DataFrame, today: date) -> dict:
    name = "order_date_in_future"
    if "order_date" not in df.columns:
        return _skipped(name, "No order_date column")
    dates = pd.to_datetime(df["order_date"], errors="coerce", format="ISO8601")
    mask = (dates > pd.Timestamp(today)).fillna(False).astype(bool)
    return _result(name, f"order_date after {today.isoformat()}", mask)


def _check_outliers(df: pd.DataFrame, column: str) -> dict:
    name = f"{column}_extreme_value"
    if column not in df.columns:
        return _skipped(name, f"No {column} column")
    if not ptypes.is_numeric_dtype(df[column]):
        return _skipped(name, f"{column} is not numeric")
    values = df[column].dropna().astype(float)
    if len(values) < MIN_ROWS_FOR_OUTLIERS:
        return _skipped(name, f"Fewer than {MIN_ROWS_FOR_OUTLIERS} values")
    q1, q3 = values.quantile(0.25), values.quantile(0.75)
    spread = OUTLIER_FACTOR * (q3 - q1)
    low, high = q1 - spread, q3 + spread
    mask = ((df[column] < low) | (df[column] > high)).fillna(False).astype(bool)
    return _result(name, f"{column} outside {low:,.2f} to {high:,.2f}", mask)


def validate_sales(df: pd.DataFrame, today: date | None = None) -> dict:
    """Run every business rule that the available columns allow.

    Expects standard column names (apply the column mapping first). Returns
    {"checks": [...], "skipped": [...]}: each check gives its rule, how many
    rows broke it, and example CSV line numbers; each skipped rule says why.
    Nothing in df is changed.
    """
    today = today or date.today()
    results = [
        _check_not_positive(df, "quantity"),
        _check_not_positive(df, "unit_price"),
        _check_future_dates(df, today),
        _check_outliers(df, "quantity"),
        _check_outliers(df, "unit_price"),
    ]
    return {
        "checks": [r for r in results if "rule" in r],
        "skipped": [r for r in results if "reason" in r],
    }
