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


def _outlier_mask(values: pd.Series) -> pd.Series:
    """Flag values far outside the middle 50% (IQR rule) of their own group."""
    numbers = values.astype(float)
    if numbers.notna().sum() < MIN_ROWS_FOR_OUTLIERS:
        return pd.Series(False, index=values.index)
    q1, q3 = numbers.quantile(0.25), numbers.quantile(0.75)
    spread = OUTLIER_FACTOR * (q3 - q1)
    return ((numbers < q1 - spread) | (numbers > q3 + spread)).fillna(False)


def _check_outliers(df: pd.DataFrame, column: str) -> dict:
    """Flag extreme values, compared within each product when possible.

    A Rs 185,000 laptop is normal for laptops but extreme next to a Rs 1,200
    phone stand, so prices are compared per product when a product column
    exists. Groups with fewer than MIN_ROWS_FOR_OUTLIERS values are not judged.
    """
    name = f"{column}_extreme_value"
    if column not in df.columns:
        return _skipped(name, f"No {column} column")
    if not ptypes.is_numeric_dtype(df[column]):
        return _skipped(name, f"{column} is not numeric")
    if df[column].notna().sum() < MIN_ROWS_FOR_OUTLIERS:
        return _skipped(name, f"Fewer than {MIN_ROWS_FOR_OUTLIERS} values")
    if "product" in df.columns:
        groups = df.groupby("product", dropna=False)[column]
        mask = groups.transform(_outlier_mask).astype(bool)
        rule = f"{column} far outside the usual range for that product"
    else:
        mask = _outlier_mask(df[column]).astype(bool)
        rule = f"{column} far outside the usual range"
    return _result(name, rule, mask)


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
