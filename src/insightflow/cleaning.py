from datetime import date

import pandas as pd
from pandas.api import types as ptypes

from insightflow.quality import detect_date_columns


def convert_date_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Return a copy of df with the approved columns converted to dates.

    The input DataFrame is not changed. Raises ValueError if a column is
    missing or if any non-empty value cannot be read as a date.
    """
    unknown = [column for column in columns if column not in df.columns]
    if unknown:
        raise ValueError(f"Columns not found in dataset: {unknown}")

    converted = df.copy()
    for column in columns:
        original = df[column]
        parsed = pd.to_datetime(original, errors="coerce", format="ISO8601")
        failed = int((original.notna() & parsed.isna()).sum())
        if failed:
            raise ValueError(
                f"Column '{column}' has {failed} value(s) that are not valid dates"
            )
        converted[column] = parsed
    return converted


def remove_duplicate_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (cleaned, removed): data without extra copies, and the copies removed.

    The first copy of each row is kept. The input DataFrame is not changed.
    """
    is_duplicate = df.duplicated(keep="first")
    return df[~is_duplicate].copy(), df[is_duplicate].copy()


FILL_STRATEGIES = {"mean", "median", "mode", "unknown"}
NUMERIC_STRATEGIES = {"mean", "median"}
TEXT_STRATEGIES = {"unknown"}


def _fill_value(values: pd.Series, strategy: str):
    """Return the value to fill with, or None if there is no reliable source."""
    if strategy == "unknown":
        return "Unknown"
    values = values.dropna()
    if values.empty:
        return None
    if strategy == "mode":
        return values.mode().iloc[0]
    if strategy == "mean":
        return values.mean()
    return values.median()


def _validate_fill_request(
    df: pd.DataFrame, strategies: dict[str, str], group_by: str | None
) -> None:
    """Raise ValueError if a column, strategy or group column is invalid."""
    for column, strategy in strategies.items():
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found in dataset")
        if strategy not in FILL_STRATEGIES:
            raise ValueError(f"Unknown strategy '{strategy}' for column '{column}'")
        is_numeric = ptypes.is_numeric_dtype(df[column])
        if strategy in NUMERIC_STRATEGIES and not is_numeric:
            raise ValueError(
                f"Strategy '{strategy}' needs a numeric column, "
                f"but '{column}' is not numeric"
            )
        if strategy in TEXT_STRATEGIES and is_numeric:
            raise ValueError(
                f"Strategy '{strategy}' needs a text column, but '{column}' is numeric"
            )
    if group_by is not None and group_by not in df.columns:
        raise ValueError(f"Column '{group_by}' not found in dataset")


def fill_missing_values(
    df: pd.DataFrame, strategies: dict[str, str], group_by: str | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (filled, audit): a copy with gaps filled, and a record of each fill.

    strategies maps column -> "mean", "median", "mode" or "unknown". With
    group_by, each gap is filled from rows in the same group only. Gaps with
    no source values stay missing. Each filled column gets a
    "<column>_was_filled" flag column. The input DataFrame is not changed.
    """
    _validate_fill_request(df, strategies, group_by)

    filled = df.copy()
    audit = []
    for column, strategy in strategies.items():
        was_filled = pd.Series(False, index=df.index)
        for row in df.index[df[column].isna()]:
            if group_by is None:
                source = df[column]
            else:
                source = df.loc[df[group_by] == df.at[row, group_by], column]
            value = _fill_value(source, strategy)
            if value is None:
                continue
            if ptypes.is_integer_dtype(df[column]):
                value = round(value)
            filled.at[row, column] = value
            was_filled[row] = True
            audit.append(
                {"row": row, "column": column, "strategy": strategy, "value": value}
            )
        filled[f"{column}_was_filled"] = was_filled
    columns = ["row", "column", "strategy", "value"]
    return filled, pd.DataFrame(audit, columns=columns)


def _looks_like_id(column: str) -> bool:
    """Return True if the column name suggests an identifier, e.g. 'order_id'."""
    name = column.strip().lower()
    return name == "id" or name.endswith(("_id", " id"))


def suggest_fill_strategies(
    df: pd.DataFrame, max_missing_share: float = 0.4
) -> dict[str, str]:
    """Suggest a fill strategy for every column that has missing values.

    Returns column -> "median", "unknown", "do_not_fill" or "too_many_missing".
    These are suggestions for the user to approve; nothing is changed.
    """
    text_dates = set(detect_date_columns(df))
    suggestions = {}
    for column in df.columns:
        series = df[column]
        name = str(column)
        missing_share = series.isna().mean()
        if missing_share == 0:
            continue
        if missing_share > max_missing_share:
            suggestions[name] = "too_many_missing"
        elif (
            _looks_like_id(name)
            or name in text_dates
            or ptypes.is_datetime64_any_dtype(series)
            or ptypes.is_bool_dtype(series)
        ):
            suggestions[name] = "do_not_fill"
        elif ptypes.is_numeric_dtype(series):
            suggestions[name] = "median"
        else:
            suggestions[name] = "unknown"
    return suggestions


NON_REVENUE_STATUSES = frozenset(
    {
        "cancelled",
        "canceled",
        "returned",
        "refunded",
        "failed",
        "rejected",
        "declined",
        "void",
        "voided",
    }
)


def _normalized_status(series: pd.Series) -> pd.Series:
    """Lower-case and trim status text so 'Cancelled ' matches 'cancelled'."""
    return series.astype("string").str.strip().str.lower()


def suggest_excluded_statuses(df: pd.DataFrame) -> list[str]:
    """Return the statuses in df that usually mean no money was received.

    Uses the exact spelling found in the data, e.g. ["Cancelled", "Returned"].
    Returns [] if there is no order_status column. Nothing is changed.
    """
    if "order_status" not in df.columns:
        return []
    statuses = df["order_status"].dropna().astype("string").unique()
    found = [s for s in statuses if s.strip().lower() in NON_REVENUE_STATUSES]
    return sorted(found)


def split_by_order_status(
    df: pd.DataFrame, excluded_statuses: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (kept, excluded): rows to count as revenue, and rows left out.

    excluded_statuses is the list the user approved; matching ignores case and
    surrounding spaces. Rows with no status are kept. The input is not changed.
    """
    if "order_status" not in df.columns:
        raise ValueError("No order_status column. Map the status column first.")
    targets = {status.strip().lower() for status in excluded_statuses}
    is_excluded = _normalized_status(df["order_status"]).isin(targets)
    is_excluded = is_excluded.fillna(False).astype(bool)
    return df[~is_excluded].copy(), df[is_excluded].copy()


def split_future_dates(
    df: pd.DataFrame, today: date | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (kept, future): rows dated up to today, and rows dated after it.

    A sale cannot happen in the future, so these are almost always typing
    errors; one of them would stretch a monthly trend into empty future months.
    Rows without a readable date are kept. The input is not changed.
    """
    if "order_date" not in df.columns:
        raise ValueError("No order_date column. Map the date column first.")
    today = today or date.today()
    dates = pd.to_datetime(df["order_date"], errors="coerce", format="ISO8601")
    is_future = (dates > pd.Timestamp(today)).fillna(False).astype(bool)
    return df[~is_future].copy(), df[is_future].copy()


def _text_columns_to_standardize(df: pd.DataFrame) -> list[str]:
    """Text columns that hold categories: not IDs and not dates."""
    text_dates = set(detect_date_columns(df))
    return [
        str(column)
        for column in df.columns
        if (ptypes.is_string_dtype(df[column]) or ptypes.is_object_dtype(df[column]))
        and not _looks_like_id(str(column))
        and str(column) not in text_dates
    ]


def standardize_text_values(
    df: pd.DataFrame, columns: list[str] | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (standardized, merges): text with consistent spelling, and a record.

    Extra spaces are removed, and values that differ only in capitals
    ("CANCELLED", "cancelled ", "Cancelled") become the most common spelling
    in the file (alphabetical order breaks ties). By default every text column
    except IDs and dates is processed. merges lists each standard value that
    absorbed other spellings. Missing values stay missing. The input is not
    changed.
    """
    columns = _text_columns_to_standardize(df) if columns is None else columns
    result = df.copy()
    records = []
    for column in columns:
        original = df[column].astype("string")
        trimmed = original.str.strip().str.replace(r"\s+", " ", regex=True)
        key = trimmed.str.casefold()
        rows = pd.DataFrame({"key": key, "value": trimmed, "original": original})
        rows = rows.dropna(subset=["key"])
        counts = rows.groupby(["key", "value"]).size().reset_index(name="n")
        counts = counts.sort_values(
            ["key", "n", "value"], ascending=[True, False, True]
        )
        standard = counts.drop_duplicates("key").set_index("key")["value"]
        result[column] = key.map(standard).astype("string")

        rows["standard"] = rows["key"].map(standard)
        changed = rows[rows["original"] != rows["standard"]]
        for value, group in changed.groupby("standard"):
            records.append(
                {
                    "column": column,
                    "standard_value": value,
                    "merged_spellings": sorted(group["original"].unique()),
                    "rows_changed": len(group),
                }
            )
    merge_columns = ["column", "standard_value", "merged_spellings", "rows_changed"]
    return result, pd.DataFrame(records, columns=merge_columns)
