import pandas as pd
from pandas.api import types as ptypes


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


FILL_STRATEGIES = {"mean", "median", "mode"}


def _fill_value(values: pd.Series, strategy: str):
    """Return the mean, median or mode of the non-missing values, or None."""
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
        if strategy in {"mean", "median"} and not ptypes.is_numeric_dtype(df[column]):
            raise ValueError(
                f"Strategy '{strategy}' needs a numeric column, "
                f"but '{column}' is not numeric"
            )
    if group_by is not None and group_by not in df.columns:
        raise ValueError(f"Column '{group_by}' not found in dataset")


def fill_missing_values(
    df: pd.DataFrame, strategies: dict[str, str], group_by: str | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (filled, audit): a copy with gaps filled, and a record of each fill.

    strategies maps column -> "mean", "median" or "mode". With group_by, each
    gap is filled from rows in the same group only. Gaps with no source values
    stay missing. The input DataFrame is not changed.
    """
    _validate_fill_request(df, strategies, group_by)

    filled = df.copy()
    audit = []
    for column, strategy in strategies.items():
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
            audit.append(
                {"row": row, "column": column, "strategy": strategy, "value": value}
            )
    columns = ["row", "column", "strategy", "value"]
    return filled, pd.DataFrame(audit, columns=columns)
