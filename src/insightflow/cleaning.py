import pandas as pd


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
