import pandas as pd


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