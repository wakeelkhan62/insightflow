from pathlib import Path

import pandas as pd


def load_csv(path: str | Path) -> pd.DataFrame:
    """Read a CSV file into a DataFrame without changing its data.

    Uses nullable types so missing values do not turn integers into decimals.
    """
    return pd.read_csv(path, dtype_backend="numpy_nullable")
