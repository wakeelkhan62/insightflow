from pathlib import Path
from typing import BinaryIO

import pandas as pd


def load_csv(source: str | Path | BinaryIO) -> pd.DataFrame:
    """Read a CSV file (a path or an open binary file) into a DataFrame.

    Uses nullable types so missing values do not turn integers into decimals.
    """
    return pd.read_csv(source, dtype_backend="numpy_nullable")
