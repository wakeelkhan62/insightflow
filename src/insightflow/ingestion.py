from pathlib import Path

import pandas as pd


def load_csv(path: str | Path) -> pd.DataFrame:
    """Read a CSV file into a DataFrame without changing its data."""
    return pd.read_csv(path)