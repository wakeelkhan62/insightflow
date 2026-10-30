import pandas as pd
import pytest
from pandas.api import types as ptypes

from insightflow.cleaning import convert_date_columns, remove_duplicate_rows


def test_converts_approved_column_and_keeps_original_unchanged():
    """The approved column becomes a date; the input DataFrame is untouched."""
    df = pd.DataFrame(
        {
            "order_date": pd.array(["2026-01-05", None], dtype="string"),
            "product": pd.array(["Laptop", "Mouse"], dtype="string"),
        }
    )

    result = convert_date_columns(df, ["order_date"])

    assert ptypes.is_datetime64_any_dtype(result["order_date"])
    assert result["order_date"].isna().sum() == 1
    assert str(df["order_date"].dtype) == "string"


def test_unknown_column_raises_clear_error():
    """Asking to convert a column that does not exist fails clearly."""
    df = pd.DataFrame({"order_date": pd.array(["2026-01-05"], dtype="string")})

    with pytest.raises(ValueError, match="date"):
        convert_date_columns(df, ["date"])


def test_invalid_date_values_are_reported_not_silently_lost():
    """A value that is not a date stops the conversion with a count."""
    df = pd.DataFrame(
        {"order_date": pd.array(["2026-01-05", "soon", None], dtype="string")}
    )

    with pytest.raises(ValueError, match="1 value"):
        convert_date_columns(df, ["order_date"])


def test_removes_extra_copies_and_returns_removed_rows():
    """Extra copies are removed, the first is kept, and removed rows are returned."""
    df = pd.DataFrame(
        {
            "order_id": [1, 2, 2, 2, 3],
            "product": ["Laptop", "Mouse", "Mouse", "Mouse", "Pen"],
        }
    )

    cleaned, removed = remove_duplicate_rows(df)

    assert list(cleaned["order_id"]) == [1, 2, 3]
    assert list(removed.index) == [2, 3]
    assert len(df) == 5


def test_clean_data_removes_nothing():
    """Data without duplicates comes back complete, with nothing removed."""
    df = pd.DataFrame({"order_id": [1, 2], "product": ["Laptop", "Mouse"]})

    cleaned, removed = remove_duplicate_rows(df)

    assert len(cleaned) == 2
    assert removed.empty
