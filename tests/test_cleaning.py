import pandas as pd
import pytest
from pandas.api import types as ptypes

from insightflow.cleaning import (
    convert_date_columns,
    fill_missing_values,
    remove_duplicate_rows,
)


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


def _sales_with_gaps() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "product": pd.array(
                ["Laptop", "Laptop", "Laptop", "Mouse", "Mouse"], dtype="string"
            ),
            "unit_price": pd.array([80000, 90000, None, 1500, None], dtype="Int64"),
        }
    )


def test_fills_median_within_each_product_and_records_audit():
    """Missing prices use the same product's median; every fill is recorded."""
    df = _sales_with_gaps()

    filled, audit = fill_missing_values(
        df, {"unit_price": "median"}, group_by="product"
    )

    assert list(filled["unit_price"]) == [80000, 90000, 85000, 1500, 1500]
    assert list(audit["row"]) == [2, 4]
    assert df["unit_price"].isna().sum() == 2


def test_fills_text_column_with_mode():
    """A text column is filled with its most common value."""
    df = pd.DataFrame(
        {"city": pd.array(["Lahore", None, "Lahore", "Karachi"], dtype="string")}
    )

    filled, _ = fill_missing_values(df, {"city": "mode"})

    assert list(filled["city"]) == ["Lahore", "Lahore", "Lahore", "Karachi"]


def test_mean_on_text_column_raises_clear_error():
    """Mean or median on a text column is rejected."""
    df = _sales_with_gaps()

    with pytest.raises(ValueError, match="numeric"):
        fill_missing_values(df, {"product": "mean"})


def test_group_without_values_stays_missing():
    """If a product has no known price, nothing is invented."""
    df = pd.DataFrame(
        {
            "product": pd.array(["Monitor", "Laptop"], dtype="string"),
            "unit_price": pd.array([None, 80000], dtype="Int64"),
        }
    )

    filled, audit = fill_missing_values(
        df, {"unit_price": "median"}, group_by="product"
    )

    assert filled["unit_price"].isna().sum() == 1
    assert audit.empty


def test_adds_flag_column_marking_filled_cells():
    """A <column>_was_filled column shows exactly which cells were estimated."""
    df = _sales_with_gaps()

    filled, _ = fill_missing_values(df, {"unit_price": "median"}, group_by="product")

    assert list(filled["unit_price_was_filled"]) == [False, False, True, False, True]
    assert "unit_price_was_filled" not in df.columns


def test_fills_text_column_with_unknown():
    """The unknown strategy fills text gaps with 'Unknown' instead of guessing."""
    df = pd.DataFrame({"city": pd.array(["Lahore", None], dtype="string")})

    filled, _ = fill_missing_values(df, {"city": "unknown"})

    assert list(filled["city"]) == ["Lahore", "Unknown"]


def test_unknown_on_numeric_column_raises_clear_error():
    """'Unknown' cannot be put into a number column."""
    df = _sales_with_gaps()

    with pytest.raises(ValueError, match="text column"):
        fill_missing_values(df, {"unit_price": "unknown"})
