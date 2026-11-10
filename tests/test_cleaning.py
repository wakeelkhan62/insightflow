from datetime import date

import pandas as pd
import pytest
from pandas.api import types as ptypes

from insightflow.cleaning import (
    convert_date_columns,
    fill_missing_values,
    remove_duplicate_rows,
    split_by_order_status,
    split_future_dates,
    standardize_text_values,
    suggest_excluded_statuses,
    suggest_fill_strategies,
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


def test_suggests_a_safe_strategy_for_each_column_with_gaps():
    """IDs and dates are never filled, sparse columns are flagged, others get
    median (numbers) or unknown (text). Complete columns are skipped."""
    df = pd.DataFrame(
        {
            "order_id": pd.array([1, None, 3, 4, 5], dtype="Int64"),
            "order_date": pd.array(
                ["2026-01-05", None, "2026-01-07", "2026-01-08", "2026-01-09"],
                dtype="string",
            ),
            "unit_price": pd.array([100, None, 300, 400, 500], dtype="Int64"),
            "city": pd.array(
                ["Lahore", None, "Karachi", "Lahore", "Multan"], dtype="string"
            ),
            "discount": pd.array([None, None, None, 5, 10], dtype="Int64"),
            "product": pd.array(["A", "B", "C", "D", "E"], dtype="string"),
        }
    )

    assert suggest_fill_strategies(df) == {
        "order_id": "do_not_fill",
        "order_date": "do_not_fill",
        "unit_price": "median",
        "city": "unknown",
        "discount": "too_many_missing",
    }


def _orders_with_status() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "order_id": [1, 2, 3, 4, 5],
            "order_status": pd.array(
                ["Delivered", "Cancelled ", "RETURNED", None, "Shipped"],
                dtype="string",
            ),
        }
    )


def test_suggests_non_revenue_statuses_using_the_files_own_spelling():
    """Cancelled and returned are suggested; delivered and shipped are not."""
    assert suggest_excluded_statuses(_orders_with_status()) == [
        "Cancelled ",
        "RETURNED",
    ]


def test_split_by_order_status_ignores_case_and_keeps_rows_without_status():
    """Approved statuses are excluded; a missing status is kept and counted."""
    df = _orders_with_status()

    kept, excluded = split_by_order_status(df, ["cancelled", "Returned"])

    assert list(kept["order_id"]) == [1, 4, 5]
    assert list(excluded["order_id"]) == [2, 3]
    assert len(df) == 5


def test_no_status_column_means_no_suggestions_and_a_clear_error():
    """Files without a status column still work; splitting explains why not."""
    df = pd.DataFrame({"order_id": [1]})

    assert suggest_excluded_statuses(df) == []
    with pytest.raises(ValueError, match="order_status"):
        split_by_order_status(df, ["Cancelled"])


def test_split_future_dates_keeps_past_and_undated_rows():
    """Orders dated after today are separated; past and missing dates are kept."""
    df = pd.DataFrame(
        {
            "order_id": [1, 2, 3, 4],
            "order_date": ["2026-11-10", "2027-03-15", None, "2025-01-01"],
        }
    )

    kept, future = split_future_dates(df, today=date(2026, 11, 10))

    assert list(kept["order_id"]) == [1, 3, 4]
    assert list(future["order_id"]) == [2]
    assert len(df) == 4


def test_standardize_merges_spellings_into_the_most_common_one():
    """Case and spacing differences are merged; every merge is recorded."""
    df = pd.DataFrame(
        {
            "order_id": ["inv-1", "INV-1", "inv-2", "inv-3", "inv-4"],
            "status": pd.array(
                ["Cancelled", "CANCELLED", "cancelled ", "Cancelled", None],
                dtype="string",
            ),
            "city": pd.array(
                ["Lahore", "lahore", "Lahore", "  Karachi", "Karachi"],
                dtype="string",
            ),
        }
    )

    result, merges = standardize_text_values(df)

    assert list(result["status"].iloc[:4]) == ["Cancelled"] * 4
    assert pd.isna(result["status"].iloc[4])
    assert list(result["city"]) == ["Lahore", "Lahore", "Lahore", "Karachi", "Karachi"]
    assert list(result["order_id"]) == list(df["order_id"])
    status = merges[merges["column"] == "status"].iloc[0]
    assert status["merged_spellings"] == ["CANCELLED", "cancelled "]
    assert status["rows_changed"] == 2
    assert df["status"].iloc[1] == "CANCELLED"


def test_standardize_on_clean_text_changes_nothing():
    """Already consistent text produces no merges."""
    df = pd.DataFrame({"city": pd.array(["Lahore", "Karachi"], dtype="string")})

    result, merges = standardize_text_values(df)

    assert list(result["city"]) == ["Lahore", "Karachi"]
    assert merges.empty
