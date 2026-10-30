import pandas as pd
import pytest

from insightflow.quality import (
    build_quality_report,
    count_duplicate_keys,
    count_duplicate_rows,
    detect_date_columns,
    find_missing_values,
    profile_column_types,
)


def test_reports_missing_count_for_every_column():
    """Every column is listed, with the right count of missing cells."""
    df = pd.DataFrame(
        {
            "product": ["Laptop", "Mouse", "Keyboard"],
            "quantity": [2, None, None],
            "price": [100, 200, None],
        }
    )

    assert find_missing_values(df) == {"product": 0, "quantity": 2, "price": 1}


def test_clean_data_reports_zero_for_every_column():
    """Clean data gives 0 for every column, with no false alarms."""
    df = pd.DataFrame({"product": ["Laptop"], "quantity": [2]})

    assert find_missing_values(df) == {"product": 0, "quantity": 0}


def test_counts_extra_copies_of_duplicate_rows():
    """Three identical rows count as 2 duplicates; the first is the original."""
    df = pd.DataFrame(
        {
            "order_id": [1, 2, 2, 2, 3],
            "product": ["Laptop", "Mouse", "Mouse", "Mouse", "Pen"],
        }
    )

    assert count_duplicate_rows(df) == 2


def test_same_id_but_different_data_is_not_a_full_duplicate():
    """Same order_id with different data is not an exact duplicate row."""
    df = pd.DataFrame({"order_id": [1, 1], "product": ["Laptop", "Mouse"]})

    assert count_duplicate_rows(df) == 0


def test_counts_reused_keys_even_when_rows_differ():
    """Same order_id with different products counts as a duplicate key."""
    df = pd.DataFrame({"order_id": [1, 1, 2], "product": ["Laptop", "Mouse", "Pen"]})

    assert count_duplicate_keys(df, "order_id") == 1


def test_missing_key_column_raises_clear_error():
    """A wrong column name fails with a message naming that column."""
    df = pd.DataFrame({"order_id": [1, 2]})

    with pytest.raises(ValueError, match="orderid"):
        count_duplicate_keys(df, "orderid")


def test_profiles_each_column_with_a_simple_type_label():
    """Each column gets a business-friendly type label."""
    df = pd.DataFrame(
        {
            "order_id": pd.array([1, 2], dtype="Int64"),
            "price": pd.array([1.5, None], dtype="Float64"),
            "product": pd.array(["Laptop", "Mouse"], dtype="string"),
            "paid": pd.array([True, None], dtype="boolean"),
            "order_date": pd.to_datetime(["2026-01-05", "2026-01-06"]),
        }
    )

    assert profile_column_types(df) == {
        "order_id": "integer",
        "price": "decimal",
        "product": "text",
        "paid": "boolean",
        "order_date": "datetime",
    }


def test_detects_text_columns_that_contain_dates():
    """Only text columns that are mostly valid dates are suggested."""
    df = pd.DataFrame(
        {
            "order_id": pd.array([2026, 2025, 2024], dtype="Int64"),
            "order_date": pd.array(["2026-01-05", "2026-01-06", None], dtype="string"),
            "product": pd.array(["Laptop", "Mouse", "Pen"], dtype="string"),
            "note": pd.array(["2026-01-05", "late", "call back"], dtype="string"),
        }
    )

    assert detect_date_columns(df) == ["order_date"]


def test_quality_report_does_not_double_count_duplicates():
    """An exact copy is a duplicate row, not also a conflicting key."""
    df = pd.DataFrame(
        {
            "order_id": [1, 1, 1, 2],
            "product": ["Laptop", "Laptop", "Mouse", "Pen"],
        }
    )

    report = build_quality_report(df, key_column="order_id")

    assert report["row_count"] == 4
    assert report["column_count"] == 2
    assert report["duplicate_rows"] == 1
    assert report["conflicting_keys"] == 1


def test_quality_report_skips_key_check_when_no_key_given():
    """Without a key column, the report has no conflicting_keys entry."""
    df = pd.DataFrame({"product": ["Laptop", "Mouse"]})

    report = build_quality_report(df)

    assert "conflicting_keys" not in report
