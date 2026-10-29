import pandas as pd
import pytest

from insightflow.quality import (
    count_duplicate_keys,
    count_duplicate_rows,
    find_missing_values,
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