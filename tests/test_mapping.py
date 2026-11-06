import pandas as pd
import pytest

from insightflow.mapping import apply_column_mapping, suggest_column_mapping


def test_maps_different_column_names_to_standard_fields():
    """Names like 'Units Sold' and 'Rate (PKR)' map to standard fields."""
    df = pd.DataFrame(
        columns=["Invoice No", "Invoice Date", "Item", "Units Sold", "Rate (PKR)"]
    )

    assert suggest_column_mapping(df) == {
        "order_id": "Invoice No",
        "order_date": "Invoice Date",
        "product": "Item",
        "quantity": "Units Sold",
        "unit_price": "Rate (PKR)",
    }


def test_unknown_columns_are_left_out():
    """Columns with no known name are not guessed."""
    df = pd.DataFrame(columns=["Qty", "Customer", "Notes"])

    assert suggest_column_mapping(df) == {"quantity": "Qty"}


def test_renames_mapped_columns_and_keeps_original_unchanged():
    """Mapped columns get standard names; others and the input stay as they are."""
    df = pd.DataFrame({"Qty": [1], "Rate (PKR)": [10], "Notes": ["x"]})

    result = apply_column_mapping(df, {"quantity": "Qty", "unit_price": "Rate (PKR)"})

    assert list(result.columns) == ["quantity", "unit_price", "Notes"]
    assert list(df.columns) == ["Qty", "Rate (PKR)", "Notes"]


def test_unknown_field_raises_clear_error():
    """A misspelled standard field is rejected."""
    df = pd.DataFrame({"Qty": [1]})

    with pytest.raises(ValueError, match="qty"):
        apply_column_mapping(df, {"qty": "Qty"})


def test_same_column_for_two_fields_raises_clear_error():
    """One column cannot be both quantity and price."""
    df = pd.DataFrame({"Qty": [1]})

    with pytest.raises(ValueError, match="only one field"):
        apply_column_mapping(df, {"quantity": "Qty", "unit_price": "Qty"})


def test_rename_that_creates_duplicate_names_raises_clear_error():
    """Renaming must not produce two columns with the same name."""
    df = pd.DataFrame({"Qty": [1], "quantity": [2]})

    with pytest.raises(ValueError, match="duplicate"):
        apply_column_mapping(df, {"quantity": "Qty"})
