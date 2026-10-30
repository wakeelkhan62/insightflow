import pandas as pd

from insightflow.mapping import suggest_column_mapping


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
