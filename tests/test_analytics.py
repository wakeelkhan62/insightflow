import pandas as pd
import pytest

from insightflow.analytics import add_revenue, revenue_summary


def _sales() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "product": pd.array(["Laptop", "Mouse", "Keyboard"], dtype="string"),
            "quantity": pd.array([2, 5, None], dtype="Int64"),
            "unit_price": pd.array([85000, 1500, 3000], dtype="Int64"),
        }
    )


def test_adds_revenue_and_keeps_missing_as_missing():
    """Revenue is quantity x price; a missing quantity gives missing revenue, not 0."""
    df = _sales()

    result = add_revenue(df)

    assert list(result["revenue"].iloc[:2]) == [170000, 7500]
    assert pd.isna(result["revenue"].iloc[2])
    assert "revenue" not in df.columns


def test_revenue_summary_reports_total_and_excluded_rows():
    """The total comes with how many rows it is based on and how many were left out."""
    assert revenue_summary(_sales()) == {
        "total_revenue": 177500.0,
        "rows_included": 2,
        "rows_excluded": 1,
    }


def test_missing_standard_columns_raise_clear_error():
    """Without mapped columns, the error says to apply the column mapping."""
    df = pd.DataFrame({"Qty": [1], "Rate": [10]})

    with pytest.raises(ValueError, match="column mapping"):
        add_revenue(df)
