import pandas as pd
import pytest

from insightflow.analytics import (
    add_revenue,
    revenue_by_month,
    revenue_by_product,
    revenue_summary,
)


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


def test_revenue_by_product_sorted_with_evidence():
    """Products are ranked by revenue; a product with no usable rows is last."""
    df = pd.DataFrame(
        {
            "product": pd.array(
                ["Laptop", "Mouse", "Keyboard", "Laptop"], dtype="string"
            ),
            "quantity": pd.array([2, 5, None, 1], dtype="Int64"),
            "unit_price": pd.array([85000, 1500, 3000, 85000], dtype="Int64"),
        }
    )

    result = revenue_by_product(df)

    assert list(result["product"]) == ["Laptop", "Mouse", "Keyboard"]
    assert list(result["revenue"].iloc[:2]) == [255000, 7500]
    assert pd.isna(result["revenue"].iloc[2])
    assert list(result["rows_included"]) == [2, 1, 0]
    assert list(result["rows_excluded"]) == [0, 0, 1]


def test_revenue_by_month_fills_gap_months_and_counts_undated_rows():
    """Months are in order, an empty month shows 0, and undated rows are counted."""
    df = pd.DataFrame(
        {
            "order_date": pd.to_datetime(
                ["2026-01-05", "2026-01-20", "2026-03-02", "2026-03-15", None]
            ),
            "quantity": pd.array([2, 1, None, 4, 1], dtype="Int64"),
            "unit_price": pd.array([100, 50, 10, 25, 99], dtype="Int64"),
        }
    )

    monthly, undated = revenue_by_month(df)

    assert list(monthly["month"]) == ["2026-01", "2026-02", "2026-03"]
    assert list(monthly["revenue"]) == [250, 0, 100]
    assert list(monthly["rows_included"]) == [2, 0, 1]
    assert list(monthly["rows_excluded"]) == [0, 0, 1]
    assert undated == 1


def test_revenue_by_month_requires_converted_dates():
    """Text dates are rejected with a hint to convert them first."""
    df = pd.DataFrame(
        {"order_date": ["2026-01-05"], "quantity": [1], "unit_price": [1]}
    )

    with pytest.raises(ValueError, match="Convert it first"):
        revenue_by_month(df)
