import pandas as pd
import pytest

from insightflow.analytics import (
    add_revenue,
    order_kpis,
    revenue_by,
    revenue_by_month,
    revenue_by_product,
    revenue_by_year,
    revenue_concentration,
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


def test_money_is_rounded_to_two_decimals():
    """0.1 + 0.2 must be 0.3 in a money report, not 0.30000000000000004."""
    df = pd.DataFrame(
        {
            "order_date": pd.to_datetime(["2026-01-05", "2026-01-06"]),
            "product": pd.array(["Pen", "Pen"], dtype="string"),
            "quantity": pd.array([1, 1], dtype="Int64"),
            "unit_price": pd.array([0.1, 0.2], dtype="Float64"),
        }
    )

    assert revenue_summary(df)["total_revenue"] == 0.3
    assert revenue_by_product(df)["revenue"].iloc[0] == 0.3
    assert revenue_by_month(df)[0]["revenue"].iloc[0] == 0.3


def _shop() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "order_id": ["A", "A", "B", "C", "D"],
            "city": pd.array(
                ["Lahore", "Lahore", "Karachi", "Lahore", "Multan"], dtype="string"
            ),
            "quantity": pd.array([1, 2, 1, 1, 1], dtype="Int64"),
            "unit_price": pd.array([100, 50, 300, 20, 30], dtype="Int64"),
        }
    )


def test_revenue_by_any_column():
    """Any column can be used for a breakdown, highest revenue first."""
    result = revenue_by(_shop(), "city")

    assert list(result["city"]) == ["Karachi", "Lahore", "Multan"]
    assert list(result["revenue"]) == [300, 220, 30]


def test_revenue_concentration_shows_the_80_20_picture():
    """Top group's share, top 3 share, and how many groups make up 80%."""
    breakdown = revenue_by(_shop(), "city")

    assert revenue_concentration(breakdown) == {
        "groups": 3,
        "top_1_share": 54.5,
        "top_3_share": 100.0,
        "groups_for_80_percent": 2,
    }


def test_order_kpis_group_lines_into_orders_and_report_the_median():
    """Order A has two lines (100 + 100); orders are A=200, B=300, C=20, D=30."""
    kpis = order_kpis(_shop())

    assert kpis["orders"] == 4
    assert kpis["units_sold"] == 6
    assert kpis["average_order_value"] == 137.5
    assert kpis["median_order_value"] == 115.0
    assert kpis["orders_counted_by"] == "order_id"


def test_order_kpis_without_order_id_count_each_row():
    kpis = order_kpis(_shop().drop(columns="order_id"))

    assert kpis["orders"] == 5
    assert kpis["orders_counted_by"] == "row"


def test_yearly_growth_only_between_complete_years():
    """2024 vs 2023 is compared; the partial year 2025 gets no growth figure."""
    months = pd.period_range("2023-01", "2025-03", freq="M").astype(str)
    revenue = [100.0] * 12 + [120.0] * 12 + [50.0] * 3
    monthly = pd.DataFrame({"month": months, "revenue": revenue})

    yearly = revenue_by_year(monthly)

    assert list(yearly["year"]) == [2023, 2024, 2025]
    assert list(yearly["revenue"]) == [1200, 1440, 150]
    assert yearly["growth_percent"].iloc[1] == 20.0
    assert pd.isna(yearly["growth_percent"].iloc[2])
    assert list(yearly["complete_year"]) == [True, True, False]
