import pandas as pd
from pandas.api import types as ptypes

REQUIRED_FOR_REVENUE = ["quantity", "unit_price"]
MONEY_DECIMALS = 2


def _require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    """Raise ValueError naming any required standard columns that are missing."""
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. Apply the column mapping first."
        )


def add_revenue(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with a revenue column (quantity x unit_price).

    Revenue is rounded to 2 decimals. Rows with a missing quantity or price get
    a missing revenue, never 0. The input DataFrame is not changed.
    """
    _require_columns(df, REQUIRED_FOR_REVENUE)
    result = df.copy()
    revenue = result["quantity"] * result["unit_price"]
    result["revenue"] = revenue.round(MONEY_DECIMALS)
    return result


def revenue_summary(df: pd.DataFrame) -> dict:
    """Return total revenue plus how many rows were included and excluded.

    Rows without a revenue (missing quantity or price) are excluded from the
    total and counted, so the user knows exactly what the number is based on.
    """
    revenue = add_revenue(df)["revenue"]
    included = revenue.notna()
    total = revenue[included].sum()
    return {
        "total_revenue": round(float(total), MONEY_DECIMALS),
        "rows_included": int(included.sum()),
        "rows_excluded": int((~included).sum()),
    }


def revenue_by(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Return revenue per value of any column, highest first, with evidence.

    Works for product, category, city, payment method or any other column. A
    group whose rows all lack revenue shows a missing revenue (not 0) and is
    listed last. The input DataFrame is not changed.
    """
    _require_columns(df, [column])
    data = add_revenue(df)
    data["included"] = data["revenue"].notna()
    summary = data.groupby(column, dropna=False).agg(
        revenue=("revenue", lambda r: r.sum(min_count=1)),
        rows_included=("included", "sum"),
        rows_excluded=("included", lambda i: (~i).sum()),
    )
    summary["revenue"] = summary["revenue"].round(MONEY_DECIMALS)
    summary = summary.sort_values("revenue", ascending=False, na_position="last")
    return summary.reset_index()


def revenue_by_product(df: pd.DataFrame) -> pd.DataFrame:
    """Return revenue per product, highest first, with rows included and excluded."""
    return revenue_by(df, "product")


def revenue_concentration(breakdown: pd.DataFrame) -> dict:
    """Describe how concentrated revenue is in a revenue_by() table.

    Returns the share of revenue from the top 1 and top 3 groups, and how many
    groups together make up 80% of revenue (the "80/20" check).
    """
    revenue = breakdown["revenue"].dropna().astype(float)
    revenue = revenue[revenue > 0].sort_values(ascending=False)
    total = revenue.sum()
    if not total:
        return {
            "groups": 0,
            "top_1_share": None,
            "top_3_share": None,
            "groups_for_80_percent": None,
        }
    cumulative = revenue.cumsum() / total
    return {
        "groups": int(len(revenue)),
        "top_1_share": round(100 * float(revenue.iloc[:1].sum() / total), 1),
        "top_3_share": round(100 * float(revenue.iloc[:3].sum() / total), 1),
        "groups_for_80_percent": int((cumulative < 0.8).sum() + 1),
    }


def order_kpis(df: pd.DataFrame) -> dict:
    """Return order count, units sold, and average and median order value.

    Rows are grouped into orders by order_id when that column exists (one order
    can have several product lines); otherwise every row counts as an order.
    Only rows with a known revenue are used. The median is reported because a
    few very large orders can pull the average far above a typical order.
    """
    data = add_revenue(df)
    data = data[data["revenue"].notna()]
    by_order = "order_id" in data.columns
    order_values = (
        data.groupby("order_id")["revenue"].sum() if by_order else data["revenue"]
    )
    if order_values.empty:
        return {
            "orders": 0,
            "units_sold": 0,
            "average_order_value": None,
            "median_order_value": None,
            "orders_counted_by": None,
        }
    return {
        "orders": int(len(order_values)),
        "units_sold": round(float(data["quantity"].sum()), 2),
        "average_order_value": round(float(order_values.mean()), MONEY_DECIMALS),
        "median_order_value": round(float(order_values.median()), MONEY_DECIMALS),
        "orders_counted_by": "order_id" if by_order else "row",
    }


def revenue_by_year(monthly: pd.DataFrame) -> pd.DataFrame:
    """Return revenue per calendar year with growth versus the previous year.

    monthly is the table from revenue_by_month(). Growth is only calculated
    between two complete years (12 months each), because comparing a full year
    with a half year would show a fake drop or jump.
    """
    data = monthly.copy()
    data["year"] = data["month"].str[:4].astype(int)
    yearly = data.groupby("year").agg(
        revenue=("revenue", lambda r: r.sum(min_count=1)),
        months=("month", "size"),
    )
    yearly["revenue"] = yearly["revenue"].astype("Float64").round(MONEY_DECIMALS)
    complete = yearly["months"] == 12
    previous_complete = complete.shift(1, fill_value=False)
    growth = 100 * (yearly["revenue"] / yearly["revenue"].shift(1) - 1)
    yearly["growth_percent"] = growth.where(complete & previous_complete).round(1)
    yearly["complete_year"] = complete
    return yearly.reset_index()


def revenue_by_month(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Return (monthly, undated): revenue per month in date order, and how many
    rows had no date and so could not be placed in any month.

    Months between the first and last sale with no rows at all show revenue 0.
    Months whose rows all lack revenue show a missing revenue, not 0.
    order_date must already be converted to dates.
    """
    _require_columns(df, ["order_date"])
    if not ptypes.is_datetime64_any_dtype(df["order_date"]):
        raise ValueError("order_date must be a date column. Convert it first.")

    data = add_revenue(df)
    has_date = data["order_date"].notna()
    undated = int((~has_date).sum())
    data = data[has_date].copy()
    data["month"] = data["order_date"].dt.to_period("M")
    data["included"] = data["revenue"].notna()

    monthly = data.groupby("month").agg(
        revenue=("revenue", lambda r: r.sum(min_count=1)),
        rows_included=("included", "sum"),
        rows_excluded=("included", lambda i: (~i).sum()),
    )
    if not monthly.empty:
        all_months = pd.period_range(monthly.index.min(), monthly.index.max())
        monthly = monthly.reindex(all_months)
        empty_month = monthly["rows_included"].isna()
        monthly["rows_included"] = monthly["rows_included"].fillna(0).astype(int)
        monthly["rows_excluded"] = monthly["rows_excluded"].fillna(0).astype(int)
        monthly.loc[empty_month, "revenue"] = 0
        monthly["revenue"] = monthly["revenue"].round(MONEY_DECIMALS)

    monthly.index = monthly.index.astype(str)
    return monthly.rename_axis("month").reset_index(), undated
