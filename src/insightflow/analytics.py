import pandas as pd
from pandas.api import types as ptypes

REQUIRED_FOR_REVENUE = ["quantity", "unit_price"]


def _require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    """Raise ValueError naming any required standard columns that are missing."""
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. Apply the column mapping first."
        )


def add_revenue(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with a revenue column (quantity x unit_price).

    Rows with a missing quantity or price get a missing revenue, never 0.
    The input DataFrame is not changed.
    """
    _require_columns(df, REQUIRED_FOR_REVENUE)
    result = df.copy()
    result["revenue"] = result["quantity"] * result["unit_price"]
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
        "total_revenue": float(total),
        "rows_included": int(included.sum()),
        "rows_excluded": int((~included).sum()),
    }


def revenue_by_product(df: pd.DataFrame) -> pd.DataFrame:
    """Return revenue per product, highest first, with rows included and excluded.

    A product whose rows all lack revenue shows a missing revenue (not 0) and
    is listed last. The input DataFrame is not changed.
    """
    _require_columns(df, ["product"])
    data = add_revenue(df)
    data["included"] = data["revenue"].notna()
    summary = data.groupby("product", dropna=False).agg(
        revenue=("revenue", lambda r: r.sum(min_count=1)),
        rows_included=("included", "sum"),
        rows_excluded=("included", lambda i: (~i).sum()),
    )
    summary = summary.sort_values("revenue", ascending=False, na_position="last")
    return summary.reset_index()


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

    monthly.index = monthly.index.astype(str)
    return monthly.rename_axis("month").reset_index(), undated
