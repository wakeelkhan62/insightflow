import pandas as pd

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
