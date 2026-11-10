"""The full InsightFlow pipeline: one place where every step runs in order.

Both the Streamlit app and the API call this module, so they always apply the
same cleaning and calculations and always show the same numbers.

1. preview_choices() looks at the file and returns the choices the user must
   confirm (statuses to leave out, discount type, future-dated rows).
2. run_pipeline() applies the confirmed choices and returns every result.
"""

from dataclasses import dataclass, field
from datetime import date

import pandas as pd

from insightflow.analytics import (
    order_kpis,
    revenue_by_month,
    revenue_by_product,
    revenue_by_year,
    revenue_summary,
)
from insightflow.cleaning import (
    convert_date_columns,
    remove_duplicate_rows,
    split_by_order_status,
    split_future_dates,
    standardize_text_values,
    suggest_excluded_statuses,
)
from insightflow.discounts import apply_discount, guess_discount_type
from insightflow.forecasting import forecast_next_month
from insightflow.mapping import apply_column_mapping
from insightflow.reconciliation import reconcile_totals
from insightflow.validation import validate_sales

REQUIRED_FIELDS = ("quantity", "unit_price")


@dataclass
class PipelineOptions:
    """The user's confirmed choices. None means "use the suggestion"."""

    mapping: dict[str, str]
    fix_text: bool = True
    remove_duplicates: bool = True
    excluded_statuses: list[str] | None = None
    leave_out_future: bool = True
    discount_type: str | None = None
    today: date | None = None


@dataclass
class PipelineResult:
    """Everything the pipeline produced, with the evidence for each number."""

    data: pd.DataFrame
    text_merges: pd.DataFrame
    checks: dict
    duplicates_removed: int
    excluded_statuses: list[str]
    status_excluded: int
    future_left_out: int
    discount: dict | None
    gross_total: float
    summary: dict
    kpis: dict
    by_product: pd.DataFrame | None
    reconciliation: dict | None
    monthly: pd.DataFrame | None = None
    undated_rows: int = 0
    yearly: pd.DataFrame | None = None
    forecast: dict = field(default_factory=dict)


def _mapped(df: pd.DataFrame, mapping: dict[str, str], fix_text: bool):
    """Apply the mapping and (optionally) text fixing; return (data, merges)."""
    missing = [f for f in REQUIRED_FIELDS if f not in mapping]
    if missing:
        raise ValueError(f"Map these required fields first: {missing}")
    data = apply_column_mapping(df, mapping)
    if fix_text:
        return standardize_text_values(data)
    return data, pd.DataFrame()


def preview_choices(
    df: pd.DataFrame,
    mapping: dict[str, str],
    fix_text: bool = True,
    today: date | None = None,
) -> dict:
    """Return the choices the user must confirm before running the pipeline.

    statuses: every status in the file (after text fixing);
    suggested_statuses: those that usually mean no money was received;
    future_rows: how many orders are dated after today;
    discount_guess: how the discount column seems to be written, or None.
    """
    data, _ = _mapped(df, mapping, fix_text)
    has_status = "order_status" in data.columns
    has_date = "order_date" in data.columns
    return {
        "statuses": (
            sorted(data["order_status"].dropna().astype(str).unique())
            if has_status
            else []
        ),
        "suggested_statuses": suggest_excluded_statuses(data),
        "future_rows": len(split_future_dates(data, today)[1]) if has_date else 0,
        "discount_guess": guess_discount_type(data),
    }


def run_pipeline(df: pd.DataFrame, options: PipelineOptions) -> PipelineResult:
    """Clean the data with the user's choices, then calculate every result.

    Raises ValueError with a user-friendly message when the mapping, the
    dates or the discount settings make the calculation impossible.
    """
    data, merges = _mapped(df, options.mapping, options.fix_text)
    checks = validate_sales(data, today=options.today)

    duplicates_removed = 0
    if options.remove_duplicates:
        data, removed = remove_duplicate_rows(data)
        duplicates_removed = len(removed)

    excluded_statuses: list[str] = []
    status_excluded = 0
    if "order_status" in data.columns:
        excluded_statuses = (
            suggest_excluded_statuses(data)
            if options.excluded_statuses is None
            else options.excluded_statuses
        )
        data, left_out = split_by_order_status(data, excluded_statuses)
        status_excluded = len(left_out)

    future_left_out = 0
    if "order_date" in data.columns and options.leave_out_future:
        data, future = split_future_dates(data, options.today)
        future_left_out = len(future)

    gross_total = revenue_summary(data)["total_revenue"]
    discount = None
    if "discount" in data.columns:
        discount_type = options.discount_type or guess_discount_type(data)
        if discount_type:
            data, discount = apply_discount(data, discount_type)

    result = PipelineResult(
        data=data,
        text_merges=merges,
        checks=checks,
        duplicates_removed=duplicates_removed,
        excluded_statuses=excluded_statuses,
        status_excluded=status_excluded,
        future_left_out=future_left_out,
        discount=discount,
        gross_total=gross_total,
        summary=revenue_summary(data),
        kpis=order_kpis(data),
        by_product=revenue_by_product(data) if "product" in data.columns else None,
        reconciliation=(
            reconcile_totals(data) if "total_amount" in data.columns else None
        ),
    )

    if "order_date" in data.columns:
        dated = convert_date_columns(data, ["order_date"])
        result.monthly, result.undated_rows = revenue_by_month(dated)
        result.yearly = revenue_by_year(result.monthly)
        try:
            result.forecast = {"available": True, **forecast_next_month(result.monthly)}
        except ValueError as error:
            result.forecast = {"available": False, "reason": str(error)}
    return result
