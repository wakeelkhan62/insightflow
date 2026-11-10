from datetime import date

import pandas as pd
import pytest

from insightflow.pipeline import PipelineOptions, preview_choices, run_pipeline

TODAY = date(2026, 11, 10)
MAPPING = {
    "order_id": "Invoice",
    "order_date": "Date",
    "product": "Item",
    "quantity": "Qty",
    "unit_price": "Price",
    "discount": "Discount",
    "order_status": "Status",
}


def _messy_sales() -> pd.DataFrame:
    """A copied row, two spellings of cancelled, a future date and discounts."""
    return pd.DataFrame(
        {
            "Invoice": ["A", "A", "B", "C", "D", "E"],
            "Date": [
                "2026-01-05",
                "2026-01-05",
                "2026-01-06",
                "2026-01-07",
                "2027-05-01",
                "2026-02-01",
            ],
            "Item": ["Pen", "Pen", "Pen", "Cup", "Cup", "Cup"],
            "Qty": pd.array([2, 2, 1, 4, 1, 3], dtype="Int64"),
            "Price": pd.array([100, 100, 100, 50, 50, 50], dtype="Int64"),
            "Discount": pd.array([0.1, 0.1, 0, 0, 0, 0.2], dtype="Float64"),
            "Status": pd.array(
                [
                    "Delivered",
                    "Delivered",
                    "cancelled ",
                    "Cancelled",
                    "Delivered",
                    "Delivered",
                ],
                dtype="string",
            ),
        }
    )


def test_preview_lists_the_choices_the_user_must_confirm():
    """Text is fixed first, so both spellings of cancelled become one choice."""
    choices = preview_choices(_messy_sales(), MAPPING, today=TODAY)

    assert choices["statuses"] == ["Cancelled", "Delivered"]
    assert choices["suggested_statuses"] == ["Cancelled"]
    assert choices["future_rows"] == 1
    assert choices["discount_guess"] == "fraction"


def test_full_pipeline_with_suggested_choices():
    """Only orders A (net 180) and E (net 120) count: 300 net, 350 gross."""
    result = run_pipeline(_messy_sales(), PipelineOptions(MAPPING, today=TODAY))

    assert result.duplicates_removed == 1
    assert result.status_excluded == 2
    assert result.future_left_out == 1
    assert result.gross_total == 350.0
    assert result.discount["total_discount"] == 50.0
    assert result.summary["total_revenue"] == 300.0
    assert result.kpis["orders"] == 2
    assert list(result.monthly["month"]) == ["2026-01", "2026-02"]
    assert result.forecast["available"] is False


def test_user_choices_override_the_suggestions():
    """Keeping duplicates, cancelled orders and future dates counts every row."""
    options = PipelineOptions(
        MAPPING,
        remove_duplicates=False,
        excluded_statuses=[],
        leave_out_future=False,
        today=TODAY,
    )

    result = run_pipeline(_messy_sales(), options)

    assert result.summary["total_revenue"] == 830.0
    assert result.duplicates_removed == 0
    assert result.status_excluded == 0


def test_missing_required_fields_raise_clear_error():
    with pytest.raises(ValueError, match="unit_price"):
        run_pipeline(_messy_sales(), PipelineOptions({"quantity": "Qty"}))
