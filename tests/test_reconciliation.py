import pandas as pd
import pytest

from insightflow.discounts import apply_discount
from insightflow.reconciliation import reconcile_totals


def _orders() -> pd.DataFrame:
    """Totals were made as: qty x price x (1 - discount) + tax + shipping."""
    return pd.DataFrame(
        {
            "quantity": pd.array([2, 1, 3, 1], dtype="Int64"),
            "unit_price": pd.array([1000, 500, 200, 400], dtype="Int64"),
            "discount": pd.array([0.1, 0, 0, 0.5], dtype="Float64"),
            "tax": pd.array([306, 85, 102, 34], dtype="Float64"),
            "shipping": pd.array([0, 250, 250, 250], dtype="Float64"),
            "total_amount": pd.array([2106, 835, 952, 9999], dtype="Float64"),
        }
    )


def test_finds_the_formula_behind_the_files_total_and_lists_mismatches():
    """net + tax + shipping explains 3 of 4 rows; the 4th is reported."""
    discounted, _ = apply_discount(_orders(), "fraction")

    result = reconcile_totals(discounted)

    assert result["best_formula"] == "net + tax + shipping"
    assert result["rows_matching"] == 3
    assert result["rows_not_matching"] == 1
    assert result["examples"] == [
        {
            "file_line": 5,
            "total_amount": 9999.0,
            "expected": 484.0,
            "difference": 9515.0,
        }
    ]


def test_without_discount_or_extras_only_gross_is_tried():
    """With just quantity, price and total, the only formula is gross."""
    df = pd.DataFrame(
        {
            "quantity": pd.array([2, 1], dtype="Int64"),
            "unit_price": pd.array([100, 50], dtype="Int64"),
            "total_amount": pd.array([200, 50], dtype="Float64"),
        }
    )

    result = reconcile_totals(df)

    assert result["best_formula"] == "gross"
    assert result["match_percent"] == 100.0
    assert list(result["all_formulas_match_percent"]) == ["gross"]


def test_reports_no_formula_when_the_total_cannot_be_explained():
    """If no formula matches at least half the rows, none is claimed."""
    df = pd.DataFrame(
        {
            "quantity": pd.array([1, 1], dtype="Int64"),
            "unit_price": pd.array([100, 100], dtype="Int64"),
            "total_amount": pd.array([555, 777], dtype="Float64"),
        }
    )

    assert reconcile_totals(df)["best_formula"] is None


def test_missing_total_column_raises_clear_error():
    df = pd.DataFrame({"quantity": [1], "unit_price": [100]})

    with pytest.raises(ValueError, match="total_amount"):
        reconcile_totals(df)
