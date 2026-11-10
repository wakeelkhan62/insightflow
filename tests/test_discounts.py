import pandas as pd
import pytest

from insightflow.analytics import revenue_summary
from insightflow.discounts import apply_discount, guess_discount_type


def _sales(discounts: list) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "quantity": pd.array([2, 1, 4], dtype="Int64"),
            "unit_price": pd.array([1000, 500, 250], dtype="Int64"),
            "discount": pd.array(discounts, dtype="Float64"),
        }
    )


def test_guesses_how_the_discount_is_written():
    """0-1 is a fraction, up to 100 a percent, above that an amount."""
    assert guess_discount_type(_sales([0.1, 0, 0.25])) == "fraction"
    assert guess_discount_type(_sales([10, 0, 25])) == "percent"
    assert guess_discount_type(_sales([200, 0, 150])) == "amount"
    assert guess_discount_type(pd.DataFrame({"quantity": [1]})) is None


def test_fraction_discount_gives_net_revenue_and_keeps_list_price():
    """Gross 2000 + 500 + 1000 = 3500; minus 10% and 20% on two lines = 3100."""
    df = _sales([0.1, 0, 0.2])

    discounted, report = apply_discount(df, "fraction")

    assert revenue_summary(discounted)["total_revenue"] == 3100.0
    assert report["total_discount"] == 400.0
    assert list(discounted["list_price"]) == [1000, 500, 250]
    assert list(df["unit_price"]) == [1000, 500, 250]


def test_percent_discount_matches_the_same_fraction():
    """10 percent and 0.10 give the same net revenue."""
    discounted, _ = apply_discount(_sales([10, 0, 20]), "percent")

    assert revenue_summary(discounted)["total_revenue"] == 3100.0


def test_amount_discount_is_taken_off_the_line_and_impossible_ones_are_counted():
    """Rs 200 off a Rs 2000 line is fine; Rs 900 off a Rs 500 line is impossible."""
    discounted, report = apply_discount(_sales([200, 900, 0]), "amount")

    summary = revenue_summary(discounted)
    assert summary["total_revenue"] == 2800.0
    assert summary["rows_excluded"] == 1
    assert report["invalid_discount_rows"] == 1


def test_blank_discount_means_no_discount_and_is_counted():
    """A missing discount is treated as 0, and the user is told how many."""
    discounted, report = apply_discount(_sales([0.1, None, 0]), "fraction")

    assert revenue_summary(discounted)["total_revenue"] == 3300.0
    assert report["blank_discount_rows"] == 1


def test_unknown_discount_type_raises_clear_error():
    with pytest.raises(ValueError, match="discount_type"):
        apply_discount(_sales([0, 0, 0]), "percentage")
