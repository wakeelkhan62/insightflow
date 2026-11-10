from datetime import date

import pandas as pd

from insightflow.validation import validate_sales

TODAY = date(2026, 11, 10)


def _check(report: dict, name: str) -> dict:
    return next(c for c in report["checks"] if c["name"] == name)


def test_flags_zero_or_negative_quantity_and_price_with_file_lines():
    """Bad values are counted and located by their line in the CSV file."""
    df = pd.DataFrame(
        {
            "quantity": pd.array([2, 0, -1, 3], dtype="Int64"),
            "unit_price": pd.array([100, 50, None, -5], dtype="Int64"),
        }
    )

    report = validate_sales(df, today=TODAY)

    quantity = _check(report, "quantity_not_positive")
    assert quantity["rows_flagged"] == 2
    assert quantity["example_file_lines"] == [3, 4]
    assert _check(report, "unit_price_not_positive")["rows_flagged"] == 1


def test_flags_future_dates_in_text_or_date_columns():
    """Orders dated after today are flagged; missing dates are not."""
    df = pd.DataFrame({"order_date": ["2026-11-09", "2027-01-01", None]})

    report = validate_sales(df, today=TODAY)

    assert _check(report, "order_date_in_future")["example_file_lines"] == [3]


def test_flags_extreme_values_only():
    """A price far outside the normal range is flagged; normal spread is not."""
    prices = [100, 105, 98, 102, 99, 101, 103, 97, 100, 104, 5000]
    df = pd.DataFrame({"unit_price": pd.array(prices, dtype="Int64")})

    report = validate_sales(df, today=TODAY)

    extreme = _check(report, "unit_price_extreme_value")
    assert extreme["rows_flagged"] == 1
    assert extreme["example_file_lines"] == [12]


def test_rules_without_their_columns_are_skipped_with_a_reason():
    """Any file can be validated; rules that cannot run explain why."""
    df = pd.DataFrame({"product": ["Laptop"]})

    report = validate_sales(df, today=TODAY)

    assert report["checks"] == []
    reasons = {s["name"]: s["reason"] for s in report["skipped"]}
    assert reasons["quantity_not_positive"] == "No quantity column"
    assert reasons["order_date_in_future"] == "No order_date column"


def test_input_is_not_changed():
    """Validation only reports; the data stays exactly as it was."""
    df = pd.DataFrame({"quantity": pd.array([-1], dtype="Int64")})
    before = df.copy()

    validate_sales(df, today=TODAY)

    pd.testing.assert_frame_equal(df, before)


def test_extreme_values_are_judged_within_each_product():
    """Expensive products are not outliers just because other products are cheap."""
    laptops = [185000, 180000, 190000, 182000, 188000, 184000, 186000, 181000]
    stands = [1200, 1150, 1250, 1180, 1220, 1210, 1190, 1230]
    df = pd.DataFrame(
        {
            "product": ["Laptop"] * 10 + ["Stand"] * 10,
            "unit_price": pd.array(
                laptops + [183000, 9999999] + stands + [1240, 1170], dtype="Int64"
            ),
        }
    )

    report = validate_sales(df, today=TODAY)

    extreme = _check(report, "unit_price_extreme_value")
    assert extreme["rows_flagged"] == 1
    assert extreme["example_file_lines"] == [11]
