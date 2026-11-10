import pandas as pd
import pytest

from insightflow.forecasting import forecast_next_month


def _monthly(values: list[float], start: str = "2024-01") -> pd.DataFrame:
    months = pd.period_range(start, periods=len(values), freq="M").astype(str)
    return pd.DataFrame({"month": months, "revenue": values})


def test_picks_the_method_with_lowest_backtest_error():
    """A repeating yearly pattern is best predicted by the same month last year."""
    pattern = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200]

    result = forecast_next_month(_monthly(pattern * 2))

    assert result["month"] == "2026-01"
    assert result["method"] == "same_month_last_year"
    assert result["forecast"] == 100
    assert result["backtest_error_percent"] == 0


def test_short_history_uses_only_methods_it_can_support():
    """With under a year of history before the test window, no seasonal method."""
    result = forecast_next_month(_monthly([100, 110, 120, 130, 140]), test_months=2)

    assert "same_month_last_year" not in result["all_methods_error_percent"]
    assert result["month"] == "2024-06"


def test_too_little_data_raises_clear_error():
    """Forecasting needs enough months to test itself first."""
    with pytest.raises(ValueError, match="Need at least"):
        forecast_next_month(_monthly([100, 200, 300]))


def test_unknown_revenue_month_raises_clear_error():
    """A month with unknown revenue would make the forecast misleading."""
    values = pd.array([100.0] * 14 + [None], dtype="Float64")

    with pytest.raises(ValueError, match="unknown revenue"):
        forecast_next_month(_monthly(list(values)))
