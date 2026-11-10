import pandas as pd

from insightflow.analytics import MONEY_DECIMALS

SEASON_LENGTH = 12


def _predict(history: list[float], method: str) -> float:
    """Predict the next value from past values with one simple method."""
    if method == "last_month":
        return history[-1]
    if method == "average_last_3_months":
        return sum(history[-3:]) / 3
    return history[-SEASON_LENGTH]  # same_month_last_year


def _available_methods(n_history: int) -> list[str]:
    """Return the methods that have enough history to make a prediction."""
    methods = ["last_month"]
    if n_history >= 3:
        methods.append("average_last_3_months")
    if n_history >= SEASON_LENGTH:
        methods.append("same_month_last_year")
    return methods


def _mape(actuals: list[float], predictions: list[float]) -> float:
    """Mean absolute percentage error, skipping months with zero revenue."""
    errors = [abs(a - p) / a for a, p in zip(actuals, predictions, strict=True) if a]
    return 100 * sum(errors) / len(errors) if errors else float("nan")


def forecast_next_month(monthly: pd.DataFrame, test_months: int = 12) -> dict:
    """Forecast next month's revenue with the simple method that did best recently.

    monthly has 'month' (e.g. "2024-12") and 'revenue', in date order. Each
    method is back-tested on the last test_months: it predicts every one of
    those months using only the months before it, like a real forecast would.
    The method with the lowest error (MAPE, in %) makes the forecast.
    Raises ValueError if there is too little data or a month has no revenue.
    """
    if monthly["revenue"].isna().any():
        raise ValueError("Some months have unknown revenue; fill missing values first")
    values = [float(v) for v in monthly["revenue"]]
    if len(values) < test_months + 3:
        raise ValueError(
            f"Need at least {test_months + 3} months of data to forecast, "
            f"got {len(values)}"
        )

    first_test = len(values) - test_months
    scores = {}
    for method in _available_methods(first_test):
        predictions = [
            _predict(values[:t], method) for t in range(first_test, len(values))
        ]
        scores[method] = round(_mape(values[first_test:], predictions), 2)

    best = min(scores, key=scores.get)
    next_month = pd.Period(monthly["month"].iloc[-1], freq="M") + 1
    return {
        "month": str(next_month),
        "forecast": round(_predict(values, best), MONEY_DECIMALS),
        "method": best,
        "backtest_error_percent": scores[best],
        "all_methods_error_percent": scores,
        "months_used": len(values),
    }
