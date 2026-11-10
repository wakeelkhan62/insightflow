"""Discounts and net revenue.

A discount column can be written three ways, and the file does not say which:
0.10 (fraction), 10 (percent) or 500 (money taken off the order line). The
type is guessed from the values and confirmed by the user.
"""

import pandas as pd
from pandas.api import types as ptypes

DISCOUNT_TYPES = ("fraction", "percent", "amount")


def guess_discount_type(df: pd.DataFrame) -> str | None:
    """Guess how the discount column is written, or None if there is none.

    All values 0-1 -> "fraction"; up to 100 -> "percent" (could also be a small
    amount, so the user must confirm); larger -> "amount".
    """
    if "discount" not in df.columns:
        return None
    values = pd.to_numeric(df["discount"], errors="coerce").dropna()
    if values.empty:
        return None
    if values.max() <= 1:
        return "fraction"
    if values.max() <= 100:
        return "percent"
    return "amount"


def apply_discount(df: pd.DataFrame, discount_type: str) -> tuple[pd.DataFrame, dict]:
    """Return (discounted, report): unit_price becomes the net price per unit.

    The original price is kept in a new list_price column, so every revenue
    function now reports net revenue. An "amount" discount is taken off the
    whole order line. A blank discount means no discount and is counted. An
    impossible discount (below 0, above 100%, or more than the line is worth)
    makes that row's price unknown, so it is left out of revenue and counted.
    The input is not changed.
    """
    for column in ("quantity", "unit_price", "discount"):
        if column not in df.columns:
            raise ValueError(f"No {column} column. Map it first.")
    if discount_type not in DISCOUNT_TYPES:
        raise ValueError(f"discount_type must be one of {DISCOUNT_TYPES}")
    if not ptypes.is_numeric_dtype(df["discount"]):
        raise ValueError("The discount column must contain numbers")

    discount = df["discount"].astype("Float64")
    blank = discount.isna()
    discount = discount.fillna(0)
    quantity = df["quantity"].astype("Float64")
    gross_line = quantity * df["unit_price"].astype("Float64")

    if discount_type == "fraction":
        valid = (discount >= 0) & (discount <= 1)
        net_line = gross_line * (1 - discount)
    elif discount_type == "percent":
        valid = (discount >= 0) & (discount <= 100)
        net_line = gross_line * (1 - discount / 100)
    else:
        valid = (discount >= 0) & ((discount <= gross_line) | gross_line.isna())
        net_line = gross_line - discount
    invalid = ~valid.fillna(True)

    net_unit_price = (net_line / quantity.where(quantity != 0)).mask(invalid)
    result = df.copy()
    result["list_price"] = df["unit_price"]
    result["unit_price"] = net_unit_price.astype("Float64")

    counted = valid.fillna(False) & gross_line.notna()
    report = {
        "discount_type": discount_type,
        "total_discount": round(float((gross_line - net_line)[counted].sum()), 2),
        "rows_with_discount": int((discount > 0).sum()),
        "blank_discount_rows": int(blank.sum()),
        "invalid_discount_rows": int(invalid.sum()),
    }
    return result, report
