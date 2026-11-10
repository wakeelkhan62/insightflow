"""Check our revenue calculation against the file's own total column.

Files often carry a total (e.g. "Total Amount") calculated by the shop's own
system. If our formula reproduces it, the revenue numbers can be trusted; rows
where it does not are worth a closer look.
"""

from itertools import combinations

import pandas as pd

TOLERANCE = 0.05  # money difference allowed for rounding in the source system
MIN_MATCH_SHARE = 0.5  # below this, no formula is considered to explain the total
MAX_EXAMPLES = 10


def _candidate_formulas(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Build every sensible way the total could have been calculated."""
    quantity = df["quantity"].astype("Float64")
    bases = {}
    if "list_price" in df.columns:
        bases["gross"] = quantity * df["list_price"].astype("Float64")
        bases["net"] = quantity * df["unit_price"].astype("Float64")
    else:
        bases["gross"] = quantity * df["unit_price"].astype("Float64")
    extras = [c for c in ("tax", "shipping") if c in df.columns]

    formulas = {}
    for base_name, base in bases.items():
        for size in range(len(extras) + 1):
            for chosen in combinations(extras, size):
                total = base.copy()
                for extra in chosen:
                    total = total + df[extra].astype("Float64").fillna(0)
                formulas[" + ".join([base_name, *chosen])] = total
    return formulas


def reconcile_totals(df: pd.DataFrame) -> dict:
    """Find which formula reproduces total_amount, and list rows that differ.

    Needs quantity, unit_price and total_amount (list_price, tax and shipping
    are used when present). Returns the best formula, how many rows it matches,
    the match rate of every formula tried, and example mismatched rows with
    their CSV line numbers. Nothing in df is changed.
    """
    for column in ("quantity", "unit_price", "total_amount"):
        if column not in df.columns:
            raise ValueError(f"No {column} column. Map it first.")

    actual = df["total_amount"].astype("Float64")
    formulas = _candidate_formulas(df)
    scores = {}
    for name, expected in formulas.items():
        checkable = actual.notna() & expected.notna()
        matches = ((expected - actual).abs() <= TOLERANCE) & checkable
        share = matches.sum() / checkable.sum() if checkable.sum() else 0.0
        scores[name] = round(100 * float(share), 2)

    best = max(scores, key=scores.get)
    expected = formulas[best]
    checkable = actual.notna() & expected.notna()
    difference = (actual - expected).where(checkable)
    mismatched = checkable & (difference.abs() > TOLERANCE).fillna(False)
    examples = [
        {
            "file_line": int(i) + 2,
            "total_amount": round(float(actual[i]), 2),
            "expected": round(float(expected[i]), 2),
            "difference": round(float(difference[i]), 2),
        }
        for i in mismatched[mismatched].index[:MAX_EXAMPLES]
    ]
    explained = scores[best] >= 100 * MIN_MATCH_SHARE
    return {
        "best_formula": best if explained else None,
        "rows_checked": int(checkable.sum()),
        "rows_matching": int(checkable.sum() - mismatched.sum()),
        "rows_not_matching": int(mismatched.sum()),
        "match_percent": scores[best],
        "all_formulas_match_percent": scores,
        "examples": examples,
    }
