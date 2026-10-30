import re

import pandas as pd

STANDARD_FIELDS: dict[str, set[str]] = {
    "order_id": {
        "order id",
        "order no",
        "order number",
        "invoice",
        "invoice id",
        "invoice no",
        "invoice number",
        "transaction id",
    },
    "order_date": {
        "order date",
        "date",
        "invoice date",
        "transaction date",
        "sale date",
    },
    "product": {"product", "product name", "item", "item name", "sku"},
    "quantity": {"quantity", "qty", "units", "units sold", "quantity sold"},
    "unit_price": {"unit price", "price", "rate", "selling price"},
}


def _normalize(name: str) -> str:
    """Lower-case a column name, drop text in brackets, and unify separators."""
    name = re.sub(r"\(.*?\)", "", name.lower())
    name = re.sub(r"[_\-.]+", " ", name)
    return " ".join(name.split())


def suggest_column_mapping(df: pd.DataFrame) -> dict[str, str]:
    """Suggest which column holds each standard field, e.g. 'Qty' -> quantity.

    Returns standard field -> original column name. Fields with no match are
    left out. Each column is used at most once. Nothing is changed.
    """
    mapping: dict[str, str] = {}
    used: set[str] = set()
    for field, synonyms in STANDARD_FIELDS.items():
        for column in df.columns:
            name = str(column)
            if name in used:
                continue
            if _normalize(name) in synonyms:
                mapping[field] = name
                used.add(name)
                break
    return mapping
