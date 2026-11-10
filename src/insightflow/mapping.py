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
    """Split camelCase, lower-case, drop text in brackets, and unify separators.

    Examples: "UnitPrice" -> "unit price", "Rate (PKR)" -> "rate".
    """
    name = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", name)
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


def apply_column_mapping(df: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    """Return a copy of df with mapped columns renamed to standard field names.

    mapping is standard field -> original column, as confirmed by the user.
    Unmapped columns keep their names. The input DataFrame is not changed.
    Raises ValueError for unknown fields, missing columns, a column used
    twice, or a rename that would clash with an existing column.
    """
    unknown_fields = sorted(set(mapping) - set(STANDARD_FIELDS))
    if unknown_fields:
        raise ValueError(f"Unknown standard fields: {unknown_fields}")
    missing = sorted(set(mapping.values()) - set(map(str, df.columns)))
    if missing:
        raise ValueError(f"Columns not found in dataset: {missing}")
    if len(set(mapping.values())) != len(mapping):
        raise ValueError("Each column can be mapped to only one field")

    renames = {column: field for field, column in mapping.items()}
    renamed = [renames.get(str(c), str(c)) for c in df.columns]
    clashes = sorted({name for name in renamed if renamed.count(name) > 1})
    if clashes:
        raise ValueError(f"Mapping creates duplicate column names: {clashes}")
    return df.rename(columns=renames)
