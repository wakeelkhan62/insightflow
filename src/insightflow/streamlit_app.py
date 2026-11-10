"""InsightFlow AI web app.

Run with: uv run streamlit run src/insightflow/streamlit_app.py
"""

import io

import pandas as pd
import streamlit as st

from insightflow.analytics import revenue_by_month, revenue_by_product, revenue_summary
from insightflow.cleaning import (
    convert_date_columns,
    remove_duplicate_rows,
    split_by_order_status,
    split_future_dates,
    standardize_text_values,
    suggest_excluded_statuses,
)
from insightflow.discounts import DISCOUNT_TYPES, apply_discount, guess_discount_type
from insightflow.forecasting import forecast_next_month
from insightflow.ingestion import load_csv
from insightflow.mapping import (
    STANDARD_FIELDS,
    apply_column_mapping,
    suggest_column_mapping,
)
from insightflow.quality import build_quality_report
from insightflow.reconciliation import reconcile_totals
from insightflow.validation import validate_sales

MAX_UPLOAD_MB = 50
NOT_IN_FILE = "(not in my file)"


@st.cache_data
def read_csv(content: bytes) -> pd.DataFrame:
    """Parse uploaded bytes once; Streamlit re-runs the script on every click."""
    return load_csv(io.BytesIO(content))


def money(value: float) -> str:
    return f"{value:,.2f}"


st.set_page_config(page_title="InsightFlow AI", layout="wide")
st.title("InsightFlow AI")
st.caption("Upload sales data, confirm the columns, and get revenue with evidence.")

uploaded = st.file_uploader("Upload a CSV file", type="csv")
if uploaded is None:
    st.info("Choose a CSV file to start.")
    st.stop()

content = uploaded.getvalue()
if len(content) > MAX_UPLOAD_MB * 1024 * 1024:
    st.error(f"File is larger than the {MAX_UPLOAD_MB} MB limit.")
    st.stop()
try:
    df = read_csv(content)
except (pd.errors.ParserError, UnicodeDecodeError, ValueError):
    st.error("The file could not be read as CSV.")
    st.stop()

# 1. Data quality
report = build_quality_report(df)
with st.expander("Data quality report", expanded=False):
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{report['row_count']:,}")
    c2.metric("Columns", report["column_count"])
    c3.metric("Exact duplicate rows", f"{report['duplicate_rows']:,}")
    quality = pd.DataFrame(
        {
            "column": list(report["column_types"]),
            "type": list(report["column_types"].values()),
            "missing": [report["missing_values"][c] for c in report["column_types"]],
        }
    )
    st.dataframe(quality, hide_index=True, width="stretch")

# 2. Column mapping: suggested automatically, confirmed by the user
st.subheader("Confirm your columns")
st.caption("We guessed which column is which. Change any that are wrong.")
suggested = suggest_column_mapping(df)
options = [NOT_IN_FILE, *map(str, df.columns)]
mapping: dict[str, str] = {}
fields = list(STANDARD_FIELDS)
for start in range(0, len(fields), 5):  # 5 dropdowns per row
    row_fields = fields[start : start + 5]
    for box, field in zip(st.columns(5), row_fields, strict=False):
        default = suggested.get(field, NOT_IN_FILE)
        choice = box.selectbox(field, options, index=options.index(default), key=field)
        if choice != NOT_IN_FILE:
            mapping[field] = choice
fix_text = st.checkbox(
    "Fix inconsistent text (extra spaces, different capital letters)", value=True
)
remove_duplicates = st.checkbox("Remove exact duplicate rows", value=True)

if "quantity" not in mapping or "unit_price" not in mapping:
    st.warning("Choose the quantity and unit_price columns to calculate revenue.")
    st.stop()
try:
    data = apply_column_mapping(df, mapping)
except ValueError as error:
    st.error(str(error))
    st.stop()

# Text standardization: "CANCELLED", "cancelled " and "Cancelled" become one value
if fix_text:
    data, merges = standardize_text_values(data)
    if not merges.empty:
        with st.expander(f"Text fixed: {int(merges['rows_changed'].sum()):,} values"):
            st.dataframe(merges, hide_index=True, width="stretch")

# Business rule checks: flag problems, the user decides what to do
checks = validate_sales(data)
problems = [c for c in checks["checks"] if c["rows_flagged"]]
with st.expander(f"Data checks ({len(problems)} problem types found)"):
    if problems:
        st.dataframe(pd.DataFrame(problems), hide_index=True, width="stretch")
    else:
        st.write("No problems found by the business rules.")
    if checks["skipped"]:
        st.caption(
            "Not checked: "
            + "; ".join(f"{s['name']} ({s['reason']})" for s in checks["skipped"])
        )

removed_count = 0
if remove_duplicates:
    data, removed = remove_duplicate_rows(data)
    removed_count = len(removed)

# Order status: leave out orders that brought in no money (user decides which)
status_excluded_count = 0
if "order_status" in data.columns:
    statuses = sorted(data["order_status"].dropna().astype(str).unique())
    chosen = st.multiselect(
        "Order statuses to leave out of revenue",
        statuses,
        default=suggest_excluded_statuses(data),
        help="Cancelled or returned orders usually brought in no money.",
    )
    data, status_excluded = split_by_order_status(data, chosen)
    status_excluded_count = len(status_excluded)
else:
    st.caption("No order status column mapped, so every order counts as revenue.")

# Future dates: a sale cannot happen after today, so these are almost always typos
future_count = 0
if "order_date" in data.columns:
    kept, future = split_future_dates(data)
    if len(future):
        leave_out_future = st.checkbox(
            f"Leave out {len(future):,} orders dated in the future",
            value=True,
            help="They would stretch the monthly chart into empty future months.",
        )
        if leave_out_future:
            data, future_count = kept, len(future)

# Discounts: net revenue = what customers actually paid for the goods
discount_report = None
if "discount" in data.columns:
    labels = {
        "fraction": "Fraction (0.10 means 10%)",
        "percent": "Percent (10 means 10%)",
        "amount": "Money off the order line (500 means Rs 500)",
    }
    guess = guess_discount_type(data) or "fraction"
    discount_type = st.selectbox(
        "How is the discount written in your file?",
        DISCOUNT_TYPES,
        index=DISCOUNT_TYPES.index(guess),
        format_func=labels.get,
        help="We guessed from the values. Values up to 100 could be a percent or "
        "a small amount, so please check.",
    )
    gross_total = revenue_summary(data)["total_revenue"]
    try:
        data, discount_report = apply_discount(data, discount_type)
    except ValueError as error:
        st.error(str(error))
        st.stop()
    if discount_report["invalid_discount_rows"]:
        st.warning(
            f"{discount_report['invalid_discount_rows']:,} rows have an impossible "
            "discount and are left out of revenue."
        )
    if discount_report["blank_discount_rows"]:
        st.caption(
            f"{discount_report['blank_discount_rows']:,} rows have no discount "
            "value and are counted as no discount."
        )

# 3. Revenue summary
st.subheader("Revenue")
summary = revenue_summary(data)
if discount_report:
    d1, d2, _ = st.columns(3)
    d1.metric("Gross sales (before discounts)", money(gross_total))
    d2.metric("Discounts given", money(discount_report["total_discount"]))
m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric(
    "Net revenue" if discount_report else "Total revenue",
    money(summary["total_revenue"]),
)
m2.metric("Rows used", f"{summary['rows_included']:,}")
m3.metric("Missing qty/price", f"{summary['rows_excluded']:,}")
m4.metric("Duplicates removed", f"{removed_count:,}")
m5.metric("Left out by status", f"{status_excluded_count:,}")
m6.metric("Future dates left out", f"{future_count:,}")
if summary["rows_excluded"]:
    st.caption(
        f"{summary['rows_excluded']:,} rows were left out because quantity or price "
        "is missing, so actual revenue is higher than shown."
    )
elif discount_report:
    st.caption("Net revenue = quantity x unit price, after discounts, before tax.")
else:
    st.caption("Revenue = quantity x unit price (gross, before discounts).")

# Reconciliation: does our calculation reproduce the file's own total column?
if "total_amount" in data.columns:
    check = reconcile_totals(data)
    title = f"Check against your file's totals: {check['match_percent']}% of rows match"
    with st.expander(title):
        if check["best_formula"]:
            st.write(
                f"Your file's total = **{check['best_formula']}** for "
                f"{check['rows_matching']:,} of {check['rows_checked']:,} rows."
            )
        else:
            st.warning("No formula we tried explains your file's total column.")
        if check["rows_not_matching"]:
            st.write(
                f"{check['rows_not_matching']:,} rows do not match. "
                "These are worth checking in your file:"
            )
            st.dataframe(pd.DataFrame(check["examples"]), hide_index=True)
        st.caption(
            "Formulas tried (% of rows matching): "
            + "; ".join(
                f"{name} {pct}%"
                for name, pct in check["all_formulas_match_percent"].items()
            )
        )

# 4. Monthly trend and forecast
if "order_date" in data.columns:
    st.subheader("Monthly revenue")
    try:
        dated = convert_date_columns(data, ["order_date"])
    except ValueError as error:
        st.error(str(error))
        st.stop()
    monthly, undated = revenue_by_month(dated)
    st.line_chart(monthly.set_index("month")["revenue"])
    if undated:
        st.caption(f"{undated:,} rows have no date and are not in the chart.")
    try:
        forecast = forecast_next_month(monthly)
    except ValueError as error:
        st.info(f"No forecast: {error}")
    else:
        st.metric(f"Forecast for {forecast['month']}", money(forecast["forecast"]))
        st.caption(
            f"Method: {forecast['method'].replace('_', ' ')}. Typical error when "
            f"tested on the last 12 months: {forecast['backtest_error_percent']}%."
        )
else:
    st.info("Map an order_date column to see the monthly trend and forecast.")

# 5. Products
st.subheader("Top 10 products by revenue")
if "product" in data.columns:
    st.dataframe(revenue_by_product(data).head(10), hide_index=True, width="stretch")
else:
    st.info("Map a product column to see revenue by product.")
