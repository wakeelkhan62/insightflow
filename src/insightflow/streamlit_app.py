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
    suggest_excluded_statuses,
)
from insightflow.forecasting import forecast_next_month
from insightflow.ingestion import load_csv
from insightflow.mapping import (
    STANDARD_FIELDS,
    apply_column_mapping,
    suggest_column_mapping,
)
from insightflow.quality import build_quality_report

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
for box, field in zip(st.columns(len(STANDARD_FIELDS)), STANDARD_FIELDS, strict=True):
    default = suggested.get(field, NOT_IN_FILE)
    choice = box.selectbox(field, options, index=options.index(default), key=field)
    if choice != NOT_IN_FILE:
        mapping[field] = choice
remove_duplicates = st.checkbox("Remove exact duplicate rows", value=True)

if "quantity" not in mapping or "unit_price" not in mapping:
    st.warning("Choose the quantity and unit_price columns to calculate revenue.")
    st.stop()
try:
    data = apply_column_mapping(df, mapping)
except ValueError as error:
    st.error(str(error))
    st.stop()

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

# 3. Revenue summary
st.subheader("Revenue")
summary = revenue_summary(data)
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Total revenue", money(summary["total_revenue"]))
m2.metric("Rows used", f"{summary['rows_included']:,}")
m3.metric("Missing qty/price", f"{summary['rows_excluded']:,}")
m4.metric("Duplicates removed", f"{removed_count:,}")
m5.metric("Left out by status", f"{status_excluded_count:,}")
if summary["rows_excluded"]:
    st.caption(
        f"{summary['rows_excluded']:,} rows were left out because quantity or price "
        "is missing, so actual revenue is higher than shown."
    )
else:
    st.caption("Revenue = quantity x unit price (gross, before discounts).")

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
