"""InsightFlow AI web app.

Run with: uv run streamlit run src/insightflow/streamlit_app.py

All cleaning and calculations happen in insightflow.pipeline; this file only
asks the user for their choices and shows the results.
"""

import io

import pandas as pd
import streamlit as st

from insightflow.analytics import revenue_by, revenue_concentration
from insightflow.discounts import DISCOUNT_TYPES
from insightflow.ingestion import load_csv
from insightflow.mapping import STANDARD_FIELDS, suggest_column_mapping
from insightflow.pipeline import PipelineOptions, preview_choices, run_pipeline
from insightflow.quality import build_quality_report

MAX_UPLOAD_MB = 50
NOT_IN_FILE = "(not in my file)"
DISCOUNT_LABELS = {
    "fraction": "Fraction (0.10 means 10%)",
    "percent": "Percent (10 means 10%)",
    "amount": "Money off the order line (500 means Rs 500)",
}


@st.cache_data
def read_csv(content: bytes) -> pd.DataFrame:
    """Parse uploaded bytes once; Streamlit re-runs the script on every click."""
    return load_csv(io.BytesIO(content))


def money(value: float) -> str:
    return f"{value:,.2f}"


st.set_page_config(page_title="InsightFlow AI", layout="wide")
st.title("InsightFlow AI")
st.caption("Upload sales data, confirm the columns, and get revenue with evidence.")

# 1. Upload
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

# 2. The user's choices
st.subheader("Confirm your columns")
st.caption("We guessed which column is which. Change any that are wrong.")
suggested = suggest_column_mapping(df)
options = [NOT_IN_FILE, *map(str, df.columns)]
mapping: dict[str, str] = {}
fields = list(STANDARD_FIELDS)
for start in range(0, len(fields), 5):  # 5 dropdowns per row
    for box, field in zip(st.columns(5), fields[start : start + 5], strict=False):
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
    preview = preview_choices(df, mapping, fix_text=fix_text)
except ValueError as error:
    st.error(str(error))
    st.stop()

excluded_statuses = None
if preview["statuses"]:
    excluded_statuses = st.multiselect(
        "Order statuses to leave out of revenue",
        preview["statuses"],
        default=preview["suggested_statuses"],
        help="Cancelled or returned orders usually brought in no money.",
    )
leave_out_future = True
if preview["future_rows"]:
    leave_out_future = st.checkbox(
        f"Leave out {preview['future_rows']:,} orders dated in the future",
        value=True,
        help="A sale cannot happen after today; these are almost always typos.",
    )
discount_type = None
if "discount" in mapping:
    guess = preview["discount_guess"] or "fraction"
    discount_type = st.selectbox(
        "How is the discount written in your file?",
        DISCOUNT_TYPES,
        index=DISCOUNT_TYPES.index(guess),
        format_func=DISCOUNT_LABELS.get,
        help="We guessed from the values. Values up to 100 could be a percent or "
        "a small amount, so please check.",
    )

# 3. Run every step with the user's choices
try:
    result = run_pipeline(
        df,
        PipelineOptions(
            mapping=mapping,
            fix_text=fix_text,
            remove_duplicates=remove_duplicates,
            excluded_statuses=excluded_statuses,
            leave_out_future=leave_out_future,
            discount_type=discount_type,
        ),
    )
except ValueError as error:
    st.error(str(error))
    st.stop()

if not result.text_merges.empty:
    fixed = int(result.text_merges["rows_changed"].sum())
    with st.expander(f"Text fixed: {fixed:,} values"):
        st.dataframe(result.text_merges, hide_index=True, width="stretch")
problems = [c for c in result.checks["checks"] if c["rows_flagged"]]
with st.expander(f"Data checks ({len(problems)} problem types found)"):
    if problems:
        st.dataframe(pd.DataFrame(problems), hide_index=True, width="stretch")
    else:
        st.write("No problems found by the business rules.")
    if result.checks["skipped"]:
        st.caption(
            "Not checked: "
            + "; ".join(
                f"{s['name']} ({s['reason']})" for s in result.checks["skipped"]
            )
        )
if result.discount and result.discount["invalid_discount_rows"]:
    st.warning(
        f"{result.discount['invalid_discount_rows']:,} rows have an impossible "
        "discount and are left out of revenue."
    )

# 4. Revenue
st.subheader("Revenue")
summary = result.summary
if result.discount:
    r1, r2, r3 = st.columns(3)
    r1.metric("Gross sales (before discounts)", money(result.gross_total))
    r2.metric("Discounts given", money(result.discount["total_discount"]))
    r3.metric("Net revenue", money(summary["total_revenue"]))
else:
    st.metric("Total revenue", money(summary["total_revenue"]))
e1, e2, e3, e4, e5 = st.columns(5)
e1.metric("Rows used", f"{summary['rows_included']:,}")
e2.metric("Missing qty/price", f"{summary['rows_excluded']:,}")
e3.metric("Duplicates removed", f"{result.duplicates_removed:,}")
e4.metric("Left out by status", f"{result.status_excluded:,}")
e5.metric("Future dates left out", f"{result.future_left_out:,}")
if summary["rows_excluded"]:
    st.caption(
        f"{summary['rows_excluded']:,} rows were left out because quantity or price "
        "is missing, so actual revenue is higher than shown."
    )
elif result.discount:
    st.caption("Net revenue = quantity x unit price, after discounts, before tax.")
else:
    st.caption("Revenue = quantity x unit price (gross, before discounts).")

kpis = result.kpis
if kpis["orders"]:
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Orders", f"{kpis['orders']:,}")
    k2.metric("Units sold", f"{kpis['units_sold']:,.0f}")
    k3.metric("Average order value", money(kpis["average_order_value"]))
    k4.metric("Median order value", money(kpis["median_order_value"]))
    st.caption(
        "A typical order is the median. When the average is much higher, a few "
        "very large orders are pulling it up."
        + (
            ""
            if kpis["orders_counted_by"] == "order_id"
            else " Each row is counted as one order because no order ID is mapped."
        )
    )

check = result.reconciliation
if check:
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

# 5. Monthly trend, yearly growth and forecast
if result.monthly is not None:
    st.subheader("Monthly revenue")
    st.line_chart(result.monthly.set_index("month")["revenue"])
    if result.undated_rows:
        st.caption(f"{result.undated_rows:,} rows have no date and are not shown.")
    st.dataframe(result.yearly, hide_index=True)
    st.caption(
        "Growth is only shown between two complete years (12 months of data each)."
    )
    forecast = result.forecast
    if forecast["available"]:
        st.metric(f"Forecast for {forecast['month']}", money(forecast["forecast"]))
        st.caption(
            f"Method: {forecast['method'].replace('_', ' ')}. Typical error when "
            f"tested on the last 12 months: {forecast['backtest_error_percent']}%."
        )
    else:
        st.info(f"No forecast: {forecast['reason']}")
else:
    st.info("Map an order_date column to see the monthly trend and forecast.")

# 6. Revenue breakdown by any column (product, category, city, payment method...)
st.subheader("Revenue by")
data = result.data
skip = {"order_id", "order_date", "quantity", "unit_price", "list_price"}
choices = [
    str(c)
    for c in data.columns
    if str(c) not in skip
    and not pd.api.types.is_numeric_dtype(data[c])
    and 2 <= data[c].nunique() <= 100
]
if choices:
    default = choices.index("product") if "product" in choices else 0
    group_by = st.selectbox("Break revenue down by", choices, index=default)
    breakdown = revenue_by(data, group_by)
    st.bar_chart(
        breakdown.head(10), x=group_by, y="revenue", horizontal=True, sort="-revenue"
    )
    concentration = revenue_concentration(breakdown)
    if concentration["groups"]:
        st.caption(
            f"Top 1 brings {concentration['top_1_share']}% of revenue, top 3 bring "
            f"{concentration['top_3_share']}%. {concentration['groups_for_80_percent']}"
            f" of {concentration['groups']} together make up 80% of revenue."
        )
    st.dataframe(breakdown, hide_index=True, width="stretch")
else:
    st.info("No text column with a manageable number of values to break down by.")
