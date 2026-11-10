# InsightFlow AI

Turn messy sales data into revenue numbers you can trust.

Upload a sales CSV and InsightFlow checks it, cleans it with your approval,
calculates gross and net revenue, breaks it down, checks itself against the
file's own totals, and forecasts next month. Every number shows its evidence:
how many rows it is based on, what was left out, and why.

> **Status:** working MVP, tested on 2,400+ and 100,000-row files. Runs locally
> as a Streamlit web app plus a FastAPI backend. Login, a database, the AI chat
> and deployment are not built yet (see [Roadmap](#roadmap)).

## What it does

**Understand the file**
- Quality report: missing values per column, exact duplicate rows, conflicting
  IDs, column types, and date detection
- Column mapping: recognises names like `Qty`, `UnitPrice`, `Invoice No`,
  `Rate (PKR)` or `TotalAmount` for 10 standard fields; you confirm or change them

**Clean, with your approval (nothing is changed silently)**
- Fixes inconsistent text: `CANCELLED`, `cancelled ` and `Cancelled` become one value
- Removes exact duplicate rows and reports how many
- Leaves out cancelled, returned and refunded orders (you choose the statuses)
- Leaves out orders dated in the future (almost always typos)
- Fills missing values by product group (median / mean / mode / "Unknown"),
  with an audit trail and `<column>_was_filled` flags

**Check the data**
- Business rules: zero or negative quantity and price, future dates, and
  extreme values judged per product (IQR rule), with CSV line numbers
- Reconciliation: finds which formula reproduces the file's own total
  (for example `net + tax + shipping`) and lists the rows that do not match

**Analyse**
- Gross sales, discounts (fraction, percent or amount, detected and confirmed)
  and net revenue
- Orders, units sold, average and median order value
- Revenue by product, category, city, payment method or any other column,
  with an 80/20 concentration check
- Monthly trend and yearly growth (only between complete years)
- Next-month forecast: three simple methods back-tested on the last 12 months;
  the most accurate one is used and its error is shown

## Principles

- **The original data is never changed silently.** Every step returns a new copy
  and a record of what changed.
- **The system suggests, the user decides.**
- **Every number has evidence:** rows used, rows left out, and the reason.
- **Unknown is not zero.** Missing revenue stays missing; values are never
  invented without a reliable source.

## Quick start (Windows PowerShell)

Requirements: Python 3.13 and [uv](https://docs.astral.sh/uv/).

```powershell
git clone https://github.com/wakeelkhan62/insightflow.git
cd insightflow
uv sync
uv run streamlit run src/insightflow/streamlit_app.py
```

Open http://localhost:8501 and upload a CSV. To try it straight away, use
`data/insightflow_test_sales.csv`: a synthetic 3-year dataset with deliberate
problems (duplicates, messy statuses, a future date, wrong totals and more).

**API** (FastAPI): `uv run fastapi dev src/insightflow/api.py`, then open
http://127.0.0.1:8000/docs

## Project structure

```
src/insightflow/
  ingestion.py        load CSV files
  quality.py          quality checks and the combined quality report
  mapping.py          suggest and apply column mapping (10 standard fields)
  cleaning.py         text fixing, dates, duplicates, status, future dates, filling
  validation.py       business rule checks with CSV line numbers
  discounts.py        discount type detection and net revenue
  reconciliation.py   check revenue against the file's own total column
  analytics.py        revenue, KPIs, breakdowns, concentration, monthly, yearly
  forecasting.py      back-tested next-month forecast
  streamlit_app.py    web app
  api.py              FastAPI endpoints
tests/                one test file per module
```

The engine modules contain no web code. The app and the API call the same
functions, so the numbers always match.

## Tests and code quality

```powershell
uv run ruff check --fix .
uv run ruff format .
uv run pytest -q
uv run ruff check .
```

89 tests. All checks must pass before every commit.

## Known limitations

- **CSV only**; Excel is not supported yet.
- **Dates must be in `YYYY-MM-DD` format.**
- **Column mapping is rule-based**; unknown names must be chosen by hand.
- **The API is behind the app:** it does not yet apply text fixing, status
  filtering, future dates or discounts.
- **Nothing is stored**, and there is **no login**. Run it locally only.
- **The forecast** predicts one month ahead with simple methods. Read its error %.

## Roadmap

- One shared pipeline used by both the app and the API
- Lost-revenue analysis (cancellations and returns)
- Excel (`.xlsx`) upload
- PostgreSQL: save datasets and analyses
- User accounts and separate workspaces per business
- AI chat (LangGraph) that answers questions using the engine's calculations
- Docker, CI and deployment