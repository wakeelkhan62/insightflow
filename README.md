# InsightFlow AI

Turn messy sales data into revenue numbers you can trust.

InsightFlow takes a sales CSV, checks it for problems, cleans it with your approval,
calculates revenue, and forecasts next month. Every number comes with its evidence:
how many rows it is based on, how many were left out, and why.

> **Status:** working MVP. It runs locally with a Streamlit web app and a FastAPI
> backend. Login, a database, the AI chat and deployment are not built yet
> (see [Roadmap](#roadmap)).

## What it does

1. **Load** a CSV file (up to 50 MB by default), keeping whole numbers as whole numbers.
2. **Check quality:** missing values per column, exact duplicate rows, conflicting
   IDs (same order ID, different data), column types, and columns that contain dates.
3. **Map columns:** recognises names like `Qty`, `UnitPrice`, `OrderID` or
   `Rate (PKR)` and suggests which standard field each one is. You confirm it.
4. **Clean, with approval:** convert date columns, remove exact duplicate rows
   (and report which ones), and fill missing values by product group with an
   audit trail and `<column>_was_filled` flag columns.
5. **Analyse:** total revenue, revenue by product, and revenue by month (with
   empty months shown as 0). Money is rounded to 2 decimals.
6. **Forecast** next month's revenue. Three simple methods are back-tested on the
   last 12 months; the most accurate one makes the forecast, and its error is shown.

## Principles

- **The original data is never changed silently.** Every cleaning step returns a
  new copy and a record of what changed.
- **The system suggests, the user decides** (column mapping, duplicates, fill strategy).
- **Every number has evidence** (rows used, rows excluded, forecast error %).
- **Unknown is not zero.** Missing revenue stays missing; values are never invented
  without a reliable source.

## Quick start (Windows PowerShell)

Requirements: Python 3.13 and [uv](https://docs.astral.sh/uv/).

```powershell
git clone <your-repo-url> InsightFlow
cd InsightFlow
uv sync
```

**Run the web app:**

```powershell
uv run streamlit run src/insightflow/streamlit_app.py
```

Open http://localhost:8501, upload a CSV, check the suggested columns, and read
the results.

**Run the API:**

```powershell
uv run fastapi dev src/insightflow/api.py
```

Interactive API docs: http://127.0.0.1:8000/docs

To change the upload limit (for example to 100 MB):

```powershell
$env:INSIGHTFLOW_MAX_UPLOAD_MB = "100"
```

## API endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Shows the server is running |
| POST | `/datasets/quality-report` | Upload a CSV, get its quality report. Optional `key_column` |
| POST | `/datasets/revenue` | Upload a CSV, get revenue summary, by product, by month and a forecast |

`/datasets/revenue` accepts an optional `mapping` form field with the confirmed
column mapping, for example:

```json
{"order_id": "OrderID", "order_date": "OrderDate", "product": "ProductName",
 "quantity": "Quantity", "unit_price": "UnitPrice"}
```

If the columns do not match and no mapping is sent, the error response includes a
`suggested_mapping`.

## Project structure

```
src/insightflow/
  ingestion.py       load CSV files
  quality.py         quality checks and the combined quality report
  cleaning.py        date conversion, duplicate removal, missing-value filling
  mapping.py         suggest and apply column mapping
  analytics.py       revenue, by product, by month
  forecasting.py     back-tested next-month forecast
  api.py             FastAPI endpoints
  streamlit_app.py   Streamlit web app
tests/               one test file per module
data/sample_sales.csv  small sample with deliberate problems
```

The engine (`ingestion` to `forecasting`) has no web code in it. The API and the
Streamlit app both call the same engine functions, so the numbers always match.

## Tests and code quality

```powershell
uv run ruff check --fix .
uv run ruff format .
uv run pytest -q
uv run ruff check .
```

All checks must pass before every commit.

## Known limitations

- **Revenue is gross** (quantity x unit price). Discounts, tax, shipping and
  cancelled or returned orders are not taken into account yet.
- **CSV only.** Excel files are not supported yet.
- **Dates must be in `YYYY-MM-DD` format** to be detected and converted.
- **Column mapping is rule-based.** Unknown column names must be chosen by hand.
- **Nothing is stored.** Uploaded files are processed in memory and forgotten.
- **No login.** Anyone who can open the app can use it. Run it locally only.
- **The forecast uses simple methods** and predicts one month ahead. Read the error %.

## Roadmap

- Net revenue: discounts and excluding cancelled or returned orders
- Excel (`.xlsx`) upload
- PostgreSQL database: save datasets and analyses
- User accounts and separate workspaces per business
- AI chat (LangGraph) that answers questions using the engine's calculations
- Docker, CI and deployment
