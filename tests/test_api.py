import pandas as pd
from fastapi.testclient import TestClient

from insightflow import api
from insightflow.api import app

client = TestClient(app)

SAMPLE_CSV = b"order_id,product,quantity\n1,Laptop,2\n1,Laptop,2\n2,Mouse,\n"


def _upload(content: bytes, filename: str = "sales.csv", **params):
    files = {"file": (filename, content, "text/csv")}
    return client.post("/datasets/quality-report", files=files, params=params)


def test_health_returns_ok():
    """The health endpoint answers 200 with a simple status."""
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_unknown_url_returns_404():
    """A URL that does not exist gives a clean 404, not a server error."""
    response = client.get("/does-not-exist")

    assert response.status_code == 404


def test_upload_returns_quality_report():
    """A valid CSV upload returns its quality report as JSON."""
    response = _upload(SAMPLE_CSV, key_column="order_id")

    assert response.status_code == 200
    report = response.json()
    assert report["row_count"] == 3
    assert report["duplicate_rows"] == 1
    assert report["conflicting_keys"] == 0
    assert report["missing_values"]["quantity"] == 1


def test_non_csv_file_is_rejected():
    """Only .csv files are accepted."""
    response = _upload(SAMPLE_CSV, filename="sales.exe")

    assert response.status_code == 400
    assert "csv" in response.json()["detail"]


def test_empty_file_is_rejected():
    """An empty upload gets a clear message instead of a crash."""
    response = _upload(b"")

    assert response.status_code == 400
    assert "empty" in response.json()["detail"]


def test_too_large_file_is_rejected(monkeypatch):
    """Files over the size limit are refused with 413."""
    monkeypatch.setattr(api, "MAX_UPLOAD_BYTES", 10)

    response = _upload(SAMPLE_CSV)

    assert response.status_code == 413


def test_unknown_key_column_returns_400():
    """A key column that is not in the file gives a clear 400 error."""
    response = _upload(SAMPLE_CSV, key_column="orderid")

    assert response.status_code == 400
    assert "orderid" in response.json()["detail"]


REVENUE_CSV = (
    b"order_id,product,quantity,unit_price\n"
    b"1,Laptop,2,85000\n"
    b"1,Laptop,2,85000\n"
    b"2,Mouse,5,1500\n"
    b"3,Keyboard,,3000\n"
)


def _upload_revenue(content: bytes, **params):
    files = {"file": ("sales.csv", content, "text/csv")}
    return client.post("/datasets/revenue", files=files, params=params)


def test_revenue_endpoint_removes_duplicates_and_returns_evidence():
    """Duplicates are removed and counted; totals come with included/excluded rows."""
    response = _upload_revenue(REVENUE_CSV)

    assert response.status_code == 200
    body = response.json()
    assert body["duplicates_removed"] == 1
    assert body["summary"] == {
        "total_revenue": 177500.0,
        "rows_included": 2,
        "rows_excluded": 1,
    }
    assert body["by_product"][0] == {
        "product": "Laptop",
        "revenue": 170000,
        "rows_included": 1,
        "rows_excluded": 0,
    }
    assert body["by_product"][2]["revenue"] is None


def test_revenue_endpoint_can_keep_duplicates():
    """With remove_duplicates=false, the copied row is counted again."""
    response = _upload_revenue(REVENUE_CSV, remove_duplicates=False)

    assert response.json()["summary"]["total_revenue"] == 347500.0


def test_revenue_endpoint_suggests_mapping_when_columns_differ():
    """Non-standard column names give a 400 with a suggested mapping."""
    content = b"Item,Qty,Rate (PKR)\nLaptop,2,85000\n"

    response = _upload_revenue(content)

    assert response.status_code == 400
    assert response.json()["detail"]["suggested_mapping"] == {
        "product": "Item",
        "quantity": "Qty",
        "unit_price": "Rate (PKR)",
    }


MAPPED_CSV = b"Item,Qty,Rate (PKR)\nLaptop,2,85000\nMouse,5,1500\n"
MAPPING = '{"product": "Item", "quantity": "Qty", "unit_price": "Rate (PKR)"}'


def _upload_revenue_with_mapping(content: bytes, mapping: str):
    files = {"file": ("sales.csv", content, "text/csv")}
    return client.post("/datasets/revenue", files=files, data={"mapping": mapping})


def test_revenue_endpoint_applies_confirmed_mapping():
    """With a confirmed mapping, a file with its own column names works."""
    response = _upload_revenue_with_mapping(MAPPED_CSV, MAPPING)

    assert response.status_code == 200
    body = response.json()
    assert body["summary"]["total_revenue"] == 177500.0
    assert body["mapping_applied"]["quantity"] == "Qty"


def test_revenue_endpoint_rejects_invalid_mapping_json():
    """Mapping text that is not JSON gives a clear 400."""
    response = _upload_revenue_with_mapping(MAPPED_CSV, "quantity=Qty")

    assert response.status_code == 400
    assert "valid JSON" in response.json()["detail"]


def test_revenue_endpoint_rejects_mapping_to_missing_column():
    """A mapping that names a column not in the file gives a clear 400."""
    response = _upload_revenue_with_mapping(MAPPED_CSV, '{"quantity": "Units"}')

    assert response.status_code == 400
    assert "Units" in response.json()["detail"]


def test_revenue_endpoint_includes_monthly_trend_when_dates_exist():
    """With an order_date column, the response includes revenue per month."""
    content = (
        b"order_date,product,quantity,unit_price\n"
        b"2026-01-05,Laptop,1,100\n"
        b"2026-03-02,Mouse,2,10\n"
    )

    body = _upload_revenue(content).json()

    assert [m["month"] for m in body["by_month"]["months"]] == [
        "2026-01",
        "2026-02",
        "2026-03",
    ]
    assert [m["revenue"] for m in body["by_month"]["months"]] == [100, 0, 20]
    assert body["by_month"]["undated_rows"] == 0


def test_revenue_endpoint_without_dates_has_no_monthly_trend():
    """Without an order_date column, by_month is null instead of an error."""
    body = _upload_revenue(REVENUE_CSV).json()

    assert body["by_month"] is None


def test_revenue_endpoint_reports_invalid_dates():
    """A date that cannot be read gives a clear 400 instead of a crash."""
    content = b"order_date,product,quantity,unit_price\nsoon,Laptop,1,100\n"

    response = _upload_revenue(content)

    assert response.status_code == 400
    assert "not valid dates" in response.json()["detail"]


def test_revenue_endpoint_forecasts_next_month_with_enough_history():
    """With 15+ months of data, the monthly section includes a forecast."""
    months = pd.period_range("2024-01", periods=15, freq="M")
    rows = "".join(f"{m}-10,Pen,1,100\n" for m in months)
    content = ("order_date,product,quantity,unit_price\n" + rows).encode()

    forecast = _upload_revenue(content).json()["by_month"]["forecast"]

    assert forecast["available"] is True
    assert forecast["month"] == "2025-04"
    assert forecast["forecast"] == 100


def test_revenue_endpoint_explains_when_forecast_is_not_possible():
    """With only a few months, the forecast says why instead of failing."""
    content = b"order_date,product,quantity,unit_price\n2026-01-05,Pen,1,100\n"

    forecast = _upload_revenue(content).json()["by_month"]["forecast"]

    assert forecast["available"] is False
    assert "Need at least" in forecast["reason"]


PIPELINE_CSV = (
    b"order_id,order_date,product,quantity,unit_price,discount,order_status\n"
    b"A,2026-01-05,Pen,2,100,0.1,Delivered\n"
    b"B,2026-01-06,Pen,1,100,0,cancelled \n"
    b"C,2026-02-01,Cup,3,50,0.2,DELIVERED\n"
)


def test_revenue_endpoint_uses_the_same_pipeline_as_the_app():
    """Cancelled orders are left out and discounts give net revenue (180 + 120)."""
    body = _upload_revenue(PIPELINE_CSV).json()

    assert body["excluded_statuses"] == ["cancelled"]
    assert body["status_excluded"] == 1
    assert body["gross_total"] == 350.0
    assert body["discount"]["discount_type"] == "fraction"
    assert body["summary"]["total_revenue"] == 300.0
    assert body["text_values_fixed"] == 2


def test_revenue_endpoint_accepts_the_users_status_choice():
    """Sending an empty list keeps every order, including the cancelled one."""
    files = {"file": ("sales.csv", PIPELINE_CSV, "text/csv")}
    response = client.post(
        "/datasets/revenue", files=files, data={"excluded_statuses": "[]"}
    )

    assert response.json()["summary"]["total_revenue"] == 400.0


def test_revenue_endpoint_rejects_invalid_status_list():
    """A status list that is not JSON gives a clear 400."""
    files = {"file": ("sales.csv", PIPELINE_CSV, "text/csv")}
    response = client.post(
        "/datasets/revenue", files=files, data={"excluded_statuses": "Cancelled"}
    )

    assert response.status_code == 400
    assert "valid JSON" in response.json()["detail"]
