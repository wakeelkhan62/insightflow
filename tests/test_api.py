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
