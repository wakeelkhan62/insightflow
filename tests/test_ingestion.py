from insightflow.ingestion import load_csv


def test_load_csv_reads_rows_and_columns(tmp_path):
    csv_file = tmp_path / "sales.csv"
    csv_file.write_text("order_id,product\n1,Laptop\n2,Mouse\n")

    df = load_csv(csv_file)

    assert len(df) == 2
    assert list(df.columns) == ["order_id", "product"]
