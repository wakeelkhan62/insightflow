from insightflow.ingestion import load_csv


def test_load_csv_reads_rows_and_columns(tmp_path):
    """A small CSV loads with the right number of rows and column names."""
    csv_file = tmp_path / "sales.csv"
    csv_file.write_text("order_id,product\n1,Laptop\n2,Mouse\n")

    df = load_csv(csv_file)

    assert len(df) == 2
    assert list(df.columns) == ["order_id", "product"]


def test_whole_numbers_stay_integers_when_values_are_missing(tmp_path):
    """An empty cell must not turn a whole-number column into decimals."""
    csv_file = tmp_path / "sales.csv"
    csv_file.write_text("order_id,quantity\n1,2\n2,\n")

    df = load_csv(csv_file)

    assert str(df["quantity"].dtype) == "Int64"
    assert df["quantity"].isna().sum() == 1
