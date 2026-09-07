import pytest

from tsda.loader import LoadError, load_timeseries

CSV = b"date,value\n2024-01-01,10\n2024-01-02,12\n2024-01-03,9\n"
JSON_RECORDS = b'[{"date": "2024-01-01", "value": 10}, {"date": "2024-01-02", "value": 12}]'
JSON_COLUMNS = b'{"date": ["2024-01-01", "2024-01-02"], "value": [10, 12]}'


def test_load_csv_happy_path():
    frame = load_timeseries("series.csv", CSV)
    assert list(frame.columns) == ["date", "value"]
    assert frame.shape == (3, 2)
    assert frame["value"].tolist() == [10, 12, 9]


def test_load_json_records():
    frame = load_timeseries("series.json", JSON_RECORDS)
    assert frame.shape == (2, 2)
    assert set(frame.columns) == {"date", "value"}


def test_load_json_columns():
    frame = load_timeseries("series.json", JSON_COLUMNS)
    assert frame.shape == (2, 2)


def test_unsupported_extension():
    with pytest.raises(LoadError, match="Unsupported file type"):
        load_timeseries("series.xlsx", b"whatever")


def test_no_extension():
    with pytest.raises(LoadError, match="Unsupported file type"):
        load_timeseries("series", CSV)


def test_empty_upload():
    with pytest.raises(LoadError, match="empty"):
        load_timeseries("series.csv", b"")


def test_malformed_csv_ragged_rows():
    bad = b'a,b,c\n1,2\n"3,4,5,6\n'
    with pytest.raises(LoadError):
        load_timeseries("bad.csv", bad)


def test_malformed_json():
    with pytest.raises(LoadError, match="parse the JSON"):
        load_timeseries("bad.json", b"{not valid json")


def test_json_scalar_is_rejected():
    with pytest.raises(LoadError):
        load_timeseries("scalar.json", b"42")


def test_csv_headers_only_is_rejected():
    with pytest.raises(LoadError, match="no rows or columns"):
        load_timeseries("empty.csv", b"date,value\n")


def test_non_utf8_csv():
    with pytest.raises(LoadError, match="UTF-8"):
        load_timeseries("latin.csv", b"date,value\n2024,\xff\xfe\n")
