import pandas as pd
import pytest

from tsda.schema import SchemaError, apply_schema, infer_schema


def _frame():
    return pd.DataFrame(
        {
            "date": ["2024-01-03", "2024-01-01", "2024-01-02"],
            "value": [9, 10, 12],
            "region": ["east", "west", "east"],
        }
    )


def test_infer_schema_detects_date_numeric_and_categorical_columns():
    inference = infer_schema(_frame())
    assert inference.date_column == "date"
    assert inference.numeric_columns == ("value",)
    assert inference.categorical_columns == ("region",)
    assert inference.date_parse_ratio == 1.0


def test_infer_schema_prefers_name_hinted_column_when_multiple_parse():
    # Both "timestamp" and "value" happen to be date-parseable strings, but
    # only "timestamp" carries a date-ish name.
    frame = pd.DataFrame(
        {
            "timestamp": ["2024-01-01", "2024-01-02"],
            "value": ["2024-02-01", "2024-02-02"],
        }
    )
    inference = infer_schema(frame)
    assert inference.date_column == "timestamp"


def test_infer_schema_no_date_column_found():
    frame = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    inference = infer_schema(frame)
    assert inference.date_column is None
    assert inference.numeric_columns == ("a",)
    assert inference.categorical_columns == ("b",)


def test_infer_schema_partial_date_parse_below_threshold_is_not_detected():
    frame = pd.DataFrame({"maybe_date": ["2024-01-01", "not-a-date", "also-not"]})
    inference = infer_schema(frame, min_date_ratio=0.9)
    assert inference.date_column is None


def test_infer_schema_already_datetime_dtype_is_detected():
    frame = pd.DataFrame({"ts": pd.to_datetime(["2024-01-01", "2024-01-02"]), "v": [1, 2]})
    inference = infer_schema(frame)
    assert inference.date_column == "ts"
    assert inference.date_parse_ratio == 1.0


def test_infer_schema_empty_dataframe_raises():
    with pytest.raises(SchemaError):
        infer_schema(pd.DataFrame())


def test_apply_schema_parses_and_sorts_chronologically():
    frame = _frame()
    inference = infer_schema(frame)
    result = apply_schema(frame, inference)
    assert pd.api.types.is_datetime64_any_dtype(result["date"])
    assert result["value"].tolist() == [10, 12, 9]  # chronological, not original order


def test_apply_schema_override_takes_precedence_over_inference(monkeypatch):
    frame = pd.DataFrame(
        {
            "date": ["2024-01-03", "2024-01-01", "2024-01-02"],
            "other_date": ["2024-03-01", "2024-01-01", "2024-02-01"],
            "value": [9, 10, 12],
        }
    )
    inference = infer_schema(frame)  # detects "date"
    result = apply_schema(frame, inference, date_column="other_date")
    assert result["value"].tolist() == [10, 12, 9]  # sorted by other_date instead


def test_apply_schema_no_date_column_available_raises():
    frame = pd.DataFrame({"a": [1, 2, 3]})
    inference = infer_schema(frame)
    with pytest.raises(SchemaError):
        apply_schema(frame, inference)


def test_apply_schema_unknown_override_column_raises():
    frame = _frame()
    inference = infer_schema(frame)
    with pytest.raises(SchemaError):
        apply_schema(frame, inference, date_column="does_not_exist")


def test_apply_schema_unparseable_column_raises():
    frame = pd.DataFrame({"date": ["not", "a", "date"], "value": [1, 2, 3]})
    inference = infer_schema(frame)
    with pytest.raises(SchemaError):
        apply_schema(frame, inference, date_column="date")
