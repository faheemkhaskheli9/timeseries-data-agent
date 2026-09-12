"""Column-type inference and schema normalization for uploaded data (issue #2).

Kept separate from the Streamlit shell, same as ``loader.py``, so detection
is unit-testable without a browser.

Detection heuristic:
1. A column pandas already parses as a datetime dtype is the date column.
2. Otherwise, try parsing every column to datetime and keep the ones that
   clear ``min_date_ratio`` of non-null values; among those, a column whose
   name hints at a date (``date``/``time``/``timestamp``/...) wins, else the
   one with the highest parse ratio.
Every other column is numeric if pandas already reports (or best-effort
coercion finds) a numeric dtype; everything left over is categorical.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

_DATE_NAME_HINTS = ("date", "time", "timestamp", "datetime", "day", "month", "year")
_DEFAULT_MIN_DATE_RATIO = 0.9


class SchemaError(Exception):
    """Raised when a DataFrame's schema can't be inferred or applied."""


@dataclass(frozen=True)
class ColumnInference:
    date_column: str | None
    numeric_columns: tuple[str, ...]
    categorical_columns: tuple[str, ...]
    date_parse_ratio: float  # fraction of the chosen date column's non-null values that parsed


def _date_parse_ratio(series: pd.Series) -> float:
    non_null = series.dropna()
    if non_null.empty:
        return 0.0
    parsed = pd.to_datetime(non_null, errors="coerce", format="mixed")
    return float(parsed.notna().mean())


def infer_schema(frame: pd.DataFrame, min_date_ratio: float = _DEFAULT_MIN_DATE_RATIO) -> ColumnInference:
    """Detect the date column plus numeric/categorical columns in ``frame``.

    Never raises for "no date column found" — that's a legitimate outcome
    the caller (UI) surfaces as "please pick one"; only an empty DataFrame
    is an error, since there is nothing to infer from.
    """

    if frame.empty:
        raise SchemaError("cannot infer schema for an empty DataFrame")

    ratios: dict[str, float] = {}
    for col in frame.columns:
        series = frame[col]
        if pd.api.types.is_datetime64_any_dtype(series):
            ratios[col] = 1.0
        elif pd.api.types.is_numeric_dtype(series):
            # A plain int/float column ("value", "quantity") is a value
            # column, not a date, even though `pd.to_datetime` will happily
            # (mis)interpret numbers as an epoch offset. Only try to parse
            # string-like columns as dates.
            ratios[col] = 0.0
        else:
            ratios[col] = _date_parse_ratio(series)

    passing = {col: ratio for col, ratio in ratios.items() if ratio >= min_date_ratio}
    date_column: str | None = None
    if passing:
        hinted = [col for col in passing if any(h in col.lower() for h in _DATE_NAME_HINTS)]
        date_column = hinted[0] if hinted else max(passing, key=passing.get)

    date_ratio = ratios.get(date_column, 0.0) if date_column else 0.0

    remaining = [c for c in frame.columns if c != date_column]
    numeric_columns = tuple(c for c in remaining if pd.api.types.is_numeric_dtype(frame[c]))
    categorical_columns = tuple(c for c in remaining if c not in numeric_columns)

    return ColumnInference(
        date_column=date_column,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        date_parse_ratio=date_ratio,
    )


def apply_schema(
    frame: pd.DataFrame,
    inference: ColumnInference,
    date_column: str | None = None,
) -> pd.DataFrame:
    """Parse the date column to datetime and sort the frame chronologically.

    ``date_column`` lets the caller override ``inference.date_column`` (the
    UI's "manually override the detected column" path) without re-running
    detection.
    """

    col = date_column if date_column is not None else inference.date_column
    if col is None:
        raise SchemaError("no date column detected or provided")
    if col not in frame.columns:
        raise SchemaError(f"date column {col!r} not found in the data")

    result = frame.copy()
    result[col] = pd.to_datetime(result[col], errors="coerce", format="mixed")
    if result[col].isna().all():
        raise SchemaError(f"column {col!r} could not be parsed as dates")
    return result.sort_values(col, kind="stable").reset_index(drop=True)
