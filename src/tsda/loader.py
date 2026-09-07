"""Parse an uploaded CSV/JSON file into a pandas DataFrame.

Kept separate from the Streamlit shell so the parsing rules are unit-testable
without a browser. Every failure path raises :class:`LoadError` with a
user-facing message — the UI shows it via ``st.error`` instead of crashing.
"""

from __future__ import annotations

import io
import json
from pathlib import PurePosixPath

import pandas as pd

SUPPORTED_SUFFIXES = {".csv", ".json"}


class LoadError(Exception):
    """Raised when an upload cannot be turned into a usable DataFrame."""


def _load_csv(data: bytes) -> pd.DataFrame:
    try:
        return pd.read_csv(io.BytesIO(data))
    except UnicodeDecodeError as exc:
        raise LoadError("The CSV file is not valid UTF-8 text.") from exc
    except pd.errors.EmptyDataError as exc:
        raise LoadError("The CSV file has no data.") from exc
    except (pd.errors.ParserError, ValueError) as exc:
        raise LoadError(f"Could not parse the CSV file: {exc}") from exc


def _load_json(data: bytes) -> pd.DataFrame:
    try:
        parsed = json.loads(data.decode("utf-8"))
    except UnicodeDecodeError as exc:
        raise LoadError("The JSON file is not valid UTF-8 text.") from exc
    except json.JSONDecodeError as exc:
        raise LoadError(f"Could not parse the JSON file: {exc}") from exc

    # Accept the two common shapes: a list of row objects, or a dict of
    # column -> values. Anything else is ambiguous — fail loudly.
    try:
        if isinstance(parsed, list):
            frame = pd.DataFrame.from_records(parsed)
        elif isinstance(parsed, dict):
            frame = pd.DataFrame(parsed)
        else:
            raise LoadError(
                "JSON must be a list of records or an object of columns."
            )
    except (ValueError, TypeError) as exc:
        raise LoadError(f"Could not build a table from the JSON file: {exc}") from exc
    return frame


def load_timeseries(filename: str, data: bytes) -> pd.DataFrame:
    """Load ``data`` (raw file bytes) named ``filename`` into a DataFrame."""

    if not data:
        raise LoadError("The uploaded file is empty.")

    suffix = PurePosixPath(filename).suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise LoadError(
            f"Unsupported file type {suffix or '(none)'!r}. Upload a .csv or .json file."
        )

    frame = _load_csv(data) if suffix == ".csv" else _load_json(data)

    if frame.empty or frame.shape[1] == 0:
        raise LoadError("The file parsed successfully but contains no rows or columns.")
    return frame
