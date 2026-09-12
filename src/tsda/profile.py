"""Data profile summary shown right after upload (issue #3).

Kept separate from the Streamlit shell, same as ``loader.py``/``schema.py``,
so profiling is unit-testable without a browser.

Performance / size limit: profiling reads at most ``MAX_PROFILE_ROWS`` rows
so it stays a bounded, sub-second operation regardless of upload size — the
documented size limit acceptance criterion #3 asks for. ``row_count`` always
reports the full uploaded row count; when the frame is larger than the cap,
summary statistics and missing-value percentages are computed on a
deterministic head-sample instead, and ``DataProfile.sampled`` is set so the
UI can disclose that the stats are approximate.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

MAX_PROFILE_ROWS = 200_000


class ProfileError(Exception):
    """Raised when a profile can't be computed."""


@dataclass(frozen=True)
class NumericColumnStats:
    mean: float
    min: float
    max: float
    std: float


@dataclass(frozen=True)
class DataProfile:
    row_count: int
    columns: tuple[str, ...]
    missing_pct: dict[str, float] = field(default_factory=dict)
    numeric_stats: dict[str, NumericColumnStats] = field(default_factory=dict)
    date_range: tuple[str, str] | None = None
    sampled: bool = False
    sample_size: int | None = None


def profile_dataframe(
    frame: pd.DataFrame,
    date_column: str | None = None,
    *,
    max_rows: int = MAX_PROFILE_ROWS,
) -> DataProfile:
    """Compute row count, column list, % missing per column, per-numeric-
    column summary stats, and (if a date column is known) the date range.

    Never returns a partial result silently: an empty DataFrame is a clear
    error, not a profile full of zeros/NaNs that looks valid.
    """
    if frame.empty or frame.shape[1] == 0:
        raise ProfileError("cannot profile an empty DataFrame")

    row_count = len(frame)
    sampled = row_count > max_rows
    stats_frame = frame.head(max_rows) if sampled else frame

    missing_pct = {
        col: float(stats_frame[col].isna().mean() * 100.0) for col in stats_frame.columns
    }

    numeric_stats: dict[str, NumericColumnStats] = {}
    for col in stats_frame.columns:
        series = stats_frame[col]
        if not pd.api.types.is_numeric_dtype(series):
            continue
        non_null = series.dropna()
        if non_null.empty:
            continue
        numeric_stats[col] = NumericColumnStats(
            mean=float(non_null.mean()),
            min=float(non_null.min()),
            max=float(non_null.max()),
            std=float(non_null.std()) if len(non_null) > 1 else 0.0,
        )

    date_range: tuple[str, str] | None = None
    if date_column is not None:
        if date_column not in frame.columns:
            raise ProfileError(f"date column {date_column!r} not found in the data")
        parsed = pd.to_datetime(frame[date_column], errors="coerce", format="mixed")
        valid = parsed.dropna()
        if not valid.empty:
            date_range = (valid.min().isoformat(), valid.max().isoformat())

    return DataProfile(
        row_count=row_count,
        columns=tuple(frame.columns),
        missing_pct=missing_pct,
        numeric_stats=numeric_stats,
        date_range=date_range,
        sampled=sampled,
        sample_size=len(stats_frame) if sampled else None,
    )
