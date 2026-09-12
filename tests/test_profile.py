import numpy as np
import pandas as pd
import pytest

from tsda.profile import ProfileError, profile_dataframe


def _frame():
    return pd.DataFrame(
        {
            "date": ["2024-01-03", "2024-01-01", "2024-01-02"],
            "value": [9, 10, None],
            "region": ["east", "west", "east"],
        }
    )


def test_profile_reports_row_count_and_columns():
    profile = profile_dataframe(_frame())
    assert profile.row_count == 3
    assert profile.columns == ("date", "value", "region")


def test_profile_computes_missing_pct_per_column():
    profile = profile_dataframe(_frame())
    assert profile.missing_pct["value"] == pytest.approx(100 / 3)
    assert profile.missing_pct["date"] == 0.0
    assert profile.missing_pct["region"] == 0.0


def test_profile_computes_numeric_stats_for_numeric_columns_only():
    profile = profile_dataframe(_frame())
    assert set(profile.numeric_stats) == {"value"}
    stats = profile.numeric_stats["value"]
    assert stats.mean == pytest.approx(9.5)
    assert stats.min == 9.0
    assert stats.max == 10.0


def test_profile_computes_date_range_when_date_column_given():
    profile = profile_dataframe(_frame(), date_column="date")
    assert profile.date_range == ("2024-01-01T00:00:00", "2024-01-03T00:00:00")


def test_profile_without_date_column_has_no_date_range():
    profile = profile_dataframe(_frame())
    assert profile.date_range is None


def test_profile_unknown_date_column_raises():
    with pytest.raises(ProfileError, match="not found"):
        profile_dataframe(_frame(), date_column="nope")


def test_profile_empty_frame_raises():
    with pytest.raises(ProfileError, match="empty"):
        profile_dataframe(pd.DataFrame())


def test_profile_all_null_numeric_column_is_skipped_not_a_crash():
    frame = pd.DataFrame({"a": [1, 2], "b": [np.nan, np.nan]})
    profile = profile_dataframe(frame)
    assert "b" not in profile.numeric_stats
    assert profile.missing_pct["b"] == 100.0


def test_profile_single_row_std_is_zero_not_nan():
    frame = pd.DataFrame({"a": [1]})
    profile = profile_dataframe(frame)
    assert profile.numeric_stats["a"].std == 0.0


def test_profile_large_frame_is_sampled_and_flagged():
    frame = pd.DataFrame({"a": range(10)})
    profile = profile_dataframe(frame, max_rows=4)

    assert profile.row_count == 10  # full count, not the sample size
    assert profile.sampled is True
    assert profile.sample_size == 4
    # Stats reflect only the first 4 rows (0..3), not the full column.
    assert profile.numeric_stats["a"].max == 3.0


def test_profile_frame_within_limit_is_not_sampled():
    frame = pd.DataFrame({"a": range(3)})
    profile = profile_dataframe(frame, max_rows=4)
    assert profile.sampled is False
    assert profile.sample_size is None
