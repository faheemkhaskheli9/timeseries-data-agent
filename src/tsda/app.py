"""Streamlit entrypoint: ``streamlit run src/tsda/app.py``.

Phase 1 scope: upload a CSV/JSON file, load it into a DataFrame held in
session state, preview it, auto-detect the date/value columns with a manual
override, and show a data profile summary. The question box arrives in a
later phase.
"""

from __future__ import annotations

import streamlit as st

from tsda.loader import LoadError, load_timeseries
from tsda.profile import MAX_PROFILE_ROWS, ProfileError, profile_dataframe
from tsda.schema import SchemaError, apply_schema, infer_schema

st.set_page_config(page_title="Time-Series Analytics Chatbot", layout="wide")
st.title("Time-Series Analytics Chatbot")
st.caption("Phase 1 — upload a CSV or JSON file to get started.")

uploaded = st.file_uploader("Upload time-series data", type=["csv", "json"])

if uploaded is not None:
    try:
        frame = load_timeseries(uploaded.name, uploaded.getvalue())
    except LoadError as exc:
        st.error(str(exc))
    else:
        st.session_state["df"] = frame
        st.session_state["source_name"] = uploaded.name
        # A new upload gets fresh detection; an override picked for the
        # previous file shouldn't silently carry over to this one.
        st.session_state.pop("date_column_override", None)
        st.success(f"Loaded **{uploaded.name}** — {frame.shape[0]} rows x {frame.shape[1]} columns.")

if "df" in st.session_state:
    frame = st.session_state["df"]
    st.subheader(f"Preview: {st.session_state.get('source_name', 'dataset')}")
    st.dataframe(frame.head(50), use_container_width=True)
    st.write("**Column types**")
    st.write({col: str(dtype) for col, dtype in frame.dtypes.items()})

    inference = infer_schema(frame)
    options = ["(none)"] + list(frame.columns)
    default_col = inference.date_column or "(none)"
    chosen = st.selectbox(
        "Detected date column (override if wrong)",
        options,
        index=options.index(default_col) if default_col in options else 0,
    )
    date_column = None if chosen == "(none)" else chosen
    if date_column and date_column != inference.date_column:
        st.caption(f"Overridden — auto-detected: {inference.date_column or 'none found'}")
    elif inference.date_column:
        st.caption(f"Auto-detected with {inference.date_parse_ratio:.0%} of values parsing as dates.")

    st.write("**Numeric columns**", list(inference.numeric_columns))
    st.write("**Categorical columns**", list(inference.categorical_columns))

    if date_column:
        try:
            sorted_frame = apply_schema(frame, inference, date_column=date_column)
        except SchemaError as exc:
            st.error(str(exc))
        else:
            st.session_state["normalized_df"] = sorted_frame
            st.session_state["date_column"] = date_column
            st.write("**Chronologically sorted preview**")
            st.dataframe(sorted_frame.head(50), use_container_width=True)
    else:
        st.info("No date column selected — pick one above to sort the data chronologically.")

    st.subheader("Data profile")
    try:
        profile = profile_dataframe(frame, date_column=date_column)
    except ProfileError as exc:
        st.error(str(exc))
    else:
        if profile.sampled:
            st.caption(
                f"Stats computed on the first {profile.sample_size:,} of "
                f"{profile.row_count:,} rows (limit: {MAX_PROFILE_ROWS:,})."
            )
        cols = st.columns(3)
        cols[0].metric("Rows", f"{profile.row_count:,}")
        cols[1].metric("Columns", len(profile.columns))
        if profile.date_range:
            cols[2].metric("Date range", f"{profile.date_range[0][:10]} to {profile.date_range[1][:10]}")

        st.write("**% missing per column**")
        st.write({col: f"{pct:.1f}%" for col, pct in profile.missing_pct.items()})

        if profile.numeric_stats:
            st.write("**Summary statistics (numeric columns)**")
            st.dataframe(
                {
                    col: {"mean": s.mean, "min": s.min, "max": s.max, "std": s.std}
                    for col, s in profile.numeric_stats.items()
                },
                use_container_width=True,
            )
else:
    st.info("No dataset loaded yet.")
