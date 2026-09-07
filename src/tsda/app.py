"""Streamlit entrypoint: ``streamlit run src/tsda/app.py``.

Phase 1 scope: upload a CSV/JSON file, load it into a DataFrame held in
session state, and preview it. Column auto-detection, profiling, and the
question box arrive in later issues.
"""

from __future__ import annotations

import streamlit as st

from tsda.loader import LoadError, load_timeseries

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
        st.success(f"Loaded **{uploaded.name}** — {frame.shape[0]} rows x {frame.shape[1]} columns.")

if "df" in st.session_state:
    frame = st.session_state["df"]
    st.subheader(f"Preview: {st.session_state.get('source_name', 'dataset')}")
    st.dataframe(frame.head(50), use_container_width=True)
    st.write("**Column types**")
    st.write({col: str(dtype) for col, dtype in frame.dtypes.items()})
else:
    st.info("No dataset loaded yet.")
