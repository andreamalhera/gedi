import tempfile
import streamlit as st

from igedi.app.config import EXAMPLE_FEATURES, EXAMPLE_VALUES

def session_workdir():
    if "workdir" not in st.session_state:
        st.session_state["workdir"] = tempfile.mkdtemp(prefix="igedi_")
    return st.session_state["workdir"]

def load_example():
    """Preset: a 3x3 grid over two features, i.e. 9 target logs."""
    st.session_state["pipeline_steps"] = ["event_logs_generation"]
    st.session_state["choice_grid"] = "A grid of logs"
    st.session_state["choice_source"] = "Type values"
    st.session_state["choice_manual_grid"] = "List of values per feature"
    st.session_state["choice_subsets"] = "All selected features together"
    st.session_state["num_values"] = 3
    st.session_state["chosen_features"] = EXAMPLE_FEATURES
    for feature, values in EXAMPLE_VALUES.items():
        for i, value in enumerate(values):
            st.session_state[f"input_{feature}: value {i+1}"] = str(value)
    st.session_state["n_trials"] = 20

def stop_with_error(message):
    st.error(message)
    st.stop()

def init_state(key, value):
    # Widgets get their defaults through session state, never through value=/default=.
    # That keeps the widget ID stable across reruns and lets load_example preset them.
    if key not in st.session_state:
        st.session_state[key] = value
    return {"key": key}
