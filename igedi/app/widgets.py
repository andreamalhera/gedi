import math
import streamlit as st

from igedi.app.config import CONFIG_HELP, INT_CONFIG_BOUNDS
from igedi.app.features import feature_description
from igedi.app.state import init_state

def config_space_inputs(config_space):
    """Typed controls for the generator's config space; updates config_space in place."""
    cols = st.columns(2)
    for i, (key, default) in enumerate(config_space.items()):
        default = list(default) if isinstance(default, (list, tuple)) else [default]
        state_key = f"config_{key}"
        with cols[i % 2]:
            if key == "time_spread":
                value = st.number_input(key, min_value=0.1, max_value=10.0, step=0.1, help=CONFIG_HELP[key],
                                        **init_state(state_key, float(default[0])))
                config_space[key] = [value]
            elif key in INT_CONFIG_BOUNDS:
                low, high = INT_CONFIG_BOUNDS[key]
                high = max(high, int(max(default)))
                value = st.slider(key, low, high, step=1, help=CONFIG_HELP.get(key),
                                  **init_state(state_key, (int(min(default)), int(max(default)))))
                config_space[key] = list(value)
            elif len(default) == 1:  # fixed probability, e.g. duplicate: [0]
                value = st.slider(key, 0.0, 1.0, step=0.01, help=CONFIG_HELP.get(key),
                                  **init_state(state_key, float(default[0])))
                config_space[key] = [value]
            else:  # probability range
                value = st.slider(key, 0.0, 1.0, step=0.01, help=CONFIG_HELP.get(key),
                                  **init_state(state_key, (float(min(default)), float(max(default)))))
                config_space[key] = [round(v, 2) for v in value]

def choose(label, off_option, on_option, key, value=False, help=None):
    """A labeled two-way choice; returns True when on_option is selected."""
    options = [off_option, on_option]
    selected = st.radio(label, options, horizontal=True, help=help,
                        **init_state(key, options[int(value)]))
    return selected == on_option

def multi_button(labels):
    cols = st.columns(len(labels))
    activations = []
    for col, label in zip(cols, labels):
        activations.append(col.button(label))
    return activations

def input_multicolumn(labels, default_values, n_cols=5):
    result = {}
    cols = st.columns(n_cols)
    factor = math.ceil(len(labels)/n_cols)
    extended = cols.copy()
    for _ in range(factor):
        extended.extend(cols)
    for label, default_value, col in zip(labels, default_values, extended):
        with col:
            key = f"input_{label}"
            feature = label.split(":")[0]  # labels look like "<feature>: value 1" or "<feature>: min"
            result[label] = col.text_input(label, help=feature_description(feature),
                                           **init_state(key, str(default_value)))
    return result.values()
