from itertools import product as cproduct
from itertools import combinations
import matplotlib
matplotlib.use("Agg") #Agg renders off-screen, which is all st.pyplot needs.
from pylab import *
import base64
import tempfile
import json
import math
import os
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import subprocess
import time
import shutil
import zipfile
import io
import sys

_real_getppid = os.getppid # Substitute a PID that's always valid (our own) for the duration of this import only.
os.getppid = os.getpid
from feeed.feature_extractor import extract_features

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gedi.utils.io_helpers import compute_feature_similarities, get_generated_file_name
from gedi.utils.target_validation import feature_bounds, fit_window, is_numeric, validate_target
os.getppid = _real_getppid

st.set_page_config(layout='wide')
INPUT_XES="output/inputlog_temp.xes"
LOGO_PATH="gedi/utils/logo.png"

# Hugging Face sets SPACE_ID in the Space runtime. When hosted, users share one machine, so
# output paths are per-session temp dirs (not editable) and job size is capped.
HOSTED = bool(os.environ.get("SPACE_ID"))
MAX_TARGETS = int(os.environ.get("IGEDI_MAX_TARGETS", 30))
MAX_TRIALS = int(os.environ.get("IGEDI_MAX_TRIALS", 100))

IGEDI_CITATION = ("A. Maldonado, S. A. Aryasomayajula, C. M. M. Frey, T. Seidl. "
                  "iGEDI: interactive Generating Event Data with Intentional Features. "
                  "ICPM 2024 Tool Demonstration Track, CEUR Workshop Proceedings, Vol. 3783, 2024.")

EXAMPLE_FEATURES = ["epa_normalized_variant_entropy", "ratio_most_common_variant"]
EXAMPLE_VALUES = {"epa_normalized_variant_entropy": [0.3, 0.5, 0.7],
                  "ratio_most_common_variant": [0.1, 0.3, 0.5]}

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

def get_base64_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()

def citation_button():
    components.html(f"""
        <button id="cite" title="Copy the iGEDI citation to the clipboard"
                style="font-family: 'Source Sans Pro', sans-serif; font-size: 14px; color: #808495;
                       background: transparent; border: 1px solid rgba(128, 128, 128, 0.4);
                       border-radius: 6px; padding: 4px 10px; cursor: pointer;">Copy citation</button>
        <script>
            const citation = {json.dumps(IGEDI_CITATION)};
            const button = document.getElementById("cite");
            button.onclick = async () => {{
                try {{
                    await navigator.clipboard.writeText(citation);
                }} catch (error) {{
                    const area = document.createElement("textarea");
                    area.value = citation;
                    document.body.appendChild(area);
                    area.select();
                    document.execCommand("copy");
                    area.remove();
                }}
                button.textContent = "Copied!";
                setTimeout(() => button.textContent = "Copy citation", 2000);
            }};
        </script>
        """, height=40)

def play_header():
    logo_base64 = get_base64_image(LOGO_PATH)

    head1, head2 = st.columns([1,8])
    head1.markdown(
        f"""
        <style>
        .header-logo {{
            display: flex;
            align-items: center;
            justify-content: flex-start;
        }}
        .header-logo img {{
            max-width: 100%; /* Adjust the size as needed */
            overflow: hidden;
            object-fit: contain;
            padding-top: 12px;
       }}
        </style>
        <div class="header-logo">
            <img src="data:image/png;base64,{logo_base64}" alt="Logo">
        </div>
        """,
        unsafe_allow_html=True
    )
    with head2:
        """
        # interactive GEDI
        """
    """
    ## **G**enerating **E**vent **D**ata with **I**ntentional Features for Benchmarking Process Mining
    """
    st.markdown(
        "Real event logs rarely cover the characteristics you want to evaluate on. "
        "iGEDI generates event logs that hit **target feature values you choose** "
        "(e.g. variant entropy, trace length statistics), either a single log or a whole "
        "grid that spans the feature space evenly. Pick features and values below, press "
        "**Run Generation**, and download the `.xes` logs."
    )
    paper_col, cite_col = st.columns([6, 1], vertical_alignment="center")
    with paper_col:
        st.markdown(
            "iGEDI is based on [the ICPM'24 Demo Paper](https://ceur-ws.org/Vol-3783/paper_349.pdf). "
            "The code is available on [GitHub](https://github.com/andreamalhera/gedi). For larger runs, use: `pip install gedi`",
        )
    with cite_col:
        citation_button()
    st.button("Load example: 3×3 grid over variant entropy × most common variant ratio",
              on_click=load_example)
    return

def init_state(key, value):
    # Widgets get their defaults through session state, never through value=/default=.
    # That keeps the widget ID stable across reruns and lets load_example preset them.
    if key not in st.session_state:
        st.session_state[key] = value
    return {"key": key}

CONFIG_HELP = {
    "mode": "Most frequent number of visible activities in the process tree.",
    "sequence": "Probability of adding a sequence operator to the tree.",
    "choice": "Probability of adding a choice operator to the tree.",
    "parallel": "Probability of adding a parallel operator to the tree.",
    "loop": "Probability of adding a loop operator to the tree.",
    "silent": "Probability of adding a silent activity to a choice or loop operator.",
    "lt_dependency": "Probability of adding a random long-term dependency to the tree.",
    "num_traces": "Number of traces in the generated log.",
    "duplicate": "Probability of duplicating an activity label.",
    "or": "Probability of adding an OR operator to the tree.",
    "time_spread": "Exponent of the artificial timestamps: 1 = one second between events; "
                   "above 1 bunches events at the start of the log, below 1 at the end.",
}
INT_CONFIG_BOUNDS = {"mode": (1, 50), "num_traces": (1, 1000)}

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

FEATURE_GROUPS = [  # (group, name prefixes); first match wins, so start/end come before activities
    ("Size", ("n_traces", "n_variants", "n_unique_activities")),
    ("Trace length", ("trace_len_",)),
    ("Variants", ("ratio_", "mean_variant_occurrence", "std_variant_occurrence",
                  "skewness_variant_occurrence", "kurtosis_variant_occurrence")),
    ("Start activities", ("n_unique_start_activities", "start_activities_")),
    ("End activities", ("n_unique_end_activities", "end_activities_")),
    ("Activities", ("activities_",)),
    ("Entropy", ("eventropy_", "epa_")),
]

STAT_NAMES = {
    "min": "minimum", "max": "maximum", "mean": "mean", "median": "median", "mode": "mode",
    "std": "standard deviation", "variance": "variance", "q1": "first quartile", "q3": "third quartile",
    "iqr": "interquartile range", "geometric_mean": "geometric mean", "geometric_std": "geometric standard deviation",
    "harmonic_mean": "harmonic mean", "skewness": "skewness", "kurtosis": "kurtosis",
    "coefficient_variation": "coefficient of variation", "entropy": "entropy",
    "skewness_hist": "skewness of the histogram", "kurtosis_hist": "kurtosis of the histogram",
}

FEATURE_DESCRIPTIONS = {
    "n_traces": "Number of traces (cases) in the log.",
    "n_variants": "Number of distinct trace variants.",
    "n_unique_activities": "Number of distinct activity labels.",
    "n_unique_start_activities": "Number of distinct activities that start a trace.",
    "n_unique_end_activities": "Number of distinct activities that end a trace.",
    "ratio_variants_per_number_of_traces": "Distinct variants divided by traces (1 = every trace is unique).",
    "ratio_most_common_variant": "Share of traces that follow the most frequent variant.",
    "mean_variant_occurrence": "Average number of traces per variant.",
    "std_variant_occurrence": "Standard deviation of the number of traces per variant.",
    "skewness_variant_occurrence": "Skewness of the number of traces per variant.",
    "kurtosis_variant_occurrence": "Kurtosis of the number of traces per variant.",
    "epa_variant_entropy": "Variant entropy of the log's prefix automaton (Back et al.).",
    "epa_normalized_variant_entropy": "Variant entropy (prefix automaton), normalized to 0–1: higher = more diverse behavior.",
    "epa_sequence_entropy": "Sequence entropy of the log's prefix automaton (Back et al.).",
    "epa_normalized_sequence_entropy": "Sequence entropy (prefix automaton), normalized to 0–1.",
    "epa_sequence_entropy_linear_forgetting": "Sequence entropy that discounts older behavior linearly.",
    "epa_normalized_sequence_entropy_linear_forgetting": "Normalized sequence entropy with linear forgetting (0–1).",
    "epa_sequence_entropy_exponential_forgetting": "Sequence entropy that discounts older behavior exponentially.",
    "epa_normalized_sequence_entropy_exponential_forgetting": "Normalized sequence entropy with exponential forgetting (0–1).",
}

def feature_group(feature):
    for group, prefixes in FEATURE_GROUPS:
        if feature.startswith(prefixes):
            return group
    return "Other"

def feature_description(feature):
    if feature in FEATURE_DESCRIPTIONS:
        return FEATURE_DESCRIPTIONS[feature]
    for prefix, subject in (("trace_len_", "the number of events per trace"),
                            ("start_activities_", "how often each start activity occurs"),
                            ("end_activities_", "how often each end activity occurs"),
                            ("activities_", "how often each activity occurs")):
        if feature.startswith(prefix):
            stat = feature[len(prefix):]
            if stat.startswith("hist"):
                return f"Share of traces in bin {stat[4:]} of the trace length histogram."
            return f"{STAT_NAMES.get(stat, stat).capitalize()} of {subject}."
    if feature.startswith("ratio_top_"):
        return f"Share of traces covered by the top {feature.split('_')[2]}% most frequent variants."
    if feature.startswith("eventropy_"):
        return "Entropy estimate of the log's behavior (eventropy, Back et al.)."
    return None

def range_hint(feature):
    low, high = feature_bounds(feature)
    if math.isinf(low) and math.isinf(high):
        return ""
    return f" ({low:g}–{high:g})" if not math.isinf(high) else f" (≥ {low:g})"

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

def split_list(input_list, n):
    # Calculate the size of each chunk
    k, m = divmod(len(input_list), n)
    # Use list comprehension to create n sublists
    return [input_list[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]

def default_values(feature, value, n):
    low, high = fit_window(feature, value, 0.2 * value * (n - 1))
    return [round(float(v), 2) for v in np.linspace(low, high, n)]

def default_range(feature, value):
    bound_low, bound_high = feature_bounds(feature)
    width = value if value > 0 else (bound_high - bound_low if not math.isinf(bound_high - bound_low) else 1.0)
    low, high = fit_window(feature, value, width)
    step = math.floor((high - low) / 3 * 100) / 100 or round((high - low) / 3, 4)
    return [round(low, 2), round(high, 2), step]

def to_float(value, label):
    try:
        return float(value)
    except (TypeError, ValueError):
        stop_with_error(f"'{value}' is not a number (field: {label}).")

def arange_values(selected_min, selected_max, step_value, label):
    selected_min, selected_max, step_value = (to_float(v, label) for v in (selected_min, selected_max, step_value))
    if step_value <= 0:
        stop_with_error(f"Step must be greater than 0 (field: {label}).")
    if selected_max < selected_min:
        stop_with_error(f"Max must be at least min (field: {label}).")
    return np.around(np.arange(selected_min, selected_max + step_value / 2, step_value), 2)

def create_objectives_grid(df, objectives, n_para_obj=2, method="combinatorial"):
        # One list of candidate values per objective; targets are their cartesian product.
        value_lists = []
        if "combinatorial" in method:
            # df: one row per feature, columns are the values (e.g. min/max or "value i").
            columns = sorted(objectives)
            objectives = df.index.to_list()
            for feature in objectives:
                value_lists.append(np.around([to_float(v, feature) for v in df.loc[feature, columns]], 2))

        elif method=="range-from-csv":
            for objective in objectives:
                min_col, max_col, step_col = st.columns(3)
                with min_col:
                    selcted_min = st.slider(objective+': min', min_value=float(df[objective].min()), max_value=float(df[objective].max()), value=df[objective].quantile(0.1), step=0.1, key=objective+"min")
                with max_col:
                    selcted_max = st.slider('max', min_value=selcted_min, max_value=float(df[objective].max()), value=df[objective].quantile(0.9), step=0.1, key=objective+"max")
                with step_col:
                    step_value = st.slider('step', min_value=float(df[objective].min()), max_value=float(df[objective].quantile(0.9)), value=df[objective].median()/(df[objective].min()+0.0001), step=0.01, key=objective+"step")
                value_lists.append(arange_values(selcted_min, selcted_max, step_value, objective))
        else :#method=="range-manual":
            if objectives != None:
                cross_labels =  [feature[0]+': '+feature[1] for feature in list(cproduct(objectives,['min', 'max', 'step']))]
                cross_values = [value for objective in objectives for value in default_range(objective, df[objective])]
                ranges = zip(objectives, split_list(list(input_multicolumn(cross_labels, cross_values, n_cols=3)), n_para_obj))
                for objective, range_value in ranges:
                    selcted_min, selcted_max, step_value = range_value
                    value_lists.append(arange_values(selcted_min, selcted_max, step_value, objective))

        if not objectives or not value_lists:
            stop_with_error("Please select at least one feature above.")
        cartesian_product = list(cproduct(*value_lists))
        return [{key: float(value[idx]) for idx, key in enumerate(objectives)} for value in cartesian_product]

@st.cache_data(show_spinner=False, max_entries=200)
def extract_xes_features(xes_bytes, features):
    with tempfile.NamedTemporaryFile(suffix=".xes", delete=False) as tmp:
        tmp.write(xes_bytes)
        path = tmp.name
    try:
        values = extract_features(path, list(features))
    finally:
        os.remove(path)
    return {key: float(value) for key, value in values.items() if key != 'log'}

def set_generator_targets(generator_params):
    def handle_csv_file(uploaded_file,grid_option):
        #TODO: This code is duplicated. Should be moved and removed.
        def column_mappings():
            column_names_short = {
            'rvpnot': 'ratio_variants_per_number_of_traces',
            'rmcv': 'ratio_most_common_variant',
            'tlcv': 'trace_len_coefficient_variation',
            'mvo': 'mean_variant_occurrence',
            'enve': 'epa_normalized_variant_entropy',
            'ense': 'epa_normalized_sequence_entropy',
            'eself': 'epa_sequence_entropy_linear_forgetting',
            'enself': 'epa_normalized_sequence_entropy_linear_forgetting',
            'eseef': 'epa_sequence_entropy_exponential_forgetting',
            'enseef': 'epa_normalized_sequence_entropy_exponential_forgetting'
            }

            return column_names_short
        # uploaded_file = st.file_uploader("Pick a csv-file containing feature values for features:", type="csv")
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            if len(df.columns) <= 1:
                raise pd.errors.ParserError("Please select a file withat least two columns (e.g. log, feature) and use ',' as a delimiter.")
            columns_to_rename = {col: column_mappings()[col] for col in df.columns if col in column_mappings()}

            # Rename the matching columns
            df.rename(columns=columns_to_rename, inplace=True)
            sel_features = st.multiselect("Selected features", list(df.columns), list(df.columns)[-1])
            if sel_features:
                df = df[sel_features]
                return df, sel_features
        return None, None

    def handle_combinatorial(sel_features, stats, tuple_values):
        triangular_option = choose("Grid over", "All selected features together",
                                   "Also every subset of them", key="choice_subsets",
                                   help="'Together' builds one grid over all selected features. "
                                        "'Every subset' also adds grids for each smaller combination "
                                        "of the features (e.g. A, B and A×B).")
        if triangular_option:
            targets = []
            elements = sel_features
            # List to store all combinations
            all_combinations = [combinations(sel_features, r) for r in range(1, len(sel_features) + 1)]
            all_combinations = [comb for sublist in all_combinations for comb in sublist]

            # Print or use the result as needed
            for comb in all_combinations:
                sel_stats = stats.loc[sorted(list(comb))]
                targets += create_objectives_grid(sel_stats, tuple_values, n_para_obj=len(tuple_values), method="combinatorial")
        else: # Square
            targets = create_objectives_grid(stats, tuple_values, n_para_obj=len(tuple_values), method="combinatorial")
        return targets

    def handle_csv_option(grid_option, df, sel_features):
        if grid_option:
            combinatorial = choose("Grid values", "Min–max–step range per feature",
                                   "Combine statistics from the file", key="choice_csv_grid",
                                   help="'Range' lets you pick min, max and step for each feature. "
                                        "'Statistics' builds the grid from the file's min, max and "
                                        "chosen quantiles.")
            if combinatorial:
                add_quantile = st.slider('Add %-quantile', min_value=0.0, max_value=100.0, value=50.0, step=5.0)
                stats = df.describe().transpose().sort_index()
                stats[f"{int(add_quantile)}%"] = df.quantile(q=add_quantile / 100)
                st.write(stats)
                tuple_values = st.multiselect("Tuples including", list(stats.columns)[3:], default=['min', 'max'])
                return handle_combinatorial(sel_features, stats, tuple_values)
            else:  # Range
                return create_objectives_grid(df, sel_features, n_para_obj=len(sel_features), method="range-from-csv")
        else:  # Point
            st.write(df)
            return df.to_dict(orient='records')

    def feature_select():
        all_features = list(generator_params['targets'].keys())
        groups = ["All"] + [g for g, _ in FEATURE_GROUPS if any(feature_group(f) == g for f in all_features)]
        group = st.selectbox("Browse features", groups, key="feature_group",
                             help="Narrows the list below; features you already selected stay selected.")
        init_state("chosen_features", [f for f in EXAMPLE_FEATURES if f in all_features] or [all_features[-1]])
        selected = st.session_state["chosen_features"]
        options = [f for f in all_features if group == "All" or feature_group(f) == group or f in selected]
        st.session_state["selected_features"] = selected

        def remember_selection():
            st.session_state["chosen_features"] = st.session_state["selected_features"]

        return st.multiselect("Selected features", options, key="selected_features",
                              on_change=remember_selection, format_func=lambda f: f + range_hint(f))

    def handle_manual_option(grid_option):
        if grid_option:
            combinatorial = choose("Grid values", "Min–max–step range per feature",
                                   "List of values per feature", key="choice_manual_grid", value=True,
                                   help="'List' lets you type each value; the grid combines them "
                                        "across features. 'Range' generates evenly spaced values.")
            if combinatorial:
                sel_features = feature_select()
                num_values = st.number_input('Values per feature', min_value=2, step=1,
                                             help="The grid combines every value of every selected feature.",
                                             **init_state("num_values", 2))

                values_indexes = ["value "+str(i+1) for i in range(num_values)]
                cross_labels =  [feature[0]+': '+feature[1] for feature in list(cproduct(sel_features,values_indexes))]
                cross_values = [value for feature in sel_features
                                for value in default_values(feature, generator_params['targets'][feature], num_values)]
                if not sel_features:
                    stop_with_error("Please select at least one feature above.")
                parameters = split_list(list(input_multicolumn(cross_labels, cross_values, n_cols=num_values)), len(sel_features))
                parameters = [[to_float(v, feature) for v in values] for feature, values in zip(sel_features, parameters)]

                tasks_df = pd.DataFrame(parameters, index=sel_features, columns=values_indexes)
                return handle_combinatorial(sel_features, tasks_df, values_indexes)

            else: # Range
                sel_features = feature_select()
                return create_objectives_grid(generator_params['targets'], sel_features, n_para_obj=len(sel_features), method="range-manual")

        else: # Point
            sel_features = feature_select()
            #sel_features = st.multiselect("Selected features", list(generator_params['targets'].keys()))

            target = {}
            for sel_feature in sel_features:
                low, high = feature_bounds(sel_feature)
                default = float(generator_params['targets'][sel_feature])
                target[sel_feature] = st.number_input(
                    sel_feature + range_hint(sel_feature),
                    min_value=None if math.isinf(low) else low, max_value=None if math.isinf(high) else high,
                    help=feature_description(sel_feature), format="%.4g",
                    **init_state(f"point_{sel_feature}", min(max(default, low), high)))
            return [target]
        return[]


    mode_col, source_col = st.columns(2)
    with mode_col:
        grid_option = choose("What do you want to generate?", "One log", "A grid of logs",
                             key="choice_grid",
                             help="A grid combines several values per feature, so the generated "
                                  "logs cover the feature space evenly.")
    with source_col:
        csv_option = choose("How do you set the targets?", "Type values", "Derive from my log or CSV",
                            key="choice_source",
                            help="Upload an .xes log to generate logs like it, or a CSV of "
                                 "feature values (one row per log).")

    if csv_option:
        uploaded_files = st.file_uploader(
            "Upload .xes event logs (one target per log) or a .csv of feature values (one row per target)",
            type=["csv", "xes"], accept_multiple_files=True)
        targets = []
        xes_files = [f for f in uploaded_files if f.name.endswith('.xes')]
        csv_files = [f for f in uploaded_files if f.name.endswith('.csv')]
        if xes_files and csv_files:
            st.warning("Using the .xes logs; remove them to use the .csv instead.")
        if xes_files:
            sel_features = feature_select()
            if sel_features:
                with st.spinner(f"Extracting features from {len(xes_files)} log(s)…"):
                    rows = {f.name: extract_xes_features(f.getvalue(), tuple(sel_features)) for f in xes_files}
                features_df = pd.DataFrame.from_dict(rows, orient="index")
                features_df.index.name = "log"
                st.dataframe(features_df)
                st.download_button("Download features as CSV", features_df.to_csv(), file_name="log_features.csv",
                                   mime="text/csv")
                targets = features_df.to_dict(orient="records")
            else:
                st.info("Select the features whose values your generated logs should match.")
        elif csv_files:
            df, sel_features = handle_csv_file(csv_files[0], grid_option)
            if df is not None and sel_features is not None:
                targets = handle_csv_option(grid_option, df, sel_features)
    else:  # Manual
        targets = handle_manual_option(grid_option)

    generator_params['targets'] = targets
    st.write(f"...result in {len(generator_params['targets'])} target(s)")

    """
    #### Generator settings
    """
    with st.expander("Advanced: process tree generator settings"):
        st.caption("GEDI searches within these ranges for process trees whose logs hit your targets. "
                   "Wider ranges give it more room; the defaults work for most targets. "
                   "[Parameter details](https://github.com/tjouck/PTandLogGenerator)")
        config_space_inputs(generator_params['config_space'])
        generator_params['n_trials'] = int(st.number_input(
            "Optimization trials per target", min_value=1, max_value=MAX_TRIALS if HOSTED else None, step=1,
            help="More trials usually get closer to the target, but take longer.",
            **init_state("n_trials", int(generator_params['n_trials']))))

    if HOSTED:
        n_targets = len(generator_params['targets'])
        st.caption(f"This run: {n_targets} target(s) × {generator_params['n_trials']} optimization trials. "
                   f"The hosted demo allows up to {MAX_TARGETS} targets and {MAX_TRIALS} trials; "
                   "use `pip install gedi` for larger benchmarks.")
        if n_targets > MAX_TARGETS:
            stop_with_error(f"{n_targets} targets exceed the hosted limit of {MAX_TARGETS}. Use fewer values or features.")
        if generator_params['n_trials'] > MAX_TRIALS:
            stop_with_error(f"n_trials exceeds the hosted limit of {MAX_TRIALS}.")

    # For features' possible ranges and the configuration space.
    target_errors, target_warnings = {}, {}
    for target in generator_params['targets']:
        errors, warnings = validate_target(target, generator_params['config_space'])
        target_errors.update(dict.fromkeys(errors))
        target_warnings.update(dict.fromkeys(warnings))
    for error in target_errors:
        st.error(f"{error} Targets with this value will be skipped.")
    for warning in target_warnings:
        st.warning(warning)

    return generator_params

def sort_key(val):
    parts = val.split('_')
    # Extract the numeric parts after the 'genEL' prefix as floats, e.g. from 'genEL_0p39'
    # get (0.39,), or from 'genEL_m0p47_nan' (a multi-objective target missing one dimension)
    # get (-0.47, nan). 'p' and 'm' encode the decimal point and minus sign in file names.
    return tuple(float(part.replace('p', '.').replace('m', '-')) for part in parts[1:])

def match_logs_to_targets(targets):
    log_targets = {}
    for _, task in pd.DataFrame(targets).iterrows():
        identifiers = [x for x in task if isinstance(x, str)]
        identifier = "genEL" + (identifiers[0] if identifiers else "")
        objectives = dict(sorted(task.drop('log', errors='ignore').to_dict().items()))
        log_targets[get_generated_file_name(objectives, identifier)] = objectives
    return log_targets

def plot_feasibility_heatmap(dataframes, targets):
    log_targets = match_logs_to_targets(targets)
    features = sorted({k for target in log_targets.values() for k, v in target.items() if is_numeric(v)})
    columns = features + ['target_similarity']

    similarities = pd.DataFrame(np.nan, index=dataframes.index, columns=columns)
    for log, generated in dataframes.iterrows():
        target = log_targets.get(log, {})
        for feature, similarity in compute_feature_similarities(target, generated.to_dict()).items():
            similarities.loc[log, feature] = similarity
    similarities['target_similarity'] = dataframes['target_similarity']

    n_logs, n_cols = similarities.shape
    cmap = plt.get_cmap('RdYlGn').copy()
    cmap.set_bad('lightgray')
    label_width = 0.07 * max(len(log) for log in similarities.index)
    label_height = 0.05 * max(len(feature) for feature in columns)
    fig, ax = plt.subplots(figsize=(2 + label_width + 0.8 * n_cols, 1.5 + label_height + 0.4 * n_logs))
    image = ax.imshow(np.ma.masked_invalid(similarities.to_numpy(dtype=float)),
                      cmap=cmap, vmin=0, vmax=1, aspect='auto')

    ax.axvline(n_cols - 1.5, color='white', linewidth=4)

    if n_logs * n_cols <= 300:
        for (row, col), value in np.ndenumerate(similarities.to_numpy(dtype=float)):
            if not np.isnan(value):
                ax.text(col, row, f"{value:.2f}", ha='center', va='center', fontsize=7,
                        color='white' if value < 0.2 or value > 0.8 else 'black')

    ax.set_xticks(range(n_cols), features + ['overall'], rotation=45, ha='right')
    ax.set_yticks(range(n_logs), similarities.index)
    ax.set_xlabel('Target feature')
    ax.set_ylabel('Log')
    fig.colorbar(image, ax=ax, label='Feasibility (similarity to target)')
    fig.tight_layout()
    return fig

SIMILARITY_GOAL = 0.9

STEP_TITLES = {
    "event_logs_generation": "Generate event logs",
    "feature_extraction": "Extract features",
}

def start_generation(output_path, config_file, step_config):
    """Starts GEDI in a subprocess and records the job in session state."""
    with open(output_path, "w") as f:
        f.write(config_file)

    path_to_logs = step_config['output_path']
    if os.path.exists(path_to_logs):
        shutil.rmtree(path_to_logs)

    log_path = os.path.join(os.path.dirname(output_path) or ".", "generation_output.txt")
    command = [sys.executable, "-W", "ignore", "main.py", "-a", output_path]
    with open(log_path, "w") as process_log:  # the child keeps its own handle
        process = subprocess.Popen(command, stdout=process_log, stderr=subprocess.STDOUT, text=True)

    st.session_state["generation_job"] = {
        "process": process,
        "path_to_logs": path_to_logs,
        "log_path": log_path,
        "config_file": config_file,
        "targets": step_config['generator_params']['targets'],
        "started": time.strftime("%Y-%m-%d %H:%M"),
    }

def feature_files(path_to_logs):
    return [os.path.join(root, file)
            for root, _, files in os.walk(path_to_logs)
            for file in files
            if os.path.basename(root) == 'features']

def wait_for_generation(job):
    """Shows progress until the job's process ends; one feature file appears per finished log."""
    process = job["process"]
    if process.poll() is not None:
        return
    n_targets = max(len(job["targets"]), 1)
    progress = st.progress(0.0, text="Starting generation…")
    while process.poll() is None:
        n_done = len(feature_files(job["path_to_logs"]))
        progress.progress(min(n_done / n_targets, 1.0), text=f"Generating logs… {n_done}/{n_targets} done")
        time.sleep(1)
    progress.empty()

def results_zip(job):
    readme = (
        "Event logs generated with iGEDI\n"
        f"Generated: {job['started']}\n"
        f"Targets: {len(job['targets'])} (see target_config.json, which reproduces this run with "
        "`python main.py -a target_config.json`)\n\n"
        "Folder layout: *.xes are the generated logs; features/*.json hold each log's features "
        "and its similarity to the target.\n\n"
        f"How to cite:\n{IGEDI_CITATION}\nhttps://ceur-ws.org/Vol-3783/paper_349.pdf\n"
    )
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for root, _, files in os.walk(job["path_to_logs"]):
            for file in files:
                path = os.path.join(root, file)
                prefix = "features/" if os.path.basename(root) == "features" else ""
                zip_file.write(path, prefix + file)
        zip_file.writestr("target_config.json", job["config_file"])
        zip_file.writestr("README.txt", readme)
    buffer.seek(0)
    return buffer

def show_results(job):
    if job["process"].returncode != 0:
        st.error("Generation failed:")
        with open(job["log_path"]) as process_log:
            st.code(process_log.read()[-5000:], language='text')
        return

    file_paths = feature_files(job["path_to_logs"])
    if not file_paths:
        st.warning("Generation finished, but no logs were produced. Check that your targets are reachable.")
        return

    st.write("## Results")
    dataframes = pd.concat([pd.read_json(file, lines=True) for file in file_paths], ignore_index=True)
    # Reorder columns with 'target_similarity' as the last column
    columns = [col for col in dataframes.columns if col != 'target_similarity'] + ['target_similarity']
    dataframes = dataframes[columns]
    dataframes = dataframes.sort_values(by='log', key=lambda col: col.map(sort_key))
    dataframes['log'] = dataframes['log'].astype(str)
    dataframes.set_index('log', inplace=True)

    n_logs = len(dataframes)
    n_close = int((dataframes['target_similarity'] >= SIMILARITY_GOAL).sum())
    st.success(f"Generated {n_logs} log(s) from your run started {job['started']}. "
               f"{n_close} of {n_logs} reached at least {SIMILARITY_GOAL} similarity to their target.")
    st.download_button(label="Download generated logs (.zip)", data=results_zip(job),
                       file_name='generated_logs.zip', mime='application/zip',
                       help="Includes the .xes logs, their features, the config to reproduce the run, "
                            "and how to cite iGEDI.")

    col1, col2 = st.columns([2, 3])
    with col1:
        st.dataframe(dataframes)
    with col2:
        st.caption("Each row is one generated log; each cell shows how close that feature came to its "
                   "target (1 = exact, green is good). The last column, *overall*, is the log's "
                   "similarity to its whole target.")
        fig = plot_feasibility_heatmap(dataframes, job['targets'])
        st.pyplot(fig, dpi=400)
        plt.close(fig)

if __name__ == '__main__':
    play_header()

    # Load the configuration layout from a JSON file
    config_layout = json.load(open("config_files/config_layout.json"))

    # Define available pipeline steps
    step_candidates = ["event_logs_generation", "feature_extraction"]

    # Streamlit multi-select for pipeline steps
    if HOSTED:
        pipeline_steps = ["event_logs_generation"]
    else:
        pipeline_steps = st.multiselect(
            "Choose pipeline step",
            step_candidates,
            format_func=lambda step: STEP_TITLES.get(step, step),
            **init_state("pipeline_steps", ["event_logs_generation"])
        )

    step_configs = []
    set_col, view_col = st.columns([3, 2])

    # Iterate through selected pipeline steps
    for pipeline_step in pipeline_steps:
        step_config = next(d for d in config_layout if d['pipeline_step'] == pipeline_step)

        with set_col:
            st.header(STEP_TITLES.get(pipeline_step, pipeline_step))

            # Iterate through step configuration keys
            for step_key in step_config.keys():
                if step_key == "generator_params":
                    st.subheader("Choose your targets")
                    step_config[step_key] = set_generator_targets(step_config[step_key])
                elif step_key == "feature_params":
                    layout_features = list(step_config[step_key]['feature_set'])
                    step_config[step_key]["feature_set"] = st.multiselect(
                        "features to extract",
                        layout_features
                    )
                elif HOSTED and pipeline_step == "event_logs_generation" and step_key == "output_path":
                    # Users must not choose server paths: the run deletes this folder first.
                    step_config[step_key] = os.path.join(session_workdir(), "generated")
                elif step_key != "pipeline_step":
                    step_config[step_key] = st.text_input(step_key, step_config[step_key])

        with view_col:
            with st.expander("Show configuration (JSON)"):
                st.json(step_config)

        step_configs.append(step_config)

    # Convert step configurations to JSON
    # The UI groups targets/config_space/n_trials under "generator_params" for editing, but
    # gedi/run.py reads them as flat top-level keys (targets, config_space, system_params) on
    # each step. Flatten into the shape the pipeline actually consumes before saving/running.
    run_configs = []
    for step_config in step_configs:
        run_config = dict(step_config)
        if run_config.get("pipeline_step") == "event_logs_generation" and "generator_params" in run_config:
            generator_params = run_config.pop("generator_params")
            run_config["targets"] = generator_params["targets"]
            run_config["config_space"] = generator_params["config_space"]
            run_config["system_params"] = {
                "output_path": run_config.get("output_path"),
                "n_trials": generator_params["n_trials"],
            }
        run_configs.append(run_config)
    config_file = json.dumps(run_configs, indent=4)

    # Streamlit input for output file path
    if HOSTED:
        output_path = os.path.join(session_workdir(), "target_config.json")
    else:
        output_path = st.text_input("Output file path", "config_files/target_config.json")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Streamlit multi-button for saving options

    button_col1, button_col2 = st.columns([1, 1])
    with button_col1:
        create_button = st.download_button(label="Download config file", data=config_file, file_name=os.path.basename(output_path), mime='application/json')
        if pipeline_steps != ["event_logs_generation"]:
            st.write("Run command:")
            st.code(f"python -W ignore main.py -a {output_path}", language='bash')
    if pipeline_steps == ["event_logs_generation"]:
        with button_col2:
            create_run_button = st.button("Run Generation")

        if create_run_button:
            start_generation(output_path, config_file, step_config)

        job = st.session_state.get("generation_job")
        if job is not None:
            wait_for_generation(job)
            show_results(job)