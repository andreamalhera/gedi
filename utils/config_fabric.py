from itertools import product as cproduct
from itertools import combinations
from pylab import *
import ast
import base64
import tempfile
import json
import math
import os
import pandas as pd
import streamlit as st
import subprocess
import time
import shutil
import zipfile
import io
import sys

# In some sandboxed containers (e.g. the Hugging Face Space runtime), os.getppid()
# returns a PID with no corresponding process. pm4py's import-time constants.py
# calls psutil.Process(os.getppid()) unguarded, which raises NoSuchProcess there.
# Substitute a PID that's always valid (our own) for the duration of this import only.
_real_getppid = os.getppid
os.getppid = os.getpid
from feeed.feature_extractor import extract_features

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gedi.utils.io_helpers import compute_feature_similarities, get_generated_file_name
from gedi.utils.target_validation import fit_window, is_numeric, validate_target
os.getppid = _real_getppid

st.set_page_config(layout='wide')
INPUT_XES="output/inputlog_temp.xes"
LOGO_PATH="gedi/utils/logo.png"

# Hugging Face sets SPACE_ID in the Space runtime. When hosted, users share one machine, so
# output paths are per-session temp dirs (not editable) and job size is capped.
HOSTED = bool(os.environ.get("SPACE_ID"))
MAX_TARGETS = int(os.environ.get("IGEDI_MAX_TARGETS", 30))
MAX_TRIALS = int(os.environ.get("IGEDI_MAX_TRIALS", 100))

IGEDI_CITATION = ("How to cite: A. Maldonado, S. A. Aryasomayajula, C. M. M. Frey, T. Seidl. "
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
    st.session_state["toggle_switch_Point-"] = True   # grid-based
    st.session_state["toggle_switch_Manual"] = False  # manual input
    st.session_state["toggle_switch_Range"] = True    # combinatorial
    st.session_state["toggle_switch_Square"] = False  # square grid
    st.session_state["num_values"] = 3
    st.session_state["selected_features"] = EXAMPLE_FEATURES
    for feature, values in EXAMPLE_VALUES.items():
        for i, value in enumerate(values):
            st.session_state[f"input_{feature}: value {i+1}"] = str(value)
    st.session_state["n_trials"] = "20"

def stop_with_error(message):
    st.error(message)
    st.stop()

def get_base64_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()

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
        "**Run Generation**, and download the `.xes` logs.  \n"
        "iGEDI is based on [the ICPM'24 Demo Paper](https://ceur-ws.org/Vol-3783/paper_349.pdf) "
        f"<span title=\"{IGEDI_CITATION}\" style=\"cursor: help; font-size: 0.85em; "
        "text-decoration: underline dotted;\">cite</span> "
        "The code is available on [GitHub](https://github.com/andreamalhera/gedi). For larger runs, use: `pip install gedi`",
        unsafe_allow_html=True,
    )
    st.button("Load example: 3×3 grid over variant entropy × most common variant ratio",
              on_click=load_example)
    return

def init_state(key, value):
    # Widgets get their defaults through session state, never through value=/default=.
    # That keeps the widget ID stable across reruns and lets load_example preset them.
    if key not in st.session_state:
        st.session_state[key] = value
    return {"key": key}

def double_switch(label_left, label_right, third_label=None, fourth_label=None, value=False):
    if third_label==None and fourth_label==None:
        # Create two columns for the labels and toggle switch
        col0, col1, col2, col3, col4 = st.columns([2,1,1,1,2])
    else:
        # Create two columns for the labels and toggle switch
        col0, col1, col2, col3, col4, col5, col6, col7, col8 = st.columns([1,1,1,1,1,1,1,1,1])

    # Add labels to the columns
    with col1:
        st.write(label_left)

    with col2:
        # Create the toggle switch
        key = "toggle_switch_"+label_left
        toggle_option = st.toggle(" ", **init_state(key, value))

    with col3:
        st.write(label_right)
    if third_label is None and fourth_label is None:return toggle_option
    else:
        with col5:
            st.write(third_label)

        with col6:
            # Create the toggle switch
            key = "toggle_switch_"+third_label
            toggle_option_2 = st.toggle(" ", **init_state(key, False))

        with col7:
            st.write(fourth_label)
        return toggle_option, toggle_option_2

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
            result[label] = col.text_input(label, **init_state(key, str(default_value)))
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
    low, high = fit_window(feature, value, value)
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
        triangular_option = double_switch("Square", "Triangular")
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
            combinatorial = double_switch("Range", "Combinatorial")
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
        options = list(generator_params['targets'].keys())
        return st.multiselect("Selected features", options, **init_state("selected_features", [options[-1]]))

    def handle_manual_option(grid_option):
        if grid_option:
            combinatorial = double_switch("Range", "Combinatorial", value=True)
            if combinatorial:
                col1, col2 = st.columns([1,4])
                with col1:
                    num_values = st.number_input('How many values to define?', min_value=2, step=1,
                                                 **init_state("num_values", 2))
                with col2:
                    sel_features = feature_select()

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

            target = {sel_feature: to_float(st.text_input(sel_feature, generator_params['targets'][sel_feature]), sel_feature)
                      for sel_feature in sel_features}
            return [target]
        return[]


    grid_option, csv_option = double_switch("Point-", "Grid-based", third_label="Manual", fourth_label="From File")

    if csv_option:
        uploaded_file = st.file_uploader("Pick a csv-file containing feature values for features (or) an xes-event log:", type=["csv","xes"])
        targets = []
        if uploaded_file is not None:
            if uploaded_file.name.endswith('.xes'):
                # Save inside this session's dir under a fixed name, so concurrent users
                # (or a malicious file name) can't clash with other files.
                xes_path = os.path.join(session_workdir(), "uploaded_log.xes")
                with open(xes_path, 'wb') as f:
                    f.write(uploaded_file.getbuffer())

                sel_features = st.multiselect("Selected features", list(generator_params['targets'].keys()))
                xes_features = extract_features(xes_path, sel_features)
                del xes_features['log']
                # removing the temporary file
                uploaded_file.close()
                if os.path.exists(xes_path):
                    os.remove(xes_path)
                xes_features = {key: float(value) for key, value in xes_features.items()}
                targets = [xes_features]

            if uploaded_file.name.endswith('.csv'):
                df, sel_features = handle_csv_file(uploaded_file,grid_option)
                if df is not None and sel_features is not None:
                    targets = handle_csv_option(grid_option, df, sel_features)
                else:
                    targets = []
    else:  # Manual
        targets = handle_manual_option(grid_option)

    generator_params['targets'] = targets
    st.write(f"...result in {len(generator_params['targets'])} target(s)")

    """
    #### Configuration space
    """
    updated_values = input_multicolumn(generator_params['config_space'].keys(), generator_params['config_space'].values())
    for key, new_value in zip(generator_params['config_space'].keys(), updated_values):
        # literal_eval only accepts Python literals such as [0.01, 1], never code.
        try:
            parsed = ast.literal_eval(new_value)
        except (ValueError, SyntaxError):
            stop_with_error(f"'{new_value}' is not a valid value for {key}. Use a list such as [0.01, 1].")
        if not isinstance(parsed, (list, tuple)) or not all(is_numeric(v) for v in parsed):
            stop_with_error(f"{key} must be a list of numbers, e.g. [0.01, 1].")
        generator_params['config_space'][key] = list(parsed)
    n_trials = st.text_input('n_trials', **init_state("n_trials", str(generator_params['n_trials'])))
    try:
        generator_params['n_trials'] = int(n_trials)
    except ValueError:
        stop_with_error(f"n_trials must be a whole number, got '{n_trials}'.")
    if generator_params['n_trials'] < 1:
        stop_with_error("n_trials must be at least 1.")

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

if __name__ == '__main__':
    play_header()

    # Load the configuration layout from a JSON file
    config_layout = json.load(open("config_files/config_layout.json"))

    # Define available pipeline steps
    step_candidates = ["event_logs_generation", "feature_extraction"]

    # Streamlit multi-select for pipeline steps
    pipeline_steps = st.multiselect(
        "Choose pipeline step",
        step_candidates,
        **init_state("pipeline_steps", ["event_logs_generation"])
    )

    step_configs = []
    set_col, view_col = st.columns([3, 2])

    # Iterate through selected pipeline steps
    for pipeline_step in pipeline_steps:
        step_config = next(d for d in config_layout if d['pipeline_step'] == pipeline_step)

        with set_col:
            st.header(pipeline_step)

            # Iterate through step configuration keys
            for step_key in step_config.keys():
                if step_key == "generator_params":
                    st.subheader("Set-up targets")
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
            st.write(step_config)

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
            # Save configuration to the specified output path
            with open(output_path, "w") as f:
                f.write(config_file)

            # Use the exact interpreter running this app rather than relying on a bare
            # "python" being resolvable on PATH in the subprocess's environment.
            command = [sys.executable, "-W", "ignore", "main.py", "-a", output_path]

            # The generator writes both logs and feature JSONs under <output_path>/<task>/,
            # with feature JSONs nested one level deeper in a "features" subfolder.
            path_to_logs = step_config['output_path'] # for log storage

            # Clean existing output path if it exists
            if os.path.exists(path_to_logs):
                shutil.rmtree(path_to_logs)

            # Run the actual command, showing progress as one feature file appears per finished log.
            n_targets = max(len(step_config['generator_params']['targets']), 1)
            progress = st.progress(0.0, text="Starting generation…")
            log_path = os.path.join(os.path.dirname(output_path) or ".", "generation_output.txt")
            with open(log_path, "w") as process_log:
                process = subprocess.Popen(command, stdout=process_log, stderr=subprocess.STDOUT, text=True)
                while process.poll() is None:
                    n_done = sum(len(files) for root, _, files in os.walk(path_to_logs)
                                 if os.path.basename(root) == 'features')
                    progress.progress(min(n_done / n_targets, 1.0),
                                      text=f"Generating logs… {n_done}/{n_targets} done")
                    time.sleep(1)
            progress.empty()

            if process.returncode != 0:
                st.error("Generation failed:")
                with open(log_path) as process_log:
                    st.code(process_log.read()[-5000:], language='text')
                st.stop()

            st.success("Logs generated!")
            st.write("## Results")

            # Collect all feature JSON file paths from the output directory
            file_paths = [os.path.join(root, file)
                            for root, _, files in os.walk(path_to_logs)
                            for file in files
                            if os.path.basename(root) == 'features']

            # Download the generated logs as a ZIP file
            download_file_paths = [os.path.join(root, file)
                            for root, _, files in os.walk(path_to_logs)
                            for file in files]

            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
                for file in download_file_paths:
                    zip_file.write(file, os.path.basename(file))
            zip_buffer.seek(0)
            st.download_button(label="Download generated logs", data=zip_buffer, file_name='generated_logs.zip', mime='application/zip')

            # Read and concatenate all JSON files into a DataFrame
            dataframes = pd.concat([pd.read_json(file, lines=True) for file in file_paths], ignore_index=True)

            # Reorder columns with 'target_similarity' as the last column
            columns = [col for col in dataframes.columns if col != 'target_similarity'] + ['target_similarity']
            dataframes = dataframes[columns]
            dataframes = dataframes.sort_values(by='log', key=lambda col: col.map(sort_key))

            # Set 'log' as the index
            dataframes['log'] = dataframes['log'].astype(str)
            dataframes.set_index('log', inplace=True)

            col1, col2 = st.columns([2, 3])

            with col1:
                st.dataframe(dataframes)

            with col2:
                st.pyplot(plot_feasibility_heatmap(dataframes, step_config['generator_params']['targets']), dpi=400)