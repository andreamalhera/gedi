from itertools import product as cproduct
from itertools import combinations
import math
import os
import tempfile
import pandas as pd
import streamlit as st

from igedi.app import extract_features, feature_bounds, validate_target
from igedi.app.config import EXAMPLE_FEATURES, HOSTED, MAX_TARGETS, MAX_TRIALS
from igedi.app.features import FEATURE_GROUPS, feature_description, feature_group, range_hint
from igedi.app.grid import create_objectives_grid, default_values, split_list, to_float
from igedi.app.state import init_state, stop_with_error
from igedi.app.widgets import choose, config_space_inputs, input_multicolumn

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
                                   mime="text/csv", on_click="ignore")
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

    st.markdown("""
    #### Generator settings
    """)
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
                   "for larger runs, prepare your config file here and run it locally with `pip install gedi`.")
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
