from itertools import product as cproduct
from itertools import combinations
from pathlib import Path
from pylab import *
import base64
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
from feeed.feature_extractor import extract_features

st.set_page_config(layout='wide')
INPUT_XES="output/inputlog_temp.xes"
LOGO_PATH="gedi/utils/logo.png"

def get_base64_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()

def play_header():
    # Convert local image to base64
    logo_base64 = get_base64_image(LOGO_PATH)

    # HTML and CSS for placing the logo at the top left corner
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
    return

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
        toggle_option = st.toggle(" ",value=value,
            key="toggle_switch_"+label_left,
        )

    with col3:
        st.write(label_right)
    if third_label is None and fourth_label is None:return toggle_option
    else:
        with col5:
            st.write(third_label)

        with col6:
            # Create the toggle switch
            toggle_option_2 = st.toggle(" ",value=False,
                key="toggle_switch_"+third_label,
            )

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
            result[label] = col.text_input(label, default_value, key=f"input_"+label+'_'+str(default_value))
    return result.values()

def split_list(input_list, n):
    # Calculate the size of each chunk
    k, m = divmod(len(input_list), n)
    # Use list comprehension to create n sublists
    return [input_list[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]

def get_ranges_from_stats(stats, tuple_values):
    col_for_row = ", ".join([f"x[\'{i}\'].astype(float)" for i in tuple_values])
    stats['range'] = stats.apply(lambda x: tuple([eval(col_for_row)]), axis=1)
    #tasks = eval(f"list(itertools.product({(parameters*n_para_obj)[:-2]}))")
    result = [f"np.around({x}, 2)" for x in stats['range']]
    result = ", ".join(result)
    return result

def create_objectives_grid(df, objectives, n_para_obj=2, method="combinatorial"):
        if "combinatorial" in method:
            sel_features = df.index.to_list()
            parameters_o = "objectives, "
            parameters = get_ranges_from_stats(df, sorted(objectives))
            objectives = sorted(sel_features)
            tasks = f"list(cproduct({parameters}))[0]"

        elif method=="range-from-csv":
            tasks = ""
            for objective in objectives:
                min_col, max_col, step_col = st.columns(3)
                with min_col:
                    selcted_min = st.slider(objective+': min', min_value=float(df[objective].min()), max_value=float(df[objective].max()), value=df[objective].quantile(0.1), step=0.1, key=objective+"min")
                with max_col:
                    selcted_max = st.slider('max', min_value=selcted_min, max_value=float(df[objective].max()), value=df[objective].quantile(0.9), step=0.1, key=objective+"max")
                with step_col:
                    step_value = st.slider('step', min_value=float(df[objective].min()), max_value=float(df[objective].quantile(0.9)), value=df[objective].median()/(df[objective].min()+0.0001), step=0.01, key=objective+"step")
                tasks += f"np.around(np.arange({selcted_min}, {selcted_max}+{step_value}, {step_value}),2), "
        else :#method=="range-manual":
            experitments = []
            tasks=""
            if objectives != None:
                cross_labels =  [feature[0]+': '+feature[1] for feature in list(cproduct(objectives,['min', 'max', 'step']))]
                cross_values = [round(eval(str(combination[0])+combination[1]), 2) for combination in list(cproduct(list(df.values()), ['*1', '*2', '/3']))]
                ranges = zip(objectives, split_list(list(input_multicolumn(cross_labels, cross_values, n_cols=3)), n_para_obj))
                for objective, range_value in ranges:
                    selcted_min, selcted_max, step_value = range_value
                    tasks += f"np.around(np.arange({selcted_min}, {selcted_max}+{step_value}, {step_value}),2), "

        try:
            cartesian_product = list(cproduct(*eval(tasks)))
            targets = [{key: value[idx] for idx, key in enumerate(objectives)} for value in cartesian_product]
            return targets
        except SyntaxError as e:
            st.write("Please select valid features above.")
            sys.exit(1)
        except TypeError as e:
            st.write("Please select at least 2 values to define.")
            sys.exit(1)

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
        return st.multiselect("Selected features", list(generator_params['target'].keys()),
                                                        list(generator_params['target'].keys())[-1])

    def handle_manual_option(grid_option):
        if grid_option:
            combinatorial = double_switch("Range", "Combinatorial", value=True)
            if combinatorial:
                col1, col2 = st.columns([1,4])
                with col1:
                    num_values = st.number_input('How many values to define?', min_value=2, step=1)
                with col2:
                    sel_features = feature_select()

                filtered_dict = {key: generator_params['target'][key] for key in sel_features if key in generator_params['target']}
                values_indexes = ["value "+str(i+1) for i in range(num_values)]
                values_defaults = ['*(1+2*0.'+str(i)+')' for i in range(num_values)]
                cross_labels =  [feature[0]+': '+feature[1] for feature in list(cproduct(sel_features,values_indexes))]
                cross_values = [round(eval(str(combination[0])+combination[1]), 2) for combination in list(cproduct(list(filtered_dict.values()), values_defaults))]
                parameters = split_list(list(input_multicolumn(cross_labels, cross_values, n_cols=num_values)), len(sel_features))
                tasks = f"list({parameters})"

                tasks_df = pd.DataFrame(eval(tasks), index=sel_features, columns=values_indexes)
                tasks_df = tasks_df.astype(float)
                return handle_combinatorial(sel_features, tasks_df, values_indexes)

            else: # Range
                sel_features = feature_select()
                return create_objectives_grid(generator_params['target'], sel_features, n_para_obj=len(sel_features), method="range-manual")

        else: # Point
            sel_features = feature_select()
            #sel_features = st.multiselect("Selected features", list(generator_params['target'].keys()))

            target = {sel_feature: float(st.text_input(sel_feature, generator_params['target'][sel_feature])) for sel_feature in sel_features}
            return [target]
        return[]


    grid_option, csv_option = double_switch("Point-", "Grid-based", third_label="Manual", fourth_label="From File")

    if csv_option:
        uploaded_file = st.file_uploader("Pick a csv-file containing feature values for features (or) an xes-event log:", type=["csv","xes"])
        targets = []
        if uploaded_file is not None:
            if uploaded_file.name.endswith('.xes'):
                with open(f"{uploaded_file.name}", 'wb') as f:
                    f.write(uploaded_file.getbuffer())

                sel_features = st.multiselect("Selected features", list(generator_params['target'].keys()))
                xes_features = extract_features(f"{uploaded_file.name}", sel_features)
                del xes_features['log']
                # removing the temporary file
                uploaded_file.close()
                if os.path.exists(f"{uploaded_file.name}"):
                    os.remove(f"{uploaded_file.name}")
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

    generator_params['target'] = targets
    st.write(f"...result in {len(generator_params['target'])} target(s)")

    """
    #### Configuration space
    """
    updated_values = input_multicolumn(generator_params['config_space'].keys(), generator_params['config_space'].values())
    for key, new_value in zip(generator_params['config_space'].keys(), updated_values):
        generator_params['config_space'][key] = eval(new_value)
    generator_params['n_trials'] = int(st.text_input('n_trials', generator_params['n_trials']))

    return generator_params

def sort_key(val):
    parts = val.split('_')
    # Extract and convert the numeric parts
    part1 = int(parts[0][5:])  # e.g., from 'genEL1', extract '1'
    return (part1)

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
        ["event_logs_generation"]
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
                elif step_key != "pipeline_step":
                    step_config[step_key] = st.text_input(step_key, step_config[step_key])

        with view_col:
            st.write(step_config)

        step_configs.append(step_config)

    # Convert step configurations to JSON
    config_file = json.dumps(step_configs, indent=4)

    # Streamlit input for output file path
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

            command = f"python -W ignore main.py -a {output_path}".split()

            # Prepare output path for feature extraction
            directory = Path(step_config['output_path']).parts
            path = os.path.join(directory[0], 'features', *directory[1:]) # for feature storage
            path_to_logs = os.path.join(*directory[:]) # for log storage

            # Clean existing output path if it exists
            if os.path.exists(path):
                shutil.rmtree(path)

            if os.path.exists(path_to_logs):
                shutil.rmtree(path_to_logs)

            # Simulate running the command with a loop and update progress
            with st.spinner("Generating logs.."):
            # Run the actual command
                result = subprocess.run(command, capture_output=True, text=True)
            st.success("Logs generated!")
            st.write("## Results")

            # Collect all file paths from the output directory
            file_paths = [os.path.join(root, file)
                            for root, _, files in os.walk(path)
                            for file in files]

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
            xticks_labels=dataframes['log'].apply(lambda x: x.split('_')[0])#+'_'+x.split('_')[1][:4]+'_'+x.split('_')[2][:4])
            dataframes.set_index('log', inplace=True)

            col1, col2 = st.columns([2, 3])

            with col1:
                st.dataframe(dataframes)

            with col2:
                plt.figure(figsize=(6, 3))
                plt.plot(xticks_labels, dataframes['target_similarity'], 'o-')
                plt.xlabel('Log')
                plt.ylabel('Target Similarity')
                if len(dataframes) > 10:
                    plt.xticks(rotation=30, ha='right')
                else:
                    plt.xticks(rotation=0, ha='center')
                plt.tight_layout()
                st.pyplot(plt, dpi=400)