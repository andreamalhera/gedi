import io
import os
import zipfile
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from igedi.app import compute_feature_similarities, get_generated_file_name, is_numeric
from igedi.app.config import IGEDI_CITATION, SIMILARITY_GOAL
from igedi.app.jobs import feature_files

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
