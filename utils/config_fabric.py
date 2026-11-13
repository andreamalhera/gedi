import json
import os
import streamlit as st
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from igedi.app.config import HOSTED, STEP_TITLES
from igedi.app.header import play_header
from igedi.app.jobs import start_generation, wait_for_generation
from igedi.app.results import show_results
from igedi.app.state import init_state, session_workdir
from igedi.app.targets import set_generator_targets

st.set_page_config(layout='wide')

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
