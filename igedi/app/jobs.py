import os
import shutil
import subprocess
import sys
import time
import streamlit as st

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
