import glob
import json
import os
import pandas as pd
import re
import shutil
import numpy as np
from collections import defaultdict
from gedi.utils.target_validation import feature_similarity, is_numeric
from pathlib import PurePath
from scipy.spatial.distance import euclidean

def select_instance(source_dir, log_path, destination=os.path.join("output","generated","instance_selection")):
    os.makedirs(destination, exist_ok=True)
    try:
        source_path=glob.glob(os.path.join(source_dir, log_path))[0]
        destination_path = os.path.join(destination, "_".join(source_path.rsplit("/")[-2:]))
        shutil.copyfile(source_path, destination_path)
    except IndexError:
        print(f"ERROR: No files found for {source_dir}{log_path}. Continuing.")
    return destination, len(os.listdir(destination))

def read_csvs(input_path, ref_feature):
    f_dict = defaultdict(pd.DataFrame)
    ref_short_name = get_keys_abbreviation([ref_feature])
    for file in glob.glob(f'{input_path}*.csv'):
        if ref_short_name in file[:-4].split(os.sep)[-1].split("_"):
            c_file = pd.read_csv(file, delimiter=",")
            if c_file.columns[0] == 'task':
                c_file = c_file.reindex(columns=[c_file.columns[1], c_file.columns[2], c_file.columns[0]])
                c_file.rename(columns={"task":"log"}, inplace=True)
            f_dict[c_file.columns[0] if c_file.columns[0]!=ref_feature else c_file.columns[1]] = c_file
    return f_dict


def sort_files(data):
    """
    Returns a alphanumeric sortered list
    """
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split("([0-9]+)", key)]
    return sorted(data, key=alphanum_key)

def get_keys_abbreviation(obj_keys):
    abbreviated_keys = []
    for obj_key in obj_keys:
        key_slices = obj_key.split("_")
        chars = []
        for key_slice in key_slices:
            for idx, single_char in enumerate(key_slice):
                if idx == 0 or single_char.isdigit():
                    chars.append(single_char)
        abbreviated_key = ''.join(chars)
        abbreviated_keys.append(abbreviated_key)
    return '_'.join(abbreviated_keys)

def get_generated_file_name(obj, identifier):
    obj_sorted = dict(sorted(obj.items()))
    obj_values = [round(x, 4) for x in [*obj_sorted.values()]]
    obj_values_joined = '_'.join(map(str, obj_values)).replace('.', '')
    return f"{identifier}_{obj_values_joined}"

def get_output_key_value_location(obj, output_path, identifier):
    obj_keys = sorted(obj.keys())
    folder_path = os.path.join(output_path, f"{len(obj_keys)}_{get_keys_abbreviation(obj_keys)}")
    generated_file_name = get_generated_file_name(obj, identifier)

    os.makedirs(folder_path, exist_ok=True)
    save_path = os.path.join(folder_path, generated_file_name)
    return save_path

def dump_features_json(features: dict, output_path, objectives=None, content_type="features", decimals=8):
    identifier = features['log']
    output_parts = PurePath(output_path.split(".xes")[0]).parts[:-1] if output_path.endswith(".xes") else PurePath(output_path).parts
    feature_dir = os.path.join(output_parts[0],
                                   *output_parts[1:], content_type)
    if objectives is not None:
        json_path = get_output_key_value_location(objectives,
                                                feature_dir, identifier)+".json"
    else:
        json_path = os.path.join(feature_dir, identifier)+".json"

    os.makedirs(os.path.split(json_path)[0], exist_ok=True)
    features = {k: round(v, decimals) if isinstance(v, (int, float)) else v for k, v in features.items()}

    with open(json_path, 'w') as fp:
        #print("DUMPING: ", len(features), type(features), features)
        json.dump(features, fp, default=int)
        print(f"SUCCESS: Saved {len(features)-2} {content_type} in {json_path}")#-2 because 'log' and 'target_similarities' are not features

def compute_feature_similarities(v1, v2):
    """
    Returns {feature: similarity in [0, 1]} for each feature shared by v1 and v2.
    Bounded features ([0, 1]) use 1 - absolute difference, unbounded ones 1 - symmetric relative difference.
    """
    # Identify common numeric keys, skipping NaNs (features a target leaves unset)
    common_keys = set(v1.keys()).intersection(set(v2.keys())) - {"log", "target_similarity"}
    numeric_keys = [k for k in sorted(common_keys) if is_numeric(v1[k]) and is_numeric(v2[k])]

    return {key: feature_similarity(key, float(v1[key]), float(v2[key])) for key in numeric_keys}

def compute_similarity(v1, v2):
    similarities = compute_feature_similarities(v1, v2)

    if not similarities:
        print("[ERROR]: No common numeric keys found for similarity calculation.")
        return None

    # Average of per-feature similarities
    target_similarity = float(np.mean(list(similarities.values())))
    return target_similarity
