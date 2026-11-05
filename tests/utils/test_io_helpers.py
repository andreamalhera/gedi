import json
import os
from gedi.utils.io_helpers import dump_features_json, get_generated_file_name

def test_dump_features_json():
    features = {'log': 'test_log'}
    identifier = "identifier_directory"
    output_path = "data/test"
    dump_features_json(features, os.path.join(output_path,identifier), content_type="features")

    json_path = os.path.join(output_path, identifier, "features", "test_log.json")
    assert os.path.exists(json_path)
    with open(json_path) as fp:
        assert json.load(fp) == features

def test_get_generated_file_name():
    assert get_generated_file_name({'n_variants': 1.0}, "genEL") == "genEL_1p0"
    assert get_generated_file_name({'n_variants': 10}, "genEL") == "genEL_10"
    assert get_generated_file_name({'trace_len_skewness': -0.44444}, "genEL") == "genEL_m0p4444"
    assert get_generated_file_name({'n_variants': float('nan')}, "genEL") == "genEL_nan"
    # Values are sorted by feature name
    assert get_generated_file_name({'ratio_top_10_variants': 0.5, 'n_variants': 3}, "genEL1") == "genEL1_3_0p5"
