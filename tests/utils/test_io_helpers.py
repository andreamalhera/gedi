import json
import os
from gedi.utils.io_helpers import dump_features_json

def test_dump_features_json():
    features = {'log': 'test_log'}
    identifier = "identifier_directory"
    output_path = "data/test"
    dump_features_json(features, os.path.join(output_path,identifier), content_type="features")

    json_path = os.path.join(output_path, identifier, "features", "test_log.json")
    assert os.path.exists(json_path)
    with open(json_path) as fp:
        assert json.load(fp) == features
