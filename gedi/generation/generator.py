import math
import os
import random
import re
import xml.etree.ElementTree as ET

from ConfigSpace.configuration_space import Configuration
from gedi.features import compute_features_from_event_data
from gedi.generation.model import create_model
from gedi.generation.scheduler import play_log
from gedi.utils.io_helpers import get_output_key_value_location, dump_features_json, compute_similarity
from pm4py import write_xes
from xml.dom import minidom

RANDOM_SEED = 10

#TODO: Move to io_helpers
def add_extension_before_traces(xes_file):
    def removeextralines(elem):
        hasWords = re.compile("\\w")
        for element in elem.iter():
            if not re.search(hasWords,str(element.tail)):
                element.tail=""
            if not re.search(hasWords,str(element.text)):
                element.text = ""
    # Register the namespace
    ET.register_namespace('', "http://www.xes-standard.org/")

    # Parse the original XML
    tree = ET.parse(xes_file)
    root = tree.getroot()

    # Add extensions
    extensions = [
        {'name': 'Lifecycle', 'prefix': 'lifecycle', 'uri': 'http://www.xes-standard.org/lifecycle.xesext'},
        {'name': 'Time', 'prefix': 'time', 'uri': 'http://www.xes-standard.org/time.xesext'},
        {'name': 'Concept', 'prefix': 'concept', 'uri': 'http://www.xes-standard.org/concept.xesext'}
    ]

    for ext in extensions:
        extension_elem = ET.Element('extension', ext)
        root.insert(0, extension_elem)

    # Add global variables
    globals = [
        {
            'scope': 'event',
            'attributes': [
                {'key': 'lifecycle:transition', 'value': 'complete'},
                {'key': 'concept:name', 'value': '__INVALID__'},
                {'key': 'time:timestamp', 'value': '1970-01-01T01:00:00.000+01:00'}
            ]
        },
        {
            'scope': 'trace',
            'attributes': [
                {'key': 'concept:name', 'value': '__INVALID__'}
            ]
        }
    ]
    for global_var in globals:
        global_elem = ET.Element('global', {'scope': global_var['scope']})
        for attr in global_var['attributes']:
            string_elem = ET.SubElement(global_elem, 'string', {'key': attr['key'], 'value': attr['value']})
        root.insert(len(extensions), global_elem)


    # Pretty print the Xes
    removeextralines(root)
    xml_str = minidom.parseString(ET.tostring(root)).toprettyxml()
    with open(xes_file, "w") as f:
        f.write(xml_str)



class PTLGenerator():
    def __init__(self, configspace=None):
        if configspace is None:
            configspace_tuples = {
                "mode": (5, 40),
                "sequence": (0.01, 1),
                "choice": (0.01, 1),
                "parallel": (0.01, 1),
                "loop": (0.01, 1),
                "silent": (0.01, 1),
                "lt_dependency": (0.01, 1),
                "num_traces": (100, 1001),
                "duplicate": (0),
                "or": (0),
            }
            print(f"WARNING: No configspace specified in config file. Continuing with {configspace_tuples}")
        else:
            configspace_lists = configspace
            configspace_tuples = {}
            for k, v in configspace_lists.items():
                if len(v) == 1:
                    configspace_tuples[k] = v[0]
                else:
                    configspace_tuples[k] = tuple(v)
        self.configspace = configspace_tuples

    def generate_log(self, config: Configuration, feature_keys, seed: int = 0):
        random.seed(RANDOM_SEED)
        tree = create_model(config)
        random.seed(RANDOM_SEED)
        log = play_log(tree, config)
        features = compute_features_from_event_data(feature_keys, log)
        return log, features

    def generate_optimized_log(self, config: Configuration, output_path, objectives, identifier=""):
        if isinstance(config, list):
            config = config[0]
        feature_keys = objectives.keys()
        generation_keys = {k: v for k, v in objectives.items() if v is not None and
                           not (isinstance(v, float) and math.isnan(v))}
        log, generated_features = self.generate_log(config, generation_keys)

        identifier = "genEL" +str(identifier)
        random.seed(RANDOM_SEED)
        save_path = get_output_key_value_location(objectives,
                                            output_path, identifier)+".xes"
        write_xes(log, save_path)
        add_extension_before_traces(save_path)
        print("SUCCESS: Saved generated event log in", save_path)
        features_to_dump = generated_features

        features_to_dump['log']= os.path.split(save_path)[1].split(".")[0]
        # calculating the manhattan distance of the generated log to the target features
        #features_to_dump['distance_to_target'] = calculate_manhattan_distance(objectives, features_to_dump)
        features_to_dump['target_similarity'] = compute_similarity(objectives, features_to_dump)
        dump_features_json(features_to_dump, save_path)

        return {
        #"configuration": config,
        #"log": log,
        "features": generated_features,
        }
