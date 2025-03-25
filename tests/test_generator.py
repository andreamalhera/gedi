import pandas as pd
import pytest
from gedi.run import gedi
from gedi.generation.generator import PTLGenerator
from gedi.generation.hpo import GediTask, GenerateEventLogs
from gedi.utils.param_keys.features import FEATURE_SET, FEATURE_PARAMS
from gedi.utils.param_keys.generator import TARGETS, CONFIG_SPACE, SYSTEM_PARAMS, GENERATOR_PARAMS

def test_GediTask_args():
    INPUT_PARAMS = {'targets': {'input_path': 'data/test/grid_feat.csv',
                                    'objectives': ['ratio_top_20_variants',
                                                    'epa_normalized_sequence_entropy_linear_forgetting']},
                          'config_space': {'mode': [5, 20], 'sequence': [0.01, 1],
                                                    'choice': [0.01, 1], 'parallel': [0.01, 1],
                                                    'loop': [0.01, 1], 'silent': [0.01, 1],
                                                    'lt_dependency': [0.01, 1], 'num_traces': [10, 100],
                                                    'duplicate': [0], 'or': [0]},
                          'system_params': {'n_trials': 50}
                        }
    VALIDATION_OUTPUT = [0.89, 0.7, 0.89, 1.0]
    genED = GediTask(INPUT_PARAMS,
                     embedded_generator = PTLGenerator,
                     targets = INPUT_PARAMS.get(TARGETS),
                     config_space = INPUT_PARAMS.get(CONFIG_SPACE),
                     system_params = INPUT_PARAMS.get(SYSTEM_PARAMS))
    similarities = [round(target['features']['target_similarity'], 2) for target in genED.generated_features]

    assert len(similarities) == len(VALIDATION_OUTPUT)
    AGREEMENT_THRESHOLD = 0.75
    agreement = sum(1 for o, u in zip(similarities, VALIDATION_OUTPUT) if o == u) / len(VALIDATION_OUTPUT) * 100
    assert agreement >= AGREEMENT_THRESHOLD

def test_GediTask():
    INPUT_PARAMS = {'targets': {'input_path': 'data/test/grid_feat.csv',
                                'objectives': ['ratio_top_20_variants',
                                               'epa_normalized_sequence_entropy_linear_forgetting']}
                    }
    VALIDATION_OUTPUT = [0.89, 0.7, 0.89, 1.0]
    genED = GediTask(INPUT_PARAMS,
                     embedded_generator = PTLGenerator,
                     targets = INPUT_PARAMS.get(TARGETS))
    similarities = [round(target['features']['target_similarity'], 2) for target in genED.generated_features]

    assert len(similarities) == len(VALIDATION_OUTPUT)
    AGREEMENT_THRESHOLD = 0.75
    agreement = sum(1 for o, u in zip(similarities, VALIDATION_OUTPUT) if o == u) / len(VALIDATION_OUTPUT) * 100
    assert agreement >= AGREEMENT_THRESHOLD

def test_GediTask_103_compatibility():
    INPUT_PARAMS = {'generator_params': {'experiment': {'input_path': 'data/test/grid_feat.csv',
                                                        'objectives': ['ratio_top_20_variants',
                                                                       'epa_normalized_sequence_entropy_linear_forgetting']},
                                         'config_space': {'mode': [5, 20], 'sequence': [0.01, 1],
                                                          'choice': [0.01, 1], 'parallel': [0.01, 1],
                                                          'loop': [0.01, 1], 'silent': [0.01, 1],
                                                          'lt_dependency': [0.01, 1], 'num_traces': [10, 10001],
                                                          'duplicate': [0], 'or': [0]},
                                         'n_trials': 50}}
    VALIDATION_OUTPUT = [0.89, 0.7, 0.89, 1.0]
    genED = GenerateEventLogs(INPUT_PARAMS)
    similarities = [round(target['features']['target_similarity'], 2) for target in genED.generated_features]

    assert len(similarities) == len(VALIDATION_OUTPUT)
    AGREEMENT_THRESHOLD = 0.75
    agreement = sum(1 for o, u in zip(similarities, VALIDATION_OUTPUT) if o == u) / len(VALIDATION_OUTPUT) * 100
    assert agreement >= AGREEMENT_THRESHOLD

def test_GediTask_pypible():
    INPUT_PARAMS =  {'targets':[
                        {"ratio_top_20_variants": 0.2, "epa_normalized_sequence_entropy_linear_forgetting": 0.4},
                        {"ratio_top_20_variants": 0.4, "epa_normalized_sequence_entropy_linear_forgetting": 0.7},
                        {"epa_normalized_sequence_entropy_linear_forgetting": 0.4},
                        {"ratio_top_20_variants": 0.2}
                        ],
                        'config_space': {'mode': [5, 20], 'sequence': [0.01, 1],
                                        'choice': [0.01, 1], 'parallel': [0.01, 1],
                                        'loop': [0.01, 1], 'silent': [0.01, 1],
                                        'lt_dependency': [0.01, 1], 'num_traces': [10, 10001],
                                        'duplicate': [0], 'or': [0]},
                        'system_params': {'n_trials': 50}}
    VALIDATION_OUTPUT = [0.89, 0.7, 0.89, 1.0]
    genED = GediTask(INPUT_PARAMS,
                     embedded_generator = PTLGenerator,
                     targets = INPUT_PARAMS.get(TARGETS))
    similarities = [round(target['features']['target_similarity'], 2) for target in genED.generated_features]

    assert len(similarities) == len(VALIDATION_OUTPUT)
    AGREEMENT_THRESHOLD = 0.75
    agreement = sum(1 for o, u in zip(similarities, VALIDATION_OUTPUT) if o == u) / len(VALIDATION_OUTPUT) * 100
    assert agreement >= AGREEMENT_THRESHOLD

def test_abbr_GediTask():
    INPUT_PARAMS = {'targets': {'input_path': 'data/test/igedi_table_1.csv',
                                'objectives': ['rmcv', 'ense']},
                    'config_space': {'mode': [5, 20], 'sequence': [0.01, 1], 'choice': [0.01, 1], 'parallel': [0.01, 1],
                                     'loop': [0.01, 1], 'silent': [0.01, 1], 'lt_dependency': [0.01, 1], 'num_traces': [10, 10001],
                                     'duplicate': [0], 'or': [0]},
                    'system_params': {'n_trials': 2}}
    VALIDATION_OUTPUT = [0.9, 0.7, 0.7]
    genED = GediTask(INPUT_PARAMS,
                     embedded_generator = PTLGenerator,
                     targets = INPUT_PARAMS.get(TARGETS))
    similarities = [round(target['features']['target_similarity'], 1) for target in genED.generated_features]
    assert len(similarities) == len(VALIDATION_OUTPUT)
    AGREEMENT_THRESHOLD = 0.75
    agreement = sum(1 for o, u in zip(similarities, VALIDATION_OUTPUT) if abs(o-u)<=0.1) / len(VALIDATION_OUTPUT) * 100
    assert agreement >= AGREEMENT_THRESHOLD
