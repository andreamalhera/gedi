import pandas as pd
import pytest
from gedi.run import gedi
from gedi.generation.generator import PTLGenerator
from gedi.generation.hpo import GediTask, GenerateEventLogs
from gedi.utils.io_helpers import compute_similarity
from gedi.utils.target_validation import validate_target
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
    genED = GediTask(INPUT_PARAMS)
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

def test_generator_wrapper_skips_failing_target(monkeypatch):
    """So that one bad target can't take down the whole multiprocessing.Pool.map() run 
    (see GediTask.__init__, which filters results on 'features' in config)."""
    def failing_optimize(self, system_params):
        raise RuntimeError("simulated failure")
    monkeypatch.setattr(GediTask.HPOTask, "optimize", failing_optimize)

    task_instance = object.__new__(GediTask)
    task_instance.output_path = "output"
    task = (0, pd.Series({"ratio_top_20_variants": 0.2, "log": "dummy"}))

    result = task_instance.generator_wrapper(
        task,
        embedded_generator=PTLGenerator,
        config_space={'mode': [5, 20], 'sequence': [0.01, 1], 'choice': [0.01, 1], 'parallel': [0.01, 1],
                      'loop': [0.01, 1], 'silent': [0.01, 1], 'lt_dependency': [0.01, 1], 'num_traces': [10, 100],
                      'duplicate': [0], 'or': [0]},
        system_params={'n_trials': 2})

    assert result == {}

def test_GediTask_skips_unknown_feature_target():
    """So that a target with an unknown feature (i.e., one that is not in
    feeed.feature_extractor.feature_type) doesn't crash the whole GediTask run.
    The other, valid target must still produce results."""
    INPUT_PARAMS = {'targets': [
                        {'ratio_top_20_variants': 0.2},
                        {'this_feature_does_not_exist_xyz': 0.5},
                    ],
                    'config_space': {'mode': [5, 20], 'sequence': [0.01, 1], 'choice': [0.01, 1], 'parallel': [0.01, 1],
                                     'loop': [0.01, 1], 'silent': [0.01, 1], 'lt_dependency': [0.01, 1], 'num_traces': [10, 10001],
                                     'duplicate': [0], 'or': [0]},
                    'system_params': {'n_trials': 2}}
    genED = GediTask(INPUT_PARAMS,
                     embedded_generator = PTLGenerator,
                     targets = INPUT_PARAMS.get(TARGETS))

    assert len(genED.generated_features) == 1
    assert 'ratio_top_20_variants' in genED.generated_features[0]['features']

def test_validate_target():
    CONFIG_SPACE = {'num_traces': [10, 100]}
    errors, warnings = validate_target({'epa_normalized_sequence_entropy_exponential_forgetting': 832.0,
                                        'n_variants': 828.0}, CONFIG_SPACE)
    assert len(errors) == 1 and 'epa_normalized_sequence_entropy_exponential_forgetting' in errors[0]
    assert any('n_variants' in w and 'num_traces' in w for w in warnings)

    errors, _ = validate_target({'n_traces': 50, 'n_variants': 60})
    assert any('cannot exceed n_traces' in e for e in errors)

    errors, _ = validate_target({'n_variants': -1, 'trace_len_skewness': -0.5})
    assert len(errors) == 1 and 'n_variants' in errors[0]

    errors, warnings = validate_target({'log': 'BPI_2012', 'ratio_top_20_variants': 0.4,
                                        'epa_normalized_sequence_entropy_linear_forgetting': float('nan')}, CONFIG_SPACE)
    assert errors == [] and warnings == []

def test_compute_similarity():
    # Bounded features: 1 - absolute difference, also beyond the range observed in BPIC logs
    assert compute_similarity({'epa_normalized_sequence_entropy': 0.6},
                              {'epa_normalized_sequence_entropy': 0.95}) == pytest.approx(0.65)
    # Unbounded features: 1 - symmetric relative difference
    assert compute_similarity({'n_variants': 828}, {'n_variants': 100}) == pytest.approx(200 / 928)
    assert compute_similarity({'n_variants': 0}, {'n_variants': 0}) == 1.0
    # Average over features, ignoring 'log' and features the target leaves unset
    similarity = compute_similarity({'n_variants': 10, 'ratio_top_20_variants': 0.3, 'trace_len_mean': float('nan')},
                                    {'n_variants': 10, 'ratio_top_20_variants': 0.5, 'trace_len_mean': 4.0, 'log': 'genEL'})
    assert similarity == pytest.approx(0.9)

def test_GediTask_skips_impossible_target():
    INPUT_PARAMS = {'targets': [
                        {'ratio_top_20_variants': 0.2},
                        {'ratio_top_20_variants': 832.0},
                    ],
                    'config_space': {'mode': [5, 20], 'sequence': [0.01, 1], 'choice': [0.01, 1], 'parallel': [0.01, 1],
                                     'loop': [0.01, 1], 'silent': [0.01, 1], 'lt_dependency': [0.01, 1], 'num_traces': [10, 100],
                                     'duplicate': [0], 'or': [0]},
                    'system_params': {'n_trials': 2}}
    genED = GediTask(INPUT_PARAMS,
                     embedded_generator = PTLGenerator,
                     targets = INPUT_PARAMS.get(TARGETS),
                     config_space = INPUT_PARAMS.get(CONFIG_SPACE),
                     system_params = INPUT_PARAMS.get(SYSTEM_PARAMS))

    assert len(genED.generated_features) == 1
    assert 0.0 <= genED.generated_features[0]['features']['target_similarity'] <= 1.0
