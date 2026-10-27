import math
import pandas as pd
import pytest
from gedi.run import gedi
from gedi.generation.generator import PTLGenerator
from gedi.generation.hpo import GediTask, GenerateEventLogs
from gedi.utils.io_helpers import compute_similarity
from gedi.utils.target_validation import feature_bounds, fit_window, validate_target
from gedi.utils.param_keys.features import FEATURE_SET, FEATURE_PARAMS
from gedi.utils.param_keys.generator import TARGETS, CONFIG_SPACE, SYSTEM_PARAMS, GENERATOR_PARAMS

AGREEMENT_THRESHOLD = 0.75

def assert_similarities(similarities, expected_windows):
    assert len(similarities) == len(expected_windows)
    assert all(0.0 <= s <= 1.0 for s in similarities)
    agreement = sum(1 for s, (low, high) in zip(similarities, expected_windows) if low <= s <= high) / len(expected_windows)
    assert agreement >= AGREEMENT_THRESHOLD

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
    VALIDATION_OUTPUT = [(0.86, 1.0), (0.73, 0.86), (0.82, 1.0), (1.0, 1.0)]
    genED = GediTask(INPUT_PARAMS,
                     embedded_generator = PTLGenerator,
                     targets = INPUT_PARAMS.get(TARGETS),
                     config_space = INPUT_PARAMS.get(CONFIG_SPACE),
                     system_params = INPUT_PARAMS.get(SYSTEM_PARAMS))
    similarities = [round(target['features']['target_similarity'], 2) for target in genED.generated_features]

    assert_similarities(similarities, VALIDATION_OUTPUT)

def test_GediTask():
    INPUT_PARAMS = {'targets': {'input_path': 'data/test/grid_feat.csv',
                                'objectives': ['ratio_top_20_variants',
                                               'epa_normalized_sequence_entropy_linear_forgetting']}
                    }
    VALIDATION_OUTPUT = [(0.86, 1.0), (0.66, 0.81), (0.86, 0.97), (1.0, 1.0)]
    genED = GediTask(INPUT_PARAMS,
                     embedded_generator = PTLGenerator,
                     targets = INPUT_PARAMS.get(TARGETS))
    similarities = [round(target['features']['target_similarity'], 2) for target in genED.generated_features]

    assert_similarities(similarities, VALIDATION_OUTPUT)

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
    VALIDATION_OUTPUT = [(0.87, 1.0), (0.74, 0.85), (0.86, 1.0), (1.0, 1.0)]
    genED = GediTask(INPUT_PARAMS)
    similarities = [round(target['features']['target_similarity'], 2) for target in genED.generated_features]

    assert_similarities(similarities, VALIDATION_OUTPUT)

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
    VALIDATION_OUTPUT = [(0.88, 1.0), (0.66, 0.84), (0.85, 0.98), (1.0, 1.0)]
    genED = GediTask(INPUT_PARAMS,
                     embedded_generator = PTLGenerator,
                     targets = INPUT_PARAMS.get(TARGETS))
    similarities = [round(target['features']['target_similarity'], 2) for target in genED.generated_features]

    assert_similarities(similarities, VALIDATION_OUTPUT)

def test_abbr_GediTask():
    INPUT_PARAMS = {'targets': {'input_path': 'data/test/igedi_table_1.csv',
                                'objectives': ['rmcv', 'ense']},
                    'config_space': {'mode': [5, 20], 'sequence': [0.01, 1], 'choice': [0.01, 1], 'parallel': [0.01, 1],
                                     'loop': [0.01, 1], 'silent': [0.01, 1], 'lt_dependency': [0.01, 1], 'num_traces': [10, 10001],
                                     'duplicate': [0], 'or': [0]},
                    'system_params': {'n_trials': 2}}
    VALIDATION_OUTPUT = [(1.0, 1.0), (0.6, 1.0), (0.6, 1.0)]
    genED = GediTask(INPUT_PARAMS,
                     embedded_generator = PTLGenerator,
                     targets = INPUT_PARAMS.get(TARGETS))
    similarities = [round(target['features']['target_similarity'], 1) for target in genED.generated_features]

    assert_similarities(similarities, VALIDATION_OUTPUT)

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

class FlakyGenerator():
    """Stub generator whose feature computation fails on selected trials, e.g. feeed's
    distinct_activities_non_overlap raising ZeroDivisionError on a log with an empty trace."""
    def __init__(self, failing_calls, failure):
        self.configspace = {'sequence': (0.01, 1.0), 'loop': (0.01, 1.0)}
        self.failing_calls = failing_calls
        self.failure = failure
        self.calls = 0

    def generate_log(self, config, feature_keys, seed=0):
        self.calls += 1
        if self.calls in self.failing_calls:
            if self.failure == "raise":
                raise ZeroDivisionError("division by zero")
            return None, {key: float('nan') for key in feature_keys}
        return None, {key: config['sequence'] for key in feature_keys}

@pytest.mark.parametrize("failure", ["raise", "nan"])
@pytest.mark.parametrize("failing_calls", [{1}, {5}, {1, 2, 3, 7, 11}])
def test_optimize_survives_failing_trials(failure, failing_calls):
    """A failing trial used to get SMAC's default crash cost (inf), which became NaN
    when normalizing costs and aborted the target with 'Input y contains NaN'."""
    TARGET, TOLERANCE = 0.3, 0.2
    task = (0, pd.Series({'log': 'dummy', 'ratio_top_20_variants': TARGET}))
    generator = FlakyGenerator(failing_calls, failure)
    hpo_task = GediTask.HPOTask(task, generator)

    incumbent = hpo_task.optimize({'n_trials': 20})
    if isinstance(incumbent, list):
        incumbent = incumbent[0]

    assert generator.calls == 20
    # SMAC must still reach TARGET despite the failed trials.
    assert incumbent['sequence'] == pytest.approx(TARGET, abs=TOLERANCE)

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

def test_fit_window():
    # Unbounded features keep the window as is
    assert fit_window('n_traces', 832, 832) == (832, 1664)
    # Bounded features shift back into [0, 1], or shrink to it if the window is too wide
    assert fit_window('ratio_variants_per_number_of_traces', 0.99, 0.2) == pytest.approx((0.8, 1.0))
    assert fit_window('ratio_variants_per_number_of_traces', 0.99, 1.5) == (0.0, 1.0)
    assert fit_window('epa_normalized_sequence_entropy', 0.39, 0.39) == (0.39, 0.78)
    assert fit_window('density_dfg', 0.9, 0.2) == pytest.approx((0.8, 1.0))

def test_feature_bounds_complexity_features():
    assert feature_bounds('separability_ratio_dfg') == (0.0, 1.0)
    assert feature_bounds('simple_trace_diversity') == (0.0, 1.0)
    assert feature_bounds('advanced_trace_diversity') == (0.0, math.inf)
    # structure is negative when ties exceed n_variants**2, e.g. a single variant a->b->c gives -1
    assert feature_bounds('structure') == (-math.inf, 1.0)

    errors, _ = validate_target({'density_dfg': 1.5, 'structure': -1.0})
    assert len(errors) == 1 and 'density_dfg' in errors[0]

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
    # Bounded complexity features also use the absolute difference
    assert compute_similarity({'separability_ratio_dfg': 0.158},
                              {'separability_ratio_dfg': 0.1667}) == pytest.approx(0.9913)

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
