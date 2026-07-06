import multiprocessing
import os
import pandas as pd
import random
import warnings
from ConfigSpace import Configuration, ConfigurationSpace
from datetime import datetime as dt
from feeed.activities import Activities as activities
from feeed.end_activities import EndActivities as end_activities
from feeed.epa_based import Epa_based as epa_based
from feeed.eventropies import Eventropies as eventropies
from feeed.feature_extractor import feature_type
from feeed.simple_stats import SimpleStats as simple_stats
from feeed.start_activities import StartActivities as start_activities
from feeed.trace_length import TraceLength as trace_length
from feeed.trace_variant import TraceVariant as trace_variant
from pm4py import write_xes
from pm4py.sim import play_out
from smac import HyperparameterOptimizationFacade, Scenario
from gedi.features import compute_features_from_event_data
from gedi.generation.generator import PTLGenerator, add_extension_before_traces
from gedi.utils.column_mappings import column_mappings
from gedi.utils.io_helpers import read_csvs
from gedi.utils.param_keys import OUTPUT_PATH, INPUT_PATH
from gedi.utils.param_keys.generator import GENERATOR_PARAMS, TARGETS, CONFIG_SPACE
from gedi.utils.param_keys.generator import N_TRIALS, GENERATOR, GENERATOR_TYPE, SYSTEM_PARAMS
from functools import partial

RANDOM_SEED = 10
random.seed(RANDOM_SEED)

def get_tasks(targets, output_path="", reference_feature=None):
    #Read tasks from file.
    if isinstance(targets, str) and targets.endswith(".csv"):
        tasks = pd.read_csv(targets, index_col=None)
        output_path=os.path.join(output_path,os.path.split(targets)[-1].split(".")[0])
        if 'task' in tasks.columns:
            tasks.rename(columns={"task":"log"}, inplace=True)
    elif isinstance(targets, str) and os.path.isdir(os.path.join(os.getcwd(), targets)):
        tasks = read_csvs(targets, reference_feature)
    #Read tasks from a real log features selection.
    elif isinstance(targets, dict) and INPUT_PATH in targets.keys():
        output_path=os.path.join(output_path,os.path.split(targets.get(INPUT_PATH))[-1].split(".")[0])
        tasks = pd.read_csv(targets.get(INPUT_PATH), index_col=None)
        id_col = tasks.select_dtypes(include=['object']).dropna(axis=1).columns[0]
        if "objectives" in targets.keys():
            incl_cols = targets["objectives"]
            tasks = tasks[(incl_cols +  [id_col])]
    # TODO: Solve/Catch error for different objective keys.
    #Read tasks from config_file with list of targets
    elif isinstance(targets, list):
        tasks = pd.DataFrame.from_dict(data=targets)
    #Read single tasks from config_file
    elif isinstance(targets, dict):
        tasks = pd.DataFrame.from_dict(data=[targets])
    else:
        raise FileNotFoundError(f"{targets} not found. Please check path in filesystem.")
    return tasks, output_path

def GenerateEventLogs(*args, **kwargs):
    warnings.warn(
        "'GenerateEventLogs' is deprecated and will be removed in a future version. Use 'GediTask' instead.",
        DeprecationWarning
    )
    return GediTask(*args, **kwargs)

class GediTask():
    """
    Generates event logs with the provided parameters.
    @param params: dict
        contains the generator parameters
        targets: contains a dict with the desired feature values
        config_space: contains a dict with possible configuration parameter ranges
        n_trials: contains the number of trials
        #system_params: contains the system parameters, which don't change e.g. n_trials
        #generator_type: contains the generator type as a string
    @return: None
    """
    def __init__(self, params = None, embedded_generator = None, targets = None, config_space = None, system_params = None) -> None:
        print("=========================== Generator ==========================")
        if embedded_generator is None and targets is None and config_space is None and system_params is None:
            if params.get("generator_params") is not None and set(params.get("generator_params").keys()) == set(['experiment', 'config_space', 'n_trials']):
                targets = params.get('generator_params').get('experiment')
                config_space = params.get('generator_params').get('config_space')
                system_params = {'n_trials': params.get('generator_params').get('n_trials')}
            else:
                raise TypeError(f"Missing 'params'. Please provide a dictionary with generator valid parameters. See https://github.com/andreamalhera/gedi for more info.")

        # If generator is not suitable, raise an error.
        if embedded_generator is not None and (not hasattr(embedded_generator, 'generate_log') or not hasattr(embedded_generator, 'generate_optimized_log')):
            raise ValueError("Unknown generator type. Please provide a valid generator or None. See https://github.com/andreamalhera/gedi for more info.")

        tasks, system_params = self.setup_GediTask(targets = targets,
                                                      system_params = system_params)
        start = dt.now()
        if True: #try:
            self.feature_keys = sorted([feature for feature in tasks.columns.tolist() if feature != "log"])
            num_cores = multiprocessing.cpu_count() if len(tasks) >= multiprocessing.cpu_count() else len(tasks)
            #NOTE FOR DEBUGGING USE THE LINE BELOW
            #self.generator_wrapper([*tasks.iterrows()][0], embedded_generator = embedded_generator, config_space = config_space, system_params = system_params)#TESTING
            with multiprocessing.Pool(num_cores) as p:
                print(f"INFO: Generator starting at {start.strftime('%H:%M:%S')} using {num_cores} cores for {len(tasks)} tasks...")
                random.seed(RANDOM_SEED)
                partial_wrapper = partial(self.generator_wrapper,
                                          embedded_generator = embedded_generator,
                                          config_space = config_space,
                                          system_params = system_params)
                generated_features = p.map(partial_wrapper, [(index, row) for index, row in tasks.iterrows()])
            self.generated_features = [
                        {
                            #'log': config.get('log'),
                            'features': config.get('features')}
                            for config in generated_features
                            if 'features' in config #and 'log' in config
                    ]
        print(f"SUCCESS: Generator took {dt.now()-start} sec. Generated {len(self.generated_features)} event log(s).")
        print(f"         Saved generated logs in {self.output_path}")
        print("========================= ~ Generator ==========================")


    def clear(self):
        print("Clearing parameters...")
        self.generated_features = None
        # self.configs = None
        # self.params = None
        self.output_path = None
        self.feature_keys = None

    def setup_GediTask(self, targets=None, system_params=None, params = None):
        tasks = None
        if targets is None and system_params is None:
            default_params = {'generator_params': {'targets': {'ratio_top_20_variants': 0.2, 'epa_normalized_sequence_entropy_linear_forgetting': 0.4},
                                                   'config_space': {'mode': [5, 20], 'sequence': [0.01, 1], 'choice': [0.01, 1], 'parallel': [0.01, 1],
                                                                    'loop': [0.01, 1], 'silent': [0.01, 1], 'lt_dependency': [0.01, 1], 'num_traces': [10, 101], 'duplicate': [0], 'or': [0]},
                                                   'system_params': {'n_trials': 50}
                                                   }}
            raise TypeError(f"Missing 'params'. Please provide a dictionary with generator parameters as so: {default_params}. See https://github.com/andreamalhera/gedi for more info.")
        print(f"INFO: Running with {targets}, {system_params}")
        if system_params is None:
            system_params = {'output_path': None, 'n_trials': None}

        self.output_path = 'data/generated' if system_params.get(OUTPUT_PATH) is None else system_params.get(OUTPUT_PATH)
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path, exist_ok=True)

        if self.output_path.endswith('csv'):
            self.generated_features = pd.read_csv(self.output_path)
            return

        ## SET UP: Tasks from targets
        #TODO: Compatibility to older versions
        """'
        if targets is None:
            targets = generator_params.get("experiment")#Compatibility with older versions
        """
        if targets is None:
                raise TypeError(f"Missing 'targets'. Please provide a dictionary with generator parameters as so: {default_params}. See https://github.com/andreamalhera/gedi for more info.")

        tasks, output_path = get_tasks(targets, self.output_path)
        columns_to_rename = {col: column_mappings()[col] for col in tasks.columns if col in column_mappings()}
        tasks = tasks.rename(columns=columns_to_rename)
        self.output_path = output_path
        return tasks, system_params

    def generator_wrapper(self, task, embedded_generator = None, config_space = None, system_params = None):
        output_path = system_params.get(OUTPUT_PATH) if system_params.get(OUTPUT_PATH) is not None else self.output_path
        embedded_generator = embedded_generator(config_space) if embedded_generator is not None else PTLGenerator(config_space)
        hpo_task = self.HPOTask(task, embedded_generator)
        configs = hpo_task.optimize(system_params)
        objectives = dict(sorted(task[1].drop('log', errors='ignore').to_dict().items()))
        random.seed(RANDOM_SEED)
        generated_features = embedded_generator.generate_optimized_log(config=configs,
                                                                       output_path=output_path,
                                                                       objectives=objectives,
                                                                       identifier = hpo_task.identifier)
        return generated_features


    class HPOTask():
        def __init__(self, task, embedded_generator):
            # TODO Asses removing 'identifier', pros: less irrelevant code specially for generation from scratch, cons: harder to map targets and when reproducing BPICS
            try:
                identifier = [x for x in task[1] if isinstance(x, str)][0]
                identifier = str(identifier)
            except IndexError:
                identifier = ""

            task = task[1].drop('log', errors='ignore')
            self.objectives = task.dropna().to_dict()
            self.identifier = identifier
            self.embedded_generator = embedded_generator
            return

        def gen_log(self, config: Configuration, seed: int = RANDOM_SEED):
            random.seed(RANDOM_SEED)
            feature_keys = [*self.objectives.keys()]
            _ , features= self.embedded_generator.generate_log(config=config,
                                                               feature_keys=feature_keys,
                                                               seed=seed)
            del _
            random.seed(RANDOM_SEED)
            result = self.eval_log(features)
            return result

        def eval_log(self, features):
            log_evaluation = {}
            for key in self.objectives.keys():
                log_evaluation[key] = abs(self.objectives[key] - features[key])
            return log_evaluation

        def optimize(self, system_params):
            #TODO: Add compatibility with older versions
            n_trials = system_params.get(N_TRIALS) if system_params.get(N_TRIALS) is not None else 20
            print(f"INFO: Running with n_trials={n_trials}")

            objectives = [*self.objectives.keys()]
            configspace = ConfigurationSpace(self.embedded_generator.configspace)

            # Scenario object specifying the multi-objective optimization environment
            scenario = Scenario(
                configspace,
                deterministic=True,
                n_trials=n_trials,
                objectives=objectives,
                n_workers=-1
            )

            # Use SMAC to find the best configuration/hyperparameters
            random.seed(RANDOM_SEED)
            multi_obj = HyperparameterOptimizationFacade.get_multi_objective_algorithm(
                    scenario,
                    objective_weights=[1]*len(self.objectives),
                )

            random.seed(RANDOM_SEED)
            smac = HyperparameterOptimizationFacade(
                scenario=scenario,
                target_function=self.gen_log,
                multi_objective_algorithm=multi_obj,
                # logging_level=False,
                overwrite=True,
            )

            random.seed(RANDOM_SEED)
            incumbent = smac.optimize()
            return incumbent
