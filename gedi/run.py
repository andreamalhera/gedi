import pandas as pd

from datetime import datetime as dt
from gedi.augmentation import InstanceAugmentator
from gedi.benchmark import BenchmarkTest
from gedi.config import get_model_params_list
from gedi.features import EventDataFeatures
from gedi.generation.generator import PTLGenerator
from gedi.generation.hpo import GediTask
from gedi.plotter import BenchmarkPlotter, FeaturesPlotter, AugmentationPlotter, GenerationPlotter
from gedi.utils.default_argparse import ArgParser
from gedi.utils.param_keys import PARAMS, PIPELINE_STEP
from gedi.utils.param_keys.generator import CONFIG_SPACE, TARGETS, SYSTEM_PARAMS

def run(kwargs:dict, model_params_list: list, filename_list:list):
    """
    This function chooses the running option for the program.
    @param kwargs: dict
        contains the running parameters and the event-log file information
    @param model_params_list: list
        contains a list of model parameters, which are used to analyse this different models.
    @param filename_list: list
        contains the list of the filenames to load multiple event-logs
    @return:
    """
    params = kwargs[PARAMS]
    ft = EventDataFeatures(None)
    augmented_ft = InstanceAugmentator()
    gen = pd.DataFrame(columns=['features'])

    for model_params in model_params_list:
        if model_params.get(PIPELINE_STEP) == 'instance_augmentation':
            augmented_ft = InstanceAugmentator(aug_params=model_params, samples=ft.feat)
            AugmentationPlotter(augmented_ft, model_params)
        elif model_params.get(PIPELINE_STEP) == 'event_logs_generation':
            gen = pd.DataFrame(GediTask(params=model_params,
                                        embedded_generator = PTLGenerator,
                                        targets = model_params.get(TARGETS),
                                        config_space = model_params.get(CONFIG_SPACE),
                                        system_params = model_params.get(SYSTEM_PARAMS)).generated_features)
            #gen = pd.DataFrame(GediTask(model_params).generated_features)
            #gen = pd.read_csv("output/features/generated/grid_2objectives_enseef_enve/2_enseef_enve_feat.csv")
            #GenerationPlotter(gen, model_params, output_path="output/plots")
        elif model_params.get(PIPELINE_STEP) == 'benchmark_test':
            benchmark = BenchmarkTest(model_params)#, event_logs=gen['log'])
            # BenchmarkPlotter(benchmark.features, output_path="output/plots")
        elif model_params.get(PIPELINE_STEP) == 'feature_extraction':
            ft = EventDataFeatures(**kwargs, ft_params=model_params)
            FeaturesPlotter(ft.feat, model_params)
        elif model_params.get(PIPELINE_STEP) == "evaluation_plotter":
            GenerationPlotter(gen, model_params, output_path=model_params['output_path'], input_path=model_params['input_path'])

def gedi(config_path):
    """
    This function runs the GEDI pipeline.
    @param config_path: str
        contains the path to the config file
    @return:
    """
    model_params_list = get_model_params_list(config_path)
    run({'params':""}, model_params_list, [])