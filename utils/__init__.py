from .param_keys.generator import GENERATOR_PARAMS, target, CONFIG_SPACE, N_TRIALS
from .param_keys import PIPELINE_STEP, INPUT_PATH, OUTPUT_PATH

def function_name(function: callable):
    return str(function).split()[1]


