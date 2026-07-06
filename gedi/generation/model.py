import random
from pm4py import generate_process_tree

def create_PTLG(config, random_seed = 10):
    RANDOM_SEED = random_seed
    """
    Parameters
        --------------
        parameters
            Parameters of the algorithm, according to the paper:
            - Parameters.MODE: most frequent number of visible activities
            - Parameters.MIN: minimum number of visible activities
            - Parameters.MAX: maximum number of visible activities
            - Parameters.SEQUENCE: probability to add a sequence operator to tree
            - Parameters.CHOICE: probability to add a choice operator to tree
            - Parameters.PARALLEL: probability to add a parallel operator to tree
            - Parameters.LOOP: probability to add a loop operator to tree
            - Parameters.OR: probability to add an or operator to tree
            - Parameters.SILENT: probability to add silent activity to a choice or loop operator
            - Parameters.DUPLICATE: probability to duplicate an activity label
            - Parameters.NO_MODELS: number of trees to generate from model population
    """
    random.seed(RANDOM_SEED)
    tree = generate_process_tree(parameters={
        "min": config["mode"],
        "max": config["mode"],
        "mode": config["mode"],
        "sequence": config["sequence"],
        "choice": config["choice"],
        "parallel": config["parallel"],
        "loop": config["loop"],
        "silent": config["silent"],
        "lt_dependency": config["lt_dependency"],
        "duplicate": config["duplicate"],
        "or": config["or"],
        "no_models": 1
    })
    return tree

def create_model(config, model_type = 'PTLG'):
    if model_type == 'PTLG':
        return create_PTLG(config)
