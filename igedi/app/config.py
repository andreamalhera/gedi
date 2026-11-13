import os

INPUT_XES="output/inputlog_temp.xes"
LOGO_PATH="gedi/utils/logo.png"

# Hugging Face sets SPACE_ID in the Space runtime. When hosted, users share one machine, so
# output paths are per-session temp dirs (not editable) and job size is capped.
HOSTED = bool(os.environ.get("SPACE_ID"))
MAX_TARGETS = int(os.environ.get("IGEDI_MAX_TARGETS", 30))
MAX_TRIALS = int(os.environ.get("IGEDI_MAX_TRIALS", 100))

IGEDI_CITATION = ("A. Maldonado, S. A. Aryasomayajula, C. M. M. Frey, T. Seidl. "
                  "iGEDI: interactive Generating Event Data with Intentional Features. "
                  "ICPM 2024 Tool Demonstration Track, CEUR Workshop Proceedings, Vol. 3783, 2024.")

EXAMPLE_FEATURES = ["epa_normalized_variant_entropy", "ratio_most_common_variant"]
EXAMPLE_VALUES = {"epa_normalized_variant_entropy": [0.3, 0.5, 0.7],
                  "ratio_most_common_variant": [0.1, 0.3, 0.5]}

CONFIG_HELP = {
    "mode": "Most frequent number of visible activities in the process tree.",
    "sequence": "Probability of adding a sequence operator to the tree.",
    "choice": "Probability of adding a choice operator to the tree.",
    "parallel": "Probability of adding a parallel operator to the tree.",
    "loop": "Probability of adding a loop operator to the tree.",
    "silent": "Probability of adding a silent activity to a choice or loop operator.",
    "lt_dependency": "Probability of adding a random long-term dependency to the tree.",
    "num_traces": "Number of traces in the generated log.",
    "duplicate": "Probability of duplicating an activity label.",
    "or": "Probability of adding an OR operator to the tree.",
    "time_spread": "Exponent of the artificial timestamps: 1 = one second between events; "
                   "above 1 bunches events at the start of the log, below 1 at the end.",
}
INT_CONFIG_BOUNDS = {"mode": (1, 50), "num_traces": (1, 1000)}

SIMILARITY_GOAL = 0.9

STEP_TITLES = {
    "event_logs_generation": "Generate event logs",
    "feature_extraction": "Extract features",
}
