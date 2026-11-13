import os
import matplotlib
matplotlib.use("Agg") #Agg renders off-screen, which is all st.pyplot needs.

_real_getppid = os.getppid # Substitute a PID that's always valid (our own) for the duration of this import only.
os.getppid = os.getpid
from feeed.feature_extractor import extract_features
from gedi.utils.io_helpers import compute_feature_similarities, get_generated_file_name
from gedi.utils.target_validation import feature_bounds, fit_window, is_numeric, validate_target
os.getppid = _real_getppid
