import math
import numbers

def feature_bounds(feature):
    # Ratios and normalized entropies are bounded to [0, 1] by definition.
    if feature.startswith("ratio_") or feature.startswith("epa_normalized_"):
        return 0.0, 1.0
    if "skewness" in feature or "kurtosis" in feature or feature.startswith("eventropy_k_block_diff"):
        return -math.inf, math.inf
    return 0.0, math.inf

def is_unit_interval_feature(feature):
    return feature_bounds(feature) == (0.0, 1.0)

def fit_window(feature, start, width):
    # Shifts [start, start + width] into the feature's bounds, shrinking it only if it doesn't fit
    low, high = feature_bounds(feature)
    width = min(width, high - low)
    start = min(max(start, low), high - width)
    return start, start + width

def is_numeric(value):
    return isinstance(value, numbers.Real) and not isinstance(value, bool) and not math.isnan(value) # numbers.Real also covers numpy scalars such as np.int64 from feature extraction


def max_config_value(config_value):
    if isinstance(config_value, (list, tuple)):
        return max(config_value)
    return config_value

def validate_target(target, config_space=None):
    """
    Checks a single target against what the features can actually take.
    @return: (errors, warnings), lists of messages.
        errors: the target is impossible and should not be generated.
        warnings: the target is reachable in principle, but not with the given config_space.
    """
    errors, warnings = [], []
    numeric_target = {k: float(v) for k, v in target.items() if k != "log" and is_numeric(v)}

    for feature, value in numeric_target.items():
        low, high = feature_bounds(feature)
        if not low <= value <= high:
            errors.append(f"{feature}={value} is outside its possible range [{low:g}, {high:g}].")

    if "n_variants" in numeric_target and "n_traces" in numeric_target \
            and numeric_target["n_variants"] > numeric_target["n_traces"]:
        errors.append(f"n_variants={numeric_target['n_variants']} cannot exceed n_traces={numeric_target['n_traces']}.")

    if config_space is not None and config_space.get("num_traces") is not None:
        max_traces = max_config_value(config_space["num_traces"])
        for feature in ("n_traces", "n_variants"):
            if feature in numeric_target and numeric_target[feature] > max_traces:
                warnings.append(f"{feature}={numeric_target[feature]} is unreachable with config_space num_traces <= {max_traces}.")

    return errors, warnings

def feature_similarity(feature, target_value, generated_value):
    """
    Similarity in [0, 1] between a target and a generated feature value.
    Bounded features use their absolute difference, unbounded features the symmetric relative difference.
    """
    if is_unit_interval_feature(feature):
        return 1 - abs(target_value - generated_value)
    denominator = abs(target_value) + abs(generated_value)
    if denominator == 0:
        return 1.0
    return 1 - abs(target_value - generated_value) / denominator
