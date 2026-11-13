import math

from igedi.app import feature_bounds

FEATURE_GROUPS = [  # (group, name prefixes); first match wins, so start/end come before activities
    ("Size", ("n_traces", "n_variants", "n_unique_activities")),
    ("Trace length", ("trace_len_",)),
    ("Variants", ("ratio_", "mean_variant_occurrence", "std_variant_occurrence",
                  "skewness_variant_occurrence", "kurtosis_variant_occurrence")),
    ("Start activities", ("n_unique_start_activities", "start_activities_")),
    ("End activities", ("n_unique_end_activities", "end_activities_")),
    ("Activities", ("activities_",)),
    ("Entropy", ("eventropy_", "epa_")),
]

STAT_NAMES = {
    "min": "minimum", "max": "maximum", "mean": "mean", "median": "median", "mode": "mode",
    "std": "standard deviation", "variance": "variance", "q1": "first quartile", "q3": "third quartile",
    "iqr": "interquartile range", "geometric_mean": "geometric mean", "geometric_std": "geometric standard deviation",
    "harmonic_mean": "harmonic mean", "skewness": "skewness", "kurtosis": "kurtosis",
    "coefficient_variation": "coefficient of variation", "entropy": "entropy",
    "skewness_hist": "skewness of the histogram", "kurtosis_hist": "kurtosis of the histogram",
}

FEATURE_DESCRIPTIONS = {
    "n_traces": "Number of traces (cases) in the log.",
    "n_variants": "Number of distinct trace variants.",
    "n_unique_activities": "Number of distinct activity labels.",
    "n_unique_start_activities": "Number of distinct activities that start a trace.",
    "n_unique_end_activities": "Number of distinct activities that end a trace.",
    "ratio_variants_per_number_of_traces": "Distinct variants divided by traces (1 = every trace is unique).",
    "ratio_most_common_variant": "Share of traces that follow the most frequent variant.",
    "mean_variant_occurrence": "Average number of traces per variant.",
    "std_variant_occurrence": "Standard deviation of the number of traces per variant.",
    "skewness_variant_occurrence": "Skewness of the number of traces per variant.",
    "kurtosis_variant_occurrence": "Kurtosis of the number of traces per variant.",
    "epa_variant_entropy": "Variant entropy of the log's prefix automaton (Back et al.).",
    "epa_normalized_variant_entropy": "Variant entropy (prefix automaton), normalized to 0–1: higher = more diverse behavior.",
    "epa_sequence_entropy": "Sequence entropy of the log's prefix automaton (Back et al.).",
    "epa_normalized_sequence_entropy": "Sequence entropy (prefix automaton), normalized to 0–1.",
    "epa_sequence_entropy_linear_forgetting": "Sequence entropy that discounts older behavior linearly.",
    "epa_normalized_sequence_entropy_linear_forgetting": "Normalized sequence entropy with linear forgetting (0–1).",
    "epa_sequence_entropy_exponential_forgetting": "Sequence entropy that discounts older behavior exponentially.",
    "epa_normalized_sequence_entropy_exponential_forgetting": "Normalized sequence entropy with exponential forgetting (0–1).",
}

def feature_group(feature):
    for group, prefixes in FEATURE_GROUPS:
        if feature.startswith(prefixes):
            return group
    return "Other"

def feature_description(feature):
    if feature in FEATURE_DESCRIPTIONS:
        return FEATURE_DESCRIPTIONS[feature]
    for prefix, subject in (("trace_len_", "the number of events per trace"),
                            ("start_activities_", "how often each start activity occurs"),
                            ("end_activities_", "how often each end activity occurs"),
                            ("activities_", "how often each activity occurs")):
        if feature.startswith(prefix):
            stat = feature[len(prefix):]
            if stat.startswith("hist"):
                return f"Share of traces in bin {stat[4:]} of the trace length histogram."
            return f"{STAT_NAMES.get(stat, stat).capitalize()} of {subject}."
    if feature.startswith("ratio_top_"):
        return f"Share of traces covered by the top {feature.split('_')[2]}% most frequent variants."
    if feature.startswith("eventropy_"):
        return "Entropy estimate of the log's behavior (eventropy, Back et al.)."
    return None

def range_hint(feature):
    low, high = feature_bounds(feature)
    if math.isinf(low) and math.isinf(high):
        return ""
    return f" ({low:g}–{high:g})" if not math.isinf(high) else f" (≥ {low:g})"
