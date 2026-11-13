from itertools import product as cproduct
import math
import numpy as np
import streamlit as st

from igedi.app import feature_bounds, fit_window
from igedi.app.state import stop_with_error
from igedi.app.widgets import input_multicolumn

def split_list(input_list, n):
    # Calculate the size of each chunk
    k, m = divmod(len(input_list), n)
    # Use list comprehension to create n sublists
    return [input_list[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]

def default_values(feature, value, n):
    low, high = fit_window(feature, value, 0.2 * value * (n - 1))
    return [round(float(v), 2) for v in np.linspace(low, high, n)]

def default_range(feature, value):
    bound_low, bound_high = feature_bounds(feature)
    width = value if value > 0 else (bound_high - bound_low if not math.isinf(bound_high - bound_low) else 1.0)
    low, high = fit_window(feature, value, width)
    step = math.floor((high - low) / 3 * 100) / 100 or round((high - low) / 3, 4)
    return [round(low, 2), round(high, 2), step]

def to_float(value, label):
    try:
        return float(value)
    except (TypeError, ValueError):
        stop_with_error(f"'{value}' is not a number (field: {label}).")

def arange_values(selected_min, selected_max, step_value, label):
    selected_min, selected_max, step_value = (to_float(v, label) for v in (selected_min, selected_max, step_value))
    if step_value <= 0:
        stop_with_error(f"Step must be greater than 0 (field: {label}).")
    if selected_max < selected_min:
        stop_with_error(f"Max must be at least min (field: {label}).")
    return np.around(np.arange(selected_min, selected_max + step_value / 2, step_value), 2)

def create_objectives_grid(df, objectives, n_para_obj=2, method="combinatorial"):
        # One list of candidate values per objective; targets are their cartesian product.
        value_lists = []
        if "combinatorial" in method:
            # df: one row per feature, columns are the values (e.g. min/max or "value i").
            columns = sorted(objectives)
            objectives = df.index.to_list()
            for feature in objectives:
                value_lists.append(np.around([to_float(v, feature) for v in df.loc[feature, columns]], 2))

        elif method=="range-from-csv":
            for objective in objectives:
                min_col, max_col, step_col = st.columns(3)
                with min_col:
                    selcted_min = st.slider(objective+': min', min_value=float(df[objective].min()), max_value=float(df[objective].max()), value=df[objective].quantile(0.1), step=0.1, key=objective+"min")
                with max_col:
                    selcted_max = st.slider('max', min_value=selcted_min, max_value=float(df[objective].max()), value=df[objective].quantile(0.9), step=0.1, key=objective+"max")
                with step_col:
                    step_value = st.slider('step', min_value=float(df[objective].min()), max_value=float(df[objective].quantile(0.9)), value=df[objective].median()/(df[objective].min()+0.0001), step=0.01, key=objective+"step")
                value_lists.append(arange_values(selcted_min, selcted_max, step_value, objective))
        else :#method=="range-manual":
            if objectives != None:
                cross_labels =  [feature[0]+': '+feature[1] for feature in list(cproduct(objectives,['min', 'max', 'step']))]
                cross_values = [value for objective in objectives for value in default_range(objective, df[objective])]
                ranges = zip(objectives, split_list(list(input_multicolumn(cross_labels, cross_values, n_cols=3)), n_para_obj))
                for objective, range_value in ranges:
                    selcted_min, selcted_max, step_value = range_value
                    value_lists.append(arange_values(selcted_min, selcted_max, step_value, objective))

        if not objectives or not value_lists:
            stop_with_error("Please select at least one feature above.")
        cartesian_product = list(cproduct(*value_lists))
        return [{key: float(value[idx]) for idx, key in enumerate(objectives)} for value in cartesian_product]
