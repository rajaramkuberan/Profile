import warnings
import pickle

import numpy as np
import pandas as pd

import xgboost
import lightgbm

import pycaret
from pycaret.classification import *

from features_fast_lane import *



def fast_lane_count():
    """
    Perform predictions for the fast lane.

    This function loads the pre-trained fast lane model, extracts features,
    and makes predictions. It then renames the prediction label to 'Class'
    and merges the predictions with the original data. Counts the occurrences
    of each unique class and creates a DataFrame with class and count columns.

    Returns:
        pandas.DataFrame: DataFrame with class and count columns.
    """
    # load the model
    fast_model = load_model("Fast_Lane_Final_Model")

    fst = fast_lane_features()

    true_label_gen_df_fast = fst

    columns_to_extract = [
        "max_unique_slow_det_obj",
        "max_area_max",
        "max_diag_len_max",
        "width_max",
        "length_max",
        "Total_detected",
        "Unique_detected",
        "max_unique_std",
        "max_area_std",
        "max_diag_len_std",
        "length_std",
        "max_area_max / max_unique_slow_det_obj",
        "length_max/width_max",
        "length_max x width_max",
        "time",
    ]

    # Extracting the specified columns
    selected_columns_df = true_label_gen_df_fast[columns_to_extract]

    # Replace infinity values with NaN
    selected_columns_df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Drop rows with NaN values
    selected_columns_df = selected_columns_df.dropna()

    selected_columns_df.to_csv("Before_time_adding_fast.csv", index=True)

    selected_columns_df.drop("time", axis=1, inplace=True)

    fast_model_predictions = predict_model(fast_model, data=selected_columns_df)

    ## Renaming Prediction label to Class

    fast_model_predictions.rename(columns={"prediction_label": "Class"}, inplace=True)

    before_time_adding_fast = pd.read_csv("Before_time_adding_fast.csv")

    merged_data_fast = pd.merge(
        fast_model_predictions,
        before_time_adding_fast[["time"]],
        left_index=True,
        right_index=True,
        how="left",
    )

    # merged_data_fast.to_excel('Final_fast_model_Predictions.xlsx', index = False)

    fast_lane_pred = merged_data_fast

    # Count the occurrences of each unique class in the 'prediction_label' column
    class_counts_fast = fast_lane_pred["Class"].value_counts()

    # Calculate the total number of unique classes
    total_classes_fast = len(class_counts_fast)

    # Calculate the total number of occurrences of all classes
    total_occurrences_fast = class_counts_fast.sum()

    # Create a new DataFrame with class and count columns
    class_counts_df_fast = pd.DataFrame(
        {"Class": class_counts_fast.index, "Count": class_counts_fast.values}
    )

    return class_counts_df_fast
