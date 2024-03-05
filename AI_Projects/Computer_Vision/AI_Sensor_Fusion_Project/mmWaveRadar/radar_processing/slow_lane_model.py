import numpy as np
import pandas as pd
import pickle
import sklearn

import pycaret
from pycaret.classification import *

import xgboost

from features_centre_lane import *
from features_slow_lane import *



def slow_lane_count():
    """
    Count the number of vehicles in the slow lane.

    This function loads the Slow Lane Model using pickle and extracts features for the slow lane.
    It then predicts the class labels using the loaded model and renames the prediction labels to 'Class'.
    After merging the predictions with the original data, it counts the occurrences of each unique class
    and calculates the total number of unique classes and occurrences of all classes.
    Finally, it returns a DataFrame with class and count columns.

    Returns:
        pandas.DataFrame: DataFrame with class and count of vehicles in the slow lane.
    """
    # Load the Slow Lane Model using pickle
    slow_model = load_model("Slow_Lane_Final_Model")

    sl = slow_lane_features()

    true_label_gen_df_slow = sl

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
    selected_columns_df = true_label_gen_df_slow[columns_to_extract]

    # Replace infinity values with NaN
    selected_columns_df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Drop rows with NaN values
    selected_columns_df = selected_columns_df.dropna()

    selected_columns_df.to_csv("Before_time_adding_slow.csv", index=True)

    selected_columns_df.drop("time", axis=1, inplace=True)

    slow_model_predictions = predict_model(slow_model, data=selected_columns_df)

    ## Renaming Prediction label to Class

    slow_model_predictions.rename(columns={"prediction_label": "Class"}, inplace=True)

    before_time_adding_slow = pd.read_csv("Before_time_adding_slow.csv")

    merged_data_slow = pd.merge(
        slow_model_predictions,
        before_time_adding_slow[["time"]],
        left_index=True,
        right_index=True,
        how="left",
    )

    # merged_data_slow.to_csv('Final_slow_model_Predictions.csv', index = False)

    slow_lane_pred = merged_data_slow

    # Count the occurrences of each unique class in the 'prediction_label' column
    class_counts_slow = slow_lane_pred["Class"].value_counts()

    # Calculate the total number of unique classes
    total_classes_slow = len(class_counts_slow)

    # Calculate the total number of occurrences of all classes
    total_occurrences_slow = class_counts_slow.sum()

    # Create a new DataFrame with class and count columns
    class_counts_df_slow = pd.DataFrame(
        {"Class": class_counts_slow.index, "Count": class_counts_slow.values}
    )

    return class_counts_df_slow
