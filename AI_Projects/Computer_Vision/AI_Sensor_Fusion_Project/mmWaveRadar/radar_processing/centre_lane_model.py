import pandas as pd
import numpy as np

from features_centre_lane import *


def centre_lane_count():
    """
    Count the number of cars in the centre lane.

    This function extracts features for the centre lane, which represents cars.
    It then counts the number of unique values in the 'Group_id_dense' column,
    which corresponds to the number of cars detected in the centre lane.

    Returns:
        pandas.DataFrame: DataFrame with class 'Car' and count of cars.
    """
    cntr = centre_lane_features()

    centre_lane_predictions = cntr

    # Count the number of unique values in the 'Group_id_dense' column
    unique_values_count_car = centre_lane_predictions["Group_id_dense"].nunique()

    # Create a new DataFrame with class and count columns
    class_counts_df_car = pd.DataFrame(
        {"Class": ["Car"], "Count": unique_values_count_car}
    )

    return class_counts_df_car
