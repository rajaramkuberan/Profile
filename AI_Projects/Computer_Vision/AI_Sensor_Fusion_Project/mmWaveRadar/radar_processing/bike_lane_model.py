import pandas as pd
import numpy as np

from features_bike_lane import *


def bike_lane_count():
    """
    Count the number of bikes in the bike lane.

    This function extracts features for the bike lane and counts the number of unique values
    in the 'Group_id_dense' column, which corresponds to the number of bikes detected in the bike lane.

    Returns:
        pandas.DataFrame: DataFrame with class 'Bike' and count of bikes.
    """
    bke = bike_lane_features()

    bike_lane_predictions = bke

    # Count the number of unique values in the 'Group_id_dense' column
    unique_values_count_bike = bike_lane_predictions["Group_id_dense"].nunique()

    # Create a new DataFrame with class and count columns
    class_counts_df_bike = pd.DataFrame(
        {"Class": ["Bike"], "Count": unique_values_count_bike}
    )

    return class_counts_df_bike
