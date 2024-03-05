# # Loading necessary Libraries

import pandas as pd
import numpy as np
import time
from shapely.geometry import Polygon, Point
from scipy.spatial import ConvexHull, QhullError
from scipy.stats import skew, kurtosis
from features_slow_lane import slow_lane_df
from features_fast_lane import fast_lane_df

from dataframe_creation import *


def centre_lane_features():
    """
    Extracts and processes features related to the center lane from the provided data.

    Returns:
        pandas.DataFrame: A DataFrame containing the extracted features for the center lane.
    """
    df = dataframe_creation()

    # Filter columns required for calculations
    required_columns = ["FrameNum", "X", "Y", "Velocity", "#DetOBj", "Time"]
    subset_df = df[required_columns]

    # Remove rows where 'v' values lie between -1.6 and +1.6
    subset_df = subset_df[
        ~subset_df["Velocity"].between(-1.6, 1.6) & (subset_df["Velocity"] <= 0)
    ]
    subset_df["Velocity"] = subset_df["Velocity"].abs()

    vertices = [(5.5, 14.5), (-4.1, 14.5), (-7.1, 3.75), (1.5, 3.75)]

    # Create a Polygon object using the vertices
    region_polygon = Polygon(vertices)

    # Function to check if a point falls within the defined region
    def point_in_region(row):
        point = Point(row["X"], row["Y"])
        return region_polygon.contains(point)

    # Filter rows based on whether points fall within the defined region
    subset_df = subset_df[subset_df.apply(point_in_region, axis=1)]

    # Group by 'FrameNum' and calculate required statistics
    grouped_df = (
        subset_df.groupby("FrameNum")
        .agg(
            mean_x=("X", "mean"),
            mean_y=("Y", "mean"),
            mean_v=("Velocity", "mean"),
            max_x=("X", "max"),
            max_y=("Y", "max"),
            max_v=("Velocity", "max"),
            min_x=("X", "min"),
            min_y=("Y", "min"),
            min_v=("Velocity", "min"),
            max_detected=("#DetOBj", "max"),
        )
        .reset_index()
    )

    subset_df["detected objects"] = subset_df["FrameNum"].map(
        subset_df["FrameNum"].value_counts()
    )

    final_grouped_df = grouped_df

    # Group subset_df by 'FrameNum' and get unique values of 'detected objects'
    unique_detected_objects = (
        subset_df.groupby("FrameNum")["detected objects"].unique().reset_index()
    )

    # Merge the unique_detected_objects with final_grouped_df on 'FrameNum'
    final_grouped_df = final_grouped_df.merge(
        unique_detected_objects, on="FrameNum", how="left"
    )

    # Rename the column containing unique detected objects
    final_grouped_df.rename(
        columns={"detected objects": "Unique_detected_objects"}, inplace=True
    )

    ## Defining the region for Centre Lane

    vertices_centre_lane = vertices_centre_lane = [
        (-4.05, 4.10),
        (-0.65, 14.59),
        (0.08, 14.46),
        (-3.22, 3.56),
    ]

    # Create a Polygon object using the vertices
    region_polygon_centre_lane = Polygon(vertices_centre_lane)

    # Function to check if a point falls within the defined region
    def point_in_region_centre_lane(row):
        point = Point(row["X"], row["Y"])
        return region_polygon_centre_lane.contains(point)

    # Filter rows based on whether points fall within the defined region
    centre_lane_df = subset_df[subset_df.apply(point_in_region_centre_lane, axis=1)]

    fast_lane_df_2 = fast_lane_df()
    slow_lane_df_2 = slow_lane_df()

    # Extract unique seconds from fast_lane_df_2 and slow_lane_df_2
    fast_lane_seconds = set(fast_lane_df_2["Time"].str[:9].unique())
    slow_lane_seconds = set(slow_lane_df_2["Time"].str[:9].unique())

    # Combine unique seconds from both dataframes
    unique_seconds = fast_lane_seconds.union(slow_lane_seconds)

    # Remove rows in centre_lane_df_2 where time starts with unique seconds
    centre_lane_df_2 = centre_lane_df[
        ~centre_lane_df["Time"].str[:9].isin(unique_seconds)
    ]

    ## Addding Area and other features

    subset_df = centre_lane_df_2

    def calculate_convex_hull_area_diagonal(group):
        try:
            if len(group) < 3:
                return pd.Series([0, 0], index=["ConvexHullArea", "ConvexHullDiagonal"])

            points = np.array(group[["X", "Y"]])
            hull = ConvexHull(points, qhull_options="QJ")  # Modification here

            convex_hull_area = hull.area
            convex_hull_diagonal = np.max(hull.max_bound - hull.min_bound)

            return pd.Series(
                [convex_hull_area, convex_hull_diagonal],
                index=["ConvexHullArea", "ConvexHullDiagonal"],
            )

        except Exception as e:
            print(f"Error occurred: {e}")
            return pd.Series([0, 0], index=["ConvexHullArea", "ConvexHullDiagonal"])

    result_convex_hull = (
        subset_df.groupby("FrameNum")
        .apply(calculate_convex_hull_area_diagonal)
        .reset_index()
    )

    # Merge the results back to the original DataFrame
    final_subset_df = pd.merge(subset_df, result_convex_hull, on="FrameNum")

    # Define the 'FrameNum' column as a categorical column to maintain the original order
    final_subset_df["FrameNum"] = pd.Categorical(
        final_subset_df["FrameNum"],
        categories=final_subset_df["FrameNum"].unique(),
        ordered=True,
    )

    # Perform the grouping and aggregation while preserving the original order
    grouped_slow_df = (
        final_subset_df.groupby("FrameNum")
        .agg(
            mean_x=("X", "mean"),
            mean_y=("Y", "mean"),
            mean_v=("Velocity", "mean"),
            mean_area=("ConvexHullArea", "mean"),
            mean_diag_len=("ConvexHullDiagonal", "mean"),
            min_x=("X", "min"),
            max_x=("X", "max"),
            min_y=("Y", "min"),
            max_y=("Y", "max"),
            min_v=("Velocity", "min"),
            max_v=("Velocity", "max"),
            max_detected_slow=("#DetOBj", "max"),
            max_area=("ConvexHullArea", "max"),
            max_diag_len=("ConvexHullDiagonal", "max"),
            time=("Time", "max"),
            mean_detected_objects_slow=("detected objects", "mean"),
        )
        .reset_index()
    )

    grouped_slow_df = grouped_slow_df.rename(columns={"mean_v": "slow_mean_v"})
    grouped_slow_df = grouped_slow_df.rename(columns={"max_v": "slow_max_v"})

    # Calculate the absolute difference between max_x and min_x
    grouped_slow_df["width"] = grouped_slow_df["max_x"] - grouped_slow_df["min_x"]
    grouped_slow_df["width"] = grouped_slow_df["width"].abs()

    # Calculate the absolute difference between max_y and min_y
    grouped_slow_df["length"] = grouped_slow_df["max_y"] - grouped_slow_df["min_y"]
    grouped_slow_df["length"] = grouped_slow_df["length"].abs()

    df = grouped_slow_df

    # Remove leading spaces from 'time' column
    df["time"] = df["time"].str.strip()

    # Convert time column to datetime
    df["time"] = pd.to_datetime(df["time"], format="%H:%M:%S:%f")

    # Calculate the difference in time and min_y
    df["FrameNum_diff"] = df["FrameNum"].diff().fillna(0).astype(int).abs()

    # Define the condition for grouping
    condition = df["FrameNum_diff"] > 7

    # Create initial Group_id based on condition starting from 1
    df["Group_id"] = condition.cumsum() + 1

    # Resetting Group_id based on the condition within each group formed by the previous condition
    prev_group_id = None
    for i in range(1, len(df)):
        if df.loc[i, "FrameNum_diff"] > 7:
            if prev_group_id is None or df.loc[i, "Group_id"] != prev_group_id:
                prev_group_id = df.loc[i, "Group_id"]
                df.loc[i:, "Group_id"] = df.loc[i:, "Group_id"].apply(lambda x: x + 1)
            else:
                df.loc[i, "Group_id"] = prev_group_id

    # Increment Group_id by 1
    df["Group_id_dense"] = df["Group_id"].rank(method="dense").astype(int)

    ## Adding other features based on Group_id_dense

    analysis_df = df

    # Renaming the columns
    analysis_df.columns = analysis_df.columns.str.replace(
        "max_detected_slow", "raw_slow_det_obj"
    )
    analysis_df.columns = analysis_df.columns.str.replace(
        "mean_detected_objects_slow", "unique_slow_det_obj"
    )

    # Calculate mean and maximum of 'unique_slow_det_obj' for each 'Group_id'
    group_stats = (
        analysis_df.groupby("Group_id_dense")[
            "unique_slow_det_obj",
            "mean_area",
            "mean_diag_len",
            "max_area",
            "max_diag_len",
            "width",
            "length",
        ]
        .agg(["mean", "max"])
        .reset_index()
    )
    group_stats.columns = [
        "Group_id_dense",
        "mean_unique_slow_det_obj",
        "max_unique_slow_det_obj",
        "mean_area_mean",
        "mean_area_max",
        "mean_diag_len_mean",
        "mean_diag_len_max",
        "max_area_mean",
        "max_area_max",
        "max_diag_len_mean",
        "max_diag_len_max",
        "width_mean",
        "width_max",
        "length_mean",
        "length_max",
    ]

    # Merge the calculated statistics back to the original DataFrame based on 'Group_id'
    analysis_df = pd.merge(analysis_df, group_stats, on="Group_id_dense", how="left")

    # Reorder columns to place mean and max columns next to 'unique_slow_det_obj'
    mean_column_index = analysis_df.columns.get_loc("unique_slow_det_obj") + 1
    analysis_df = pd.concat(
        [
            analysis_df.iloc[:, :mean_column_index],
            analysis_df.iloc[:, -2:],
            analysis_df.iloc[:, mean_column_index:-2],
        ],
        axis=1,
    )

    # Create the 'Total_detected' column
    analysis_df["Total_detected"] = analysis_df.groupby("Group_id_dense")[
        "unique_slow_det_obj"
    ].transform("sum")

    # Create the 'Unique_detected' column
    analysis_df["Unique_detected"] = analysis_df.groupby("Group_id_dense")[
        "unique_slow_det_obj"
    ].transform("count")

    analysis_df["max_unique_std"] = analysis_df.groupby("Group_id_dense")[
        "unique_slow_det_obj"
    ].transform("std")
    analysis_df["max_unique_skew"] = analysis_df.groupby("Group_id_dense")[
        "unique_slow_det_obj"
    ].transform(skew)
    analysis_df["max_unique_kurtosis"] = analysis_df.groupby("Group_id_dense")[
        "unique_slow_det_obj"
    ].transform(kurtosis)

    analysis_df["max_area_std"] = analysis_df.groupby("Group_id_dense")[
        "max_area"
    ].transform("std")
    analysis_df["max_area_skew"] = analysis_df.groupby("Group_id_dense")[
        "max_area"
    ].transform(skew)
    analysis_df["max_area_kurtosis"] = analysis_df.groupby("Group_id_dense")[
        "max_area"
    ].transform(kurtosis)

    analysis_df["max_diag_len_std"] = analysis_df.groupby("Group_id_dense")[
        "max_diag_len"
    ].transform("std")
    analysis_df["max_diag_len_skew"] = analysis_df.groupby("Group_id_dense")[
        "max_diag_len"
    ].transform(skew)
    analysis_df["max_diag_len_kurtosis"] = analysis_df.groupby("Group_id_dense")[
        "max_diag_len"
    ].transform(kurtosis)

    analysis_df["width_std"] = analysis_df.groupby("Group_id_dense")["width"].transform(
        "std"
    )
    analysis_df["width_skew"] = analysis_df.groupby("Group_id_dense")[
        "width"
    ].transform(skew)
    analysis_df["width_kurtosis"] = analysis_df.groupby("Group_id_dense")[
        "width"
    ].transform(kurtosis)

    analysis_df["length_std"] = analysis_df.groupby("Group_id_dense")[
        "length"
    ].transform("std")
    analysis_df["length_skew"] = analysis_df.groupby("Group_id_dense")[
        "length"
    ].transform(skew)
    analysis_df["length_kurtosis"] = analysis_df.groupby("Group_id_dense")[
        "length"
    ].transform(kurtosis)

    analysis_df["max_area_max / Unique detected"] = (
        analysis_df["max_area_max"] / analysis_df["Unique_detected"]
    ) * 100
    analysis_df["max_area_max / max_unique_slow_det_obj"] = (
        analysis_df["max_area_max"] / analysis_df["max_unique_slow_det_obj"]
    ) * 100
    analysis_df["max_area_max / Total detected"] = (
        analysis_df["max_area_max"] / analysis_df["Total_detected"]
    ) * 100
    analysis_df["Total_detected/Unique detected"] = (
        analysis_df["Total_detected"] / analysis_df["Unique_detected"]
    )
    analysis_df["length_max/width_max"] = (
        analysis_df["length_max"] / analysis_df["width_max"]
    )
    analysis_df["length_mean/width_mean"] = (
        analysis_df["length_mean"] / analysis_df["width_mean"]
    )
    analysis_df["length_max x width_max"] = (
        analysis_df["length_max"] * analysis_df["width_max"]
    )
    analysis_df["length_mean x width_mean"] = (
        analysis_df["length_mean"] * analysis_df["width_mean"]
    )
    analysis_df["Max_triangular_area"] = (
        0.5 * analysis_df["length_max"] * analysis_df["width_max"]
    )
    analysis_df["Mean_triangular_area"] = (
        0.5 * analysis_df["length_mean"] * analysis_df["width_mean"]
    )

    analysis_df = analysis_df.fillna(0)

    # Extracting specific columns
    extracted_columns = analysis_df[
        [
            "time",
            "Group_id_dense",
            "unique_slow_det_obj",
            "mean_unique_slow_det_obj",
            "max_unique_slow_det_obj",
            "max_area_max",
            "max_diag_len_max",
            "max_area_mean",
            "max_diag_len_mean",
            "width_max",
            "length_max",
            "width_mean",
            "length_mean",
            "Total_detected",
            "Unique_detected",
            "max_unique_std",
            "max_unique_skew",
            "max_unique_kurtosis",
            "max_area_std",
            "max_area_skew",
            "max_area_kurtosis",
            "max_diag_len_std",
            "max_diag_len_skew",
            "max_diag_len_kurtosis",
            "width_std",
            "width_skew",
            "width_kurtosis",
            "length_std",
            "length_skew",
            "length_kurtosis",
            "max_area_max / Unique detected",
            "max_area_max / max_unique_slow_det_obj",
            "max_area_max / Total detected",
            "Total_detected/Unique detected",
            "length_max/width_max",
            "length_mean/width_mean",
            "length_max x width_max",
            "length_mean x width_mean",
            "Max_triangular_area",
            "Mean_triangular_area",
        ]
    ]

    # Creating a new dataframe with the extracted columns
    extracted_df = pd.DataFrame(extracted_columns)

    # Extracting specific columns
    extracted_columns_2 = extracted_df[
        [
            "time",
            "Group_id_dense",
            "unique_slow_det_obj",
            "mean_unique_slow_det_obj",
            "max_unique_slow_det_obj",
            "max_area_max",
            "max_diag_len_max",
            "max_area_mean",
            "max_diag_len_mean",
            "width_max",
            "length_max",
            "width_mean",
            "length_mean",
            "Total_detected",
            "Unique_detected",
            "max_unique_std",
            "max_unique_skew",
            "max_unique_kurtosis",
            "max_area_std",
            "max_area_skew",
            "max_area_kurtosis",
            "max_diag_len_std",
            "max_diag_len_skew",
            "max_diag_len_kurtosis",
            "width_std",
            "width_skew",
            "width_kurtosis",
            "length_std",
            "length_skew",
            "length_kurtosis",
            "max_area_max / Unique detected",
            "max_area_max / max_unique_slow_det_obj",
            "max_area_max / Total detected",
            "Total_detected/Unique detected",
            "length_max/width_max",
            "length_mean/width_mean",
            "length_max x width_max",
            "length_mean x width_mean",
            "Max_triangular_area",
            "Mean_triangular_area",
        ]
    ]

    # Creating a new dataframe with the extracted columns
    extracted_df_2 = pd.DataFrame(extracted_columns_2)

    classify_df = extracted_df_2.drop_duplicates(
        subset=[
            "Group_id_dense",
            "mean_unique_slow_det_obj",
            "max_unique_slow_det_obj",
            "max_area_max",
            "max_diag_len_max",
            "max_area_mean",
            "max_diag_len_mean",
            "width_max",
            "length_max",
            "width_mean",
            "length_mean",
            "Total_detected",
            "Unique_detected",
            "max_unique_std",
            "max_unique_skew",
            "max_unique_kurtosis",
            "max_area_std",
            "max_area_skew",
            "max_area_kurtosis",
            "max_diag_len_std",
            "max_diag_len_skew",
            "max_diag_len_kurtosis",
            "width_std",
            "width_skew",
            "width_kurtosis",
            "length_std",
            "length_skew",
            "length_kurtosis",
            "max_area_max / Unique detected",
            "max_area_max / max_unique_slow_det_obj",
            "max_area_max / Total detected",
            "Total_detected/Unique detected",
            "length_max/width_max",
            "length_mean/width_mean",
            "length_max x width_max",
            "length_mean x width_mean",
            "Max_triangular_area",
            "Mean_triangular_area",
        ]
    )

    classify_df.drop("unique_slow_det_obj", axis=1, inplace=True)

    classify_df_ml = classify_df

    centre_lane_input_df = classify_df_ml

    return centre_lane_input_df
