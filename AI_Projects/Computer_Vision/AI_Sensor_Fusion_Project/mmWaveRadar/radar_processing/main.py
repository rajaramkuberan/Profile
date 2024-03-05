import csv
import logging
import os
import time
from datetime import datetime
import warnings

import numpy as np
import pandas as pd

from dataframe_creation import *
from fast_lane_model import *
from slow_lane_model import *
from centre_lane_model import *
from bike_lane_model import *


current_date = time.strftime("%Y-%m-%d")
log_dir = f"D:/radar_processing/radar_processing/{current_date}"
os.makedirs(log_dir, exist_ok=True)


def configure_logging(log_file):
    """
    Configure logging settings.

    Args:
        log_file (str): The name of the log file.

    Returns:
        None
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - [%(levelname)s]- %(message)s",
        filename=os.path.join(log_dir, log_file),
    )


def main():
    """
    Process data and perform analysis.

    This function performs data processing and analysis. It checks if the dataframe
    is successfully created and then calculates counts for different vehicle types
    in different lanes. It logs the counts of each vehicle type if the dataframe is
    created successfully, otherwise logs a message indicating no files to process.

    Returns:
        None
    """
    df_check = dataframe_creation()
    if df_check is not None:
        class_counts_df_slow = slow_lane_count()
        class_counts_df_fast = fast_lane_count()
        class_counts_df_bike = bike_lane_count()
        class_counts_df_car = centre_lane_count()

        # Combine DataFrames into one
        combined_df = pd.concat(
            [
                class_counts_df_slow,
                class_counts_df_fast,
                class_counts_df_bike,
                class_counts_df_car,
            ],
            ignore_index=True,
        )

        total_counts_df = combined_df.groupby("Class")["Count"].sum().reset_index()

        total_dict = dict(zip(total_counts_df["Class"], total_counts_df["Count"]))

        Bike_count = 0
        Car_count = 0
        Bus_count = 0
        MAV2Axle_count = 0
        MAV4Axle_count = 0
        MiniLCV_count = 0
        LCV_count = 0
        MAV3Axle_count = 0

        for i, j in total_dict.items():
            if i == "Bike":
                Bike_count = j
            elif i == "Bus":
                Bus_count = j
            elif i == "Car":
                Car_count = j
            elif i == "MiniLCV":
                MiniLCV_count = j
            elif i == "LCV":
                LCV_count = j
            elif i == "MAV2Axle":
                MAV2Axle_count = j
            elif i == "MAV4+Axle":
                MAV4Axle_count = j
            elif i == "MAV3Axle":
                MAV3Axle_count = j

        total_vehicle_count = (
            Bike_count
            + Car_count
            + Bus_count
            + MiniLCV_count
            + LCV_count
            + MAV2Axle_count
            + MAV3Axle_count
            + MAV4Axle_count
        )

        total_vehicle_count = f"total_vehicle_count: {total_vehicle_count}"
        logging.info(total_vehicle_count)
        Bike_count = f"Bike_count: {Bike_count}"
        logging.info(Bike_count)
        Car_count = f"Car_count: {Car_count}"
        logging.info(Car_count)
        Bus_count = f"Bus_count: {Bus_count}"
        logging.info(Bus_count)
        MiniLCV_count = f"MiniLCV_count: {MiniLCV_count}"
        logging.info(MiniLCV_count)
        LCV_count = f"LCV_count: {LCV_count}"
        logging.info(LCV_count)
        MAV2Axle_count = f"MAV2Axle_count: {MAV2Axle_count}"
        logging.info(MAV2Axle_count)
        MAV3Axle_count = f"MAV3Axle_count: {MAV3Axle_count}"
        logging.info(MAV3Axle_count)
        MAV4Axle_count = f"MAV4Axle_count: {MAV4Axle_count}"
        logging.info(MAV4Axle_count)

    else:
        msg = "No files to process"
        logging.info(msg)


if __name__ == "__main__":
    current_date = datetime.now()
    filename = f'script_log_{current_date.strftime("%Y%m%d%H%M%S")}.log'
    configure_logging(filename)
    main()
