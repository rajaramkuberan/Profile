import os
import shutil
import time
import pandas as pd
import glob


def dataframe_creation():

    try:
        df1 = pd.read_csv(r"data_20240224-113829.csv")

        # Read the second CSV file
        df2 = pd.read_csv(r"data_20240224-114329.csv")  # Replace "second_file.csv" with the actual filename of your second CSV file

        # Concatenate the two dataframes
        #df = pd.concat([df1, df2], ignore_index=True)
        combined_df = pd.concat([df1, df2], ignore_index=True)

        # Display or use the combined DataFrame
                    
        return combined_df
        
    except Exception as e:
        print(f"An error occurred: {e}")

