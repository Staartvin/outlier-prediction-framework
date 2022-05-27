import contextlib
import datetime
import glob

import os
import sys
import time

from typing import List, Tuple

import pandas
import pandas as pd
from pandas import DataFrame
from tqdm import tqdm

"""
This file reads traffic data from The Hague (in csv format), cleans it, aggregates it and stores it as csv again.
It expects data to be available in some specific format in the /input folder.
"""


def import_intersection_data(intersection_name: str) -> DataFrame:
    """
    Read traffic data of a particular intersection
    :param intersection_name: Name of the intersection
    :return: Dataframe containing the traffic data
    """

    # Temporary storage of data
    # A list of lists, where each inner list is a tuple of three elements
    intersection_data_total = None

    # Read all files in the correct directory
    for file in glob.iglob(
        f"input/traffic data/raw/{intersection_name}/**/*.csv", recursive=True
    ):
        filename = os.fsdecode(file)

        # Ignore files that are not csv's
        if not filename.endswith(".csv"):
            continue

        splitted_string = filename.split("-")

        # Check if we can read the date from the file
        if len(splitted_string) < 3:
            print(
                f"Could not read {filename} in input/{intersection_name} because it does not have the correct "
                f"filename structure "
            )
            continue

        # Read csv file
        dataframe: DataFrame = pd.read_csv(
            f"{filename}",
            sep=";",
        )

        # Add intersection_name column
        dataframe["intersection_name"] = dataframe.columns[0]
        # Move it to be the second column of the dataframe
        dataframe.insert(1, "intersection_name", dataframe.pop("intersection_name"))

        # Rename column containing date to be 'date'
        dataframe.rename(columns={dataframe.columns[0]: "date"}, inplace=True)

        # Remove row that has 'totaal' in it
        dataframe = dataframe[dataframe["date"] != "totaal"]

        # Remove columns that have all NaNs in them
        dataframe.dropna(axis=1, how="all", inplace=True)
        # Set all remaining NaNs to be zero
        dataframe.fillna(axis=1, value=0, inplace=True)

        # Try reading date
        try:
            dataframe["date"] = pd.to_datetime(
                dataframe["date"], format="%Y-%m-%d %H:%M:%S"
            )
        except ValueError:
            # Use a different format
            print(
                f"Could not import date using conventional format, so trying a legacy one instead.."
            )
            dataframe["date"] = pd.to_datetime(
                dataframe["date"], format="%d-%m-%Y %H:%M"
            )

        # Add the preprocessed data to the list of dataframes
        if intersection_data_total is None:
            intersection_data_total = [dataframe]
        else:
            intersection_data_total.append(dataframe)

    # After all files, merge the data into one giant dataframe
    intersection_dataframe = pd.concat(intersection_data_total)

    # Sort the dataframe by date
    intersection_dataframe.sort_values(by="date", inplace=True)
    # Reset the index so it is increasing again.
    intersection_dataframe.reset_index(drop=True, inplace=True)

    return intersection_dataframe


def get_all_intersections() -> List[str]:
    """
    Find all available intersections based on the directory structure in the input folder.
    :return: a list of intersection names
    """
    # Grab all intersection names from the input data
    subdirectories = list(
        filter(
            lambda folder: os.path.isdir(f"input/traffic data/raw/{folder}"),
            os.listdir("input/traffic data/raw/"),
        )
    )
    names = []

    for subdir in subdirectories:
        if "K" in subdir:
            names.append(subdir)

    return names


# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    print(f"\n\n --------- Importing traffic data --------- \n")

    # Loop over each intersection and convert the data
    for intersection in tqdm(get_all_intersections()):
        start_time = time.perf_counter()

        print(f"Importing data of intersection {intersection}")

        intersection_data = import_intersection_data(intersection_name=intersection)

        # Store the data as CSV
        intersection_data.to_csv(
            f"input/traffic data/processed/{intersection}.csv", sep=";", index=False
        )

        print(f"\nTook {time.perf_counter() - start_time} seconds.")

    print(f"--------- Done processing traffic data! ---------")

    print(f"Done!")
