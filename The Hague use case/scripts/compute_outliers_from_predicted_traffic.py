import os
from datetime import datetime, timedelta, time

import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.neighbors import LocalOutlierFactor

"""
This file computes outliers from the predicted data of an intersection
"""

# Whether we want to use predictions from the enriched of baseline model
model = "enriched"

# Year to obtain predictions from
year = 2020

# Intersection to obtain data from
intersection = "K159"

# LOF number
LOF_NUMBER_OF_NEIGHBORS = 5

# Path to the predictions
path_to_file = (
    f"output/prediction/{model}/traffic/{year}/{intersection}/traffic_prediction.csv"
)

# Read the csv file
predicted_intersection_data = pd.read_csv(path_to_file, sep=";")

# Convert the date column to timestamp objects
predicted_intersection_data["date"] = pd.to_datetime(
    predicted_intersection_data["date"]
)

# Compute the day name to a day number in the week
def day_name_to_day_of_week_number(day_name: str) -> int:
    if day_name == "Monday":
        return 0
    elif day_name == "Tuesday":
        return 1
    elif day_name == "Wednesday":
        return 2
    elif day_name == "Thursday":
        return 3
    elif day_name == "Friday":
        return 4
    elif day_name == "Saturday":
        return 5
    elif day_name == "Sunday":
        return 6


def convert_intersection_data_to_lof_computable(
    dataframe: DataFrame,
    time_window_begin: datetime.time = None,
    time_window_end: datetime.time = None,
) -> DataFrame:
    """
    This method is fed a dataframe in a specific format and transforms it into a different format so we can use it to run LOF
    :param dataframe: Dataframe with traffic volume data
    :param time_window_begin: The start of the time window
    :param time_window_end: The end of the time window
    :return: a dataframe containing the same data, but in a different format.
    """
    temporary_data_list = []
    column_names = [
        "date",
    ]

    current_time = time_window_begin

    while current_time < time_window_end:
        next_time = (
            datetime.combine(datetime.today(), current_time) + timedelta(minutes=5)
        ).time()

        column_names.append(f"{current_time.minute} - {next_time.minute}")

        if next_time < current_time:
            break

        current_time = next_time

    # Compute sum of all detections in the selected lanes
    dataframe["total_detections"] = dataframe.sum(numeric_only=True, axis=1)

    # Drop the intersection name
    dataframe = dataframe.drop("Model forecast", axis=1)

    dataframe = dataframe.set_index("date")
    dataframe = dataframe.between_time(
        time_window_begin, time_window_end, inclusive="left"
    )

    grouped_data = dataframe.groupby(by=dataframe.index.date)

    for name, group in grouped_data:
        transposed_group = group.transpose()

        filtered_row = transposed_group[(transposed_group.index == "total_detections")]

        date = filtered_row.columns[0]

        # Generate points that are zero from start
        list_of_row = [0] * (len(column_names) - 1)

        if len(list_of_row) == 0:
            continue

        # Loop over the number of detections per 5 minutes and put them in the right slot in the list
        for index, col_name in enumerate(filtered_row):
            date_of_detection = col_name
            number_of_detections = filtered_row[col_name][0]

            list_of_row[
                int((date_of_detection.minute - time_window_begin.minute) / 5)
            ] = number_of_detections

        # Insert date at the beginning of the list
        list_of_row.insert(0, date)

        # Make sure to check whether the list has the correct length
        assert len(list_of_row) == len(column_names)

        temporary_data_list.append(list_of_row)

    converted_dataframe = pd.DataFrame(temporary_data_list, columns=column_names)

    return converted_dataframe


# Store reference of outliers.
outlier_names = []

# Iterate over each day in the week and generate the outliers
for day in [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]:
    # Generate a mask that picks datapoints that are only on the selected day of the week
    day_name_mask = predicted_intersection_data[
        "date"
    ].dt.day_of_week == day_name_to_day_of_week_number(day)

    # Grab rows that are on the specific day and hour
    predicted_intersection_data = predicted_intersection_data.loc[day_name_mask].copy()

    begin_index_minute = 0
    end_index_minute = 1440
    step_minute = 30

    # For each time interval of 30 minutes, try to detect outliers
    for time_interval in np.arange(begin_index_minute, end_index_minute, step_minute):
        time_interval = int(time_interval)

        # Determine start of the time window
        selected_time_window_begin = (
            datetime.combine(datetime.today(), time(hour=0, minute=0))
            + timedelta(minutes=time_interval)
        ).time()

        # Determine end of the time window
        selected_time_window_end = (
            datetime.combine(datetime.today(), time(hour=0, minute=0))
            + timedelta(minutes=time_interval + step_minute)
        ).time()

        # Generate a new dataframe that can be used to compute outliers
        converted_intersection_data = convert_intersection_data_to_lof_computable(
            predicted_intersection_data,
            selected_time_window_begin,
            selected_time_window_end,
        )

        # Drop date column from data
        vectorized_data = converted_intersection_data.drop(labels=["date"], axis=1)

        # Perform LOF to detect outlier time patterns
        outlier_detector = LocalOutlierFactor(
            n_neighbors=LOF_NUMBER_OF_NEIGHBORS,
            metric="chebyshev",
        )

        # Skip the data if it has no rows
        if vectorized_data.shape[0] == 0:
            continue

        # Determine whether a time series is an outlier or not
        outlier_scores = outlier_detector.fit_predict(vectorized_data)

        # Loop over outlier scores and see which ones are outliers
        for index, outlier_score in enumerate(outlier_scores):
            # We have an outlier!
            if outlier_score == -1:
                outlier_name = converted_intersection_data.iloc[index]["date"]
                outlier_name = datetime.combine(
                    outlier_name, selected_time_window_begin
                )
                outlier_names.append(outlier_name)

print(f"Found {len(outlier_names)} outliers")

# Store the outliers in a dataframe
outliers_dataframe = pd.DataFrame(outlier_names, columns=["Date"])

# Make the directories if they do not exist yet.
os.makedirs(f"output/outliers/traffic/{year}/{intersection}/", exist_ok=True)

# And write the outliers to a file again
outliers_dataframe.to_csv(
    f"output/outliers/traffic/{year}/{intersection}/traffic_outliers_lof{LOF_NUMBER_OF_NEIGHBORS}.csv",
    index=False,
    sep=";",
)
