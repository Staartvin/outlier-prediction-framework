import glob
import os
import time
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from pandas import DataFrame

"""
This file generates a set of images (one for each day of the week) that show the distribution of correlation values when
one intersection is shifted over another intersection. 
"""

# Traffic flows from intersection one
intersection_one = "K159"

# To intersection two
intersection_two = "K183"

begin_index_minute = 0
end_index_minute = 1440
step_minute = 5

# Keep reference of traffic data for each intersection (key = intersection, value = dataframe)
traffic_data_per_intersection: Dict[str, DataFrame] = dict()

# Interesting traffic flows:
# 159 (u081) -> 128 ["071", "072", "081", "081"]
# 206 (021, 022) -> 128 (131, 141, 142)
# 703 ["U08_2", "U08_1"] -> 128 ["071", "072", "081", "081"]
# 561 ["u02_1", "u02_2"] -> 504 ["03_1", "02_2", "02_1", "01_2", "01_1"]

# The lanes that we want to use for both intersections
interesting_lanes_per_intersection = {
    "K559": ["u02_1", "u02_2"],
    "K557": ["02_5"],
    "K504": ["03_1", "02_2", "02_1", "01_2", "01_1"],
    "K206": ["021", "022"],
    "K051": ["U081", "U082"],
    "K101": ["u621", "u622"],
    "K128": ["131", "062", "061"],
    "K182": ["081", "082"],
    "K183": ["081", "082", "091"],
    "K703": ["U08_2", "U08_1"],
    "K561": ["u02_1", "u02_2"],
    "K502": ["011", "021", "022", "031"],
    "K159": ["u081"],
    "K139": ["021", "022", "101"],
}


def import_traffic_data():
    """
    Import the traffic data for all intersections. It creates cached versions of the intersection files if they were not created yet.
    :return: nothing
    """

    global traffic_data_per_intersection

    start_time = time.perf_counter()

    # Read all files in the correct directory
    for file in glob.iglob(f"input/traffic data/processed/*.csv", recursive=True):
        file_path = os.fsdecode(file)

        intersection_name = os.path.basename(file_path).replace(".csv", "")

        # Try to find a cached version
        if os.path.exists(f"input/traffic data/processed/cache/{intersection_name}.feather"):
            dataframe = pd.read_feather(f"input/traffic data/processed/cache/{intersection_name}.feather")
        else:
            # Read from csv (non-cached)
            print(f"Loading non-cached version of {intersection_name}")
            dataframe = pd.read_csv(file_path, sep=";")
            dataframe["date"] = pd.to_datetime(dataframe["date"], format="%Y-%m-%d %H:%M:%S")

            columns_to_rename = dataframe.columns.drop(["date", "intersection_name"])
            dataframe.loc[:, columns_to_rename] = dataframe.loc[:, columns_to_rename].astype("Int16")

            # Create a cached version
            dataframe.to_feather(f"input/traffic data/processed/cache/{intersection_name}.feather")

        traffic_data_per_intersection[intersection_name] = dataframe

    print(f"Read all traffic data in {time.perf_counter() - start_time} seconds")


def grab_interesting_lanes(intersection_data: DataFrame, intersection_name: str) -> DataFrame:
    """
    Given some intersection data, select only the interesting lanes
    :param intersection_data: Data to select lanes from
    :param intersection_name: Name of the intersection
    :return: dataframe consisting of only interesting lanes
    """
    # Grab the lanes that are interesting for this intersection
    interesting_lanes: List[str] = interesting_lanes_per_intersection[intersection_name]

    # Only select columns of lanes we want to keep (and the date and intersection name)
    filtered_intersection_data = intersection_data.loc[:, ["date"] + interesting_lanes]

    # Compute sum of all detections in the selected lanes
    filtered_intersection_data["total_detections"] = filtered_intersection_data.drop("date", axis=1).sum(axis=1)

    # Return a copy of the intersection data
    return filtered_intersection_data.loc[:, ["date", "total_detections"]].copy()


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


def compute_correlation_each_timestep(
        traffic_data_intersection_one: DataFrame,
        traffic_data_intersection_two: DataFrame,
        day: str,
        path_to_save_figure_to: str = None,
):
    """
    Compute the correlation distribution between two intersections for a given day
    :param traffic_data_intersection_one: Dataframe of intersection one
    :param traffic_data_intersection_two: Dataframe of intersection two
    :param day: Day to use
    :param path_to_save_figure_to: Optional path to save the figures to
    :return:
    """
    # Merge two dataframes together to make sure they align on datetimes
    merged_data = traffic_data_intersection_one.merge(traffic_data_intersection_two, on="date",
                                                      suffixes=("_one", "_two"))
    offsets_in_minutes = []
    highest_correlations = []

    # Filter data so only correct days of the week show up
    merged_data = merged_data[merged_data["date"].dt.day_of_week == day_name_to_day_of_week_number(day)]

    # Group dataframe by date
    data_grouped_by_date = merged_data.groupby(merged_data["date"].dt.date)

    # For each date, slide the second intersection of the first
    for date, data_at_date in data_grouped_by_date:

        # Keep track of the correlation values
        lagged_correlations = []

        # Grab the data of the first intersection
        intersection_one_series = data_at_date["total_detections_one"]

        max_lag_in_hours = 1

        # Slide the second intersection over the first, in steps of 5 minutes.
        for offset in range(-max_lag_in_hours * 12, max_lag_in_hours * 12):
            # Shift the second intersection by the offset
            shifted_data_intersection_two = merged_data["total_detections_two"].shift(offset)

            intersection_two_series = shifted_data_intersection_two[intersection_one_series.index]

            # Compute the correlation between the two intersections
            correlation = intersection_one_series.corr(intersection_two_series)

            # Store the correlation
            lagged_correlations.append(correlation)

        # Find the highest correlation
        index_of_highest_correlation = np.abs(lagged_correlations).argmax()
        highest_correlation = lagged_correlations[index_of_highest_correlation]

        # Compute what that would mean in offset (of minutes)
        offset_in_minutes = -int((int(index_of_highest_correlation) - (len(lagged_correlations) / 2)) * 5)
        highest_correlations.append(highest_correlation)

        # Store this offset for this date
        offsets_in_minutes.append(offset_in_minutes)

    # Compute the average offset over all dates.
    average_offset = float(np.nanmean(offsets_in_minutes))

    # Generate a kernel-density plot for the distribution of offsets
    sns.kdeplot(offsets_in_minutes, label="Lag")
    plt.xlabel(f"Lag to {intersection_one} (minutes)")
    plt.suptitle(f"Distribution of lag for {intersection_two} on {day}")
    plt.title(f"Mean lag = {round(average_offset, 2)}", fontsize=10)

    # Add a vertical line that indicates the mean lag
    plt.axvline(
        average_offset,
        color="r",
        linestyle="--",
        label="Mean lag",
    )
    plt.legend()

    # Generate the directory
    os.makedirs(f"output/figures/correlation/{intersection_one}/{intersection_two}/", exist_ok=True)

    # Save the figure
    plt.savefig(
        f"output/figures/correlation/{intersection_one}/{intersection_two}/traffic_lag_{day_name_to_day_of_week_number(day)}_{day}.png")
    plt.show()

    # Another KDE plot for peak correlations
    sns.kdeplot(highest_correlations, label="Peak correlations")
    plt.xlabel("Pearson correlation")
    plt.title(f"Distribution of peak correlation for {intersection_two} vs {intersection_one} on {day}")

    plt.legend()

    os.makedirs(f"output/figures/correlation/{intersection_one}/{intersection_two}/",
                exist_ok=True)
    plt.savefig(
        f"output/figures/correlation/{intersection_one}/{intersection_two}/traffic_correlation_{day_name_to_day_of_week_number(day)}_{day}.png")

    plt.show()

    print(f"Average offset of {intersection_two} should be {round(average_offset, 2)} minutes on {day}")


def show_synchrony_plot(correlation_values: Sequence[float], title: str, path_to_save_figure_to: str = None):
    x_axis_time_series = []

    tick_at_each_minutes = 30

    size_of_one_half_of_interval = (len(correlation_values) / 2) * 5

    # Create x-axis value so we have a tick for each hour
    for i in np.arange(-size_of_one_half_of_interval, size_of_one_half_of_interval + 1, tick_at_each_minutes):
        x_axis_time_series.append(int(i))

    index_of_highest_correlation = np.abs(correlation_values).argmax()
    highest_correlation = correlation_values[index_of_highest_correlation]

    offset_in_minutes = int((int(index_of_highest_correlation) - (len(correlation_values) / 2)) * 5)

    f, ax = plt.subplots(figsize=(10, 5))
    ax.plot(correlation_values)
    plt.xticks(ticks=np.arange(0, len(correlation_values) + 1, tick_at_each_minutes / 5), labels=x_axis_time_series)
    ax.axvline(
        index_of_highest_correlation,
        color="r",
        linestyle="--",
        label="Peak synchrony",
    )
    # Add two horizontal lines indicating the highest correlation that was found
    ax.axline(
        (0, highest_correlation),
        slope=0,
        color="g",
        linestyle="dotted",
        label="Peak correlation",
    )
    ax.axline(
        (0, -highest_correlation),
        slope=0,
        color="g",
        linestyle="dotted",
    )

    ax.set(
        title=f"Peak correlation @ = {offset_in_minutes} minutes (r = {round(highest_correlation, 2)}), {title}",
        ylim=[-1.0, 1.0],
        xlim=[0, len(correlation_values)],
        xlabel="Offset time (minutes)",
        ylabel="Pearson r",
    )

    if (offset_in_minutes < 0):
        plt.suptitle(f"Intersection {intersection_two} leads intersection {intersection_one}")
    elif (offset_in_minutes > 0):
        plt.suptitle(f"Intersection {intersection_one} leads intersection {intersection_two}")
    else:
        plt.suptitle("Both intersections are simultaneous")

    plt.legend()

    # Save the figures if required
    if path_to_save_figure_to is not None:
        # Make the directories to the save path
        os.makedirs(os.path.dirname(path_to_save_figure_to), exist_ok=True)
        plt.savefig(path_to_save_figure_to)

    plt.show()


# Import data of all intersections
import_traffic_data()

# Grab only those intersections that are interesting.
traffic_data_intersection_one = grab_interesting_lanes(traffic_data_per_intersection[intersection_one],
                                                       intersection_one)
traffic_data_intersection_two = grab_interesting_lanes(traffic_data_per_intersection[intersection_two],
                                                       intersection_two)

# For each day of the week, compute the correlation between two intersections
for day in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]:
    # Compute the lagged correlation
    compute_correlation_each_timestep(
        traffic_data_intersection_one,
        traffic_data_intersection_two,
        day
    )
