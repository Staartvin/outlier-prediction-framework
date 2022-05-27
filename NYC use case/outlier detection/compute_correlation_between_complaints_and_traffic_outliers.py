import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pandas import DataFrame

"""
This file is used to compute the correlation between a set of complaint outliers and traffic outliers
"""

YEAR = 2017
zone_set = "Lower South Manhattan"


def compute_lagged_correlation(
    complaint_outliers: DataFrame,
    traffic_outliers: DataFrame,
    lag_time=50,
    title=None,
    path_to_save_figure_to=None,
):

    complaint_outliers_series = complaint_outliers["Outlier"].reset_index(drop=True)
    traffic_outliers_series = traffic_outliers["Outlier"].reset_index(drop=True)

    lagged_correlations = [
        traffic_outliers_series.corr(
            complaint_outliers_series.shift(lag, fill_value=0), method="pearson"
        )
        for lag in range(-int(lag_time), int(lag_time + 1))
    ]

    index_of_highest_correlation = np.abs(lagged_correlations).argmax()
    highest_correlation = lagged_correlations[index_of_highest_correlation]

    print(f"Highest correlation found was {highest_correlation}")

    offset = np.floor(len(lagged_correlations) / 2) - index_of_highest_correlation
    f, ax = plt.subplots(figsize=(10, 5))
    ax.plot(lagged_correlations)
    ax.axvline(
        np.ceil(len(lagged_correlations) / 2), color="k", linestyle="--", label="Center"
    )
    ax.axvline(
        index_of_highest_correlation,
        color="r",
        linestyle="--",
        label="Peak synchrony",
    )
    # Add two horizontal lines indicating the highest correlation that was found
    ax.axline(
        (-lag_time, highest_correlation),
        slope=0,
        color="g",
        linestyle="dotted",
        label="Peak correlation",
    )
    ax.axline(
        (-lag_time, -highest_correlation),
        slope=0,
        color="g",
        linestyle="dotted",
    )

    ax.set(
        title=f"Offset = {offset} days (r = {round(highest_correlation, 2)}), {title}\nTraffic leads < | > Complaints lead",
        ylim=[-0.5, 0.5],
        xlim=[0, lag_time * 2 + 1],
        xlabel="Offset (days)",
        ylabel="Pearson r",
    )

    x_ticks = []
    x_ticks_labels = []

    number_of_ticks = 5

    # Generate indices and labels of x-axis ticks
    for i in range(0, number_of_ticks + 1):
        tick_value = i * int((lag_time * 2) / number_of_ticks)

        tick_label = -lag_time + tick_value

        if tick_value >= lag_time:
            tick_value += 1

        x_ticks.append(tick_value)
        x_ticks_labels.append(tick_label)

    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_ticks_labels)
    plt.legend()

    # Save the figuresif required
    if path_to_save_figure_to is not None:
        # Make the directories to the save path
        os.makedirs(os.path.dirname(path_to_save_figure_to), exist_ok=True)
        plt.savefig(path_to_save_figure_to)

    plt.show()


# Try different values of LOF and save the figures to a directory
for lof_neighbor_value_complaints in [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]:
    path_to_complaint_outliers = f"output/outliers/complaints/{YEAR}/{zone_set}/outliers_lof{lof_neighbor_value_complaints}.csv"

    # Skip a file if it does not exist
    if not os.path.exists(path_to_complaint_outliers):
        print(f"Could not find {os.path.basename(path_to_complaint_outliers)}!")
        continue

    complaint_outliers = pd.read_csv(
        path_to_complaint_outliers,
        sep=";",
        decimal=",",
        usecols=["Date", "Value"],
    ).rename({"Value": "Outlier"}, axis=1)

    # For several LOF values, ...
    for lof_neighbor_value_traffic in [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]:

        path_to_traffic_outliers = f"output/outliers/traffic/{YEAR}/{zone_set}/outliers_lof{lof_neighbor_value_traffic}.csv"

        # Skip a file if it does not exist
        if not os.path.exists(path_to_traffic_outliers):
            print(f"Could not find {os.path.basename(path_to_traffic_outliers)}!")
            continue

        traffic_outliers = pd.read_csv(
            path_to_traffic_outliers,
            sep=";",
            decimal=",",
            usecols=["Date", "Value"],
        ).rename({"Value": "Outlier"}, axis=1)

        # Compute the lagged correlation
        compute_lagged_correlation(
            complaint_outliers,
            traffic_outliers,
            title=f"Complaint lof {lof_neighbor_value_complaints}, traffic lof {lof_neighbor_value_traffic}",
            path_to_save_figure_to=f"output/figures/correlation/{YEAR}/{zone_set}/correlation_complaint{lof_neighbor_value_complaints}_traffic_{lof_neighbor_value_traffic}",
        )
