import os
import time
from datetime import date, timedelta
from typing import List

import dash
import dtw
import numpy as np
import pandas as pd
import plotly.express as px
from dash import html, dcc
from dash.exceptions import PreventUpdate
from pandas import DataFrame, Timestamp
from scipy.spatial.distance import euclidean
from sklearn.neighbors import LocalOutlierFactor
import matplotlib.pyplot as plt

"""
This file is used to display outliers in an interactive visualization using Dash and plotly. It shows the outliers of complaints and traffic from the ground truth values.
"""

app = dash.Dash(__name__)

# Keep reference of traffic data for each intersection (key = intersection, value = dataframe)
global_complaint_data: DataFrame = None
# Global taxi data
global_taxi_data: DataFrame = None

# Store a cached version of the figure if we don't want to redraw
global_cached_figure = None

# Number of neighbors to use during the LOF calculation (for complaint dataset)
NUMBER_NEIGHBORS_LOF_COMPLAINTS = 30
# Number of neighbors to use during the LOF calculation (for traffic dataset)
NUMBER_NEIGHBORS_LOF_TRAFFIC = 30

# How many hours do we want in one bin of data?
HOUR_SAMPLING = 4

# Whether we should convert the data to a probability density function before applying LOF
TRANSFORM_TO_PDF_BEFORE_DETECTING_OUTLIERS = False

# Which distance metric to use for LOF outlier detection
LOF_METRIC = "chebyshev"

# Which year to use for selecting complaints
YEAR_SELECTOR = 2018

SAVE_TRAFFIC_OUTLIERS = True
SAVE_COMPLAINT_OUTLIERS = True


# Import all complaint data from the csvs
def import_complaint_data():
    global global_complaint_data

    start_time = time.perf_counter()

    global_complaint_data = pd.read_csv(
        f"input/complaints/preprocessed complaints manhattan with zones and zonesets.csv",
        sep=";",
        usecols=["Unique Key", "Closed Date", "zone", "zone set"],
    )

    global_complaint_data["Closed Date"] = pd.to_datetime(
        global_complaint_data["Closed Date"], format="%d-%m-%Y %H:%M:%S"
    )

    print(f"Read all complaint data in {time.perf_counter() - start_time} seconds")


# Import all traffic data from the csvs
def import_taxi_data():
    global global_taxi_data

    start_time = time.perf_counter()

    dropoff_data = pd.read_csv(
        f"input/taxi data/{YEAR_SELECTOR}/preprocessed/dropoff_per_zone_{YEAR_SELECTOR}_with_zonesets.csv",
        sep=";",
    )

    dropoff_data["date"] = pd.to_datetime(dropoff_data["date"], format="%d-%m-%Y")

    pickup_data = pd.read_csv(
        f"input/taxi data/{YEAR_SELECTOR}/preprocessed/pickups_per_zone_{YEAR_SELECTOR}_with_zonesets.csv",
        sep=";",
    )

    pickup_data["date"] = pd.to_datetime(pickup_data["date"], format="%d-%m-%Y")

    # Combine both pickup and dropoff data
    global_taxi_data = pd.concat([dropoff_data, pickup_data])

    print(f"Read all taxi data in {time.perf_counter() - start_time} seconds")


def filter_complaint_data(zone_set="North Manhattan") -> DataFrame:
    dates_to_match = []

    # Grab taxi data of a specific zone
    complaint_data: DataFrame = global_complaint_data[
        global_complaint_data["zone set"] == zone_set
    ].copy()

    mask = complaint_data["Closed Date"].dt.year == YEAR_SELECTOR

    # Keep all rows that have data with the particular day name
    complaint_data = complaint_data[mask]

    # Sort data on datetime
    complaint_data = complaint_data.sort_values("Closed Date")

    return complaint_data


def convert_filtered_complaint_data_to_lof_computable(
    dataframe: DataFrame,
) -> DataFrame:
    temporary_data_list = []
    column_names = [
        "Row name",
        "Date",
    ]

    # Add appropriate column names based on hour sampling
    for i in range(HOUR_SAMPLING, 24 + 1, HOUR_SAMPLING):
        column_names.append(f"{i}")

    resampled = dataframe.resample(
        f"{HOUR_SAMPLING}H",
        on="Closed Date",
        origin="start_day",
    )

    current_date = None
    current_data_row = []

    name: Timestamp
    group: DataFrame
    for name, group in resampled:
        # Format name of row by time of group name
        row_name = name.strftime("%A %d %B %Y")
        count = group.count()["Unique Key"]

        # Update current date so we know which day we are working with
        if current_date is None:
            current_date = name
            current_data_row = [row_name, name.date()]
            current_data_row.extend([0] * int(24 / HOUR_SAMPLING))

        # determine the difference in time between the start of this row and the current group
        time_difference = name - current_date

        # Reset the list when we are at a new day
        if time_difference.days >= 1:
            # Submit previous data
            temporary_data_list.append(current_data_row)

            # Create new data for the upcoming day
            current_date = name
            current_data_row = [row_name, name.date()]
            current_data_row.extend([0] * int(24 / HOUR_SAMPLING))

        # Determine the index of the column we need to place this data into
        index = int(name.hour / HOUR_SAMPLING) + 2

        # Add the count to the current row (at the correct column)
        current_data_row[index] = count

    # Add the last day to the list
    temporary_data_list.append(current_data_row)

    converted_dataframe = pd.DataFrame(temporary_data_list, columns=column_names)

    return converted_dataframe


# Convert name of day to a number in the week
def day_name_to_week_number(day_name):
    if day_name == "Monday":
        return 1
    if day_name == "Tuesday":
        return 2
    if day_name == "Wednesday":
        return 3
    if day_name == "Thursday":
        return 4
    if day_name == "Friday":
        return 5
    if day_name == "Saturday":
        return 6
    if day_name == "Sunday":
        return 7


def dates_of_particular_week(year: int, week: int) -> List[date]:
    dates: List[date] = []

    start_date = date(year, 1, 1) + timedelta(weeks=+(week - 1))

    dates.append(start_date)

    # Find the 7 days of the week
    for _ in range(1, 7):
        start_date += timedelta(days=+1)
        dates.append(start_date)

    return dates


def last_week_of_a_year(year: int) -> int:
    return date(year, 12, 28).isocalendar()[1]


# Get all dates of a year that fall on a particular day name (e.g. Thursday)
def dates_of_particular_day_name(year: int, day_name: str):
    d = date(year, 1, 1)
    d += timedelta(days=(day_name_to_week_number(day_name) - d.weekday()) % 7)
    while d.year == year:
        yield d
        d += timedelta(days=7)


# Generate a dataframe (based on the global traffic data) that contains data of all days (e.g. Tuesdays) from a
# specific intersection
def generate_comparable_taxi_data(day_name="Monday", zone="TestZone") -> DataFrame:
    weekday_to_filter = day_name

    dates_to_match = [
        d for d in dates_of_particular_day_name(YEAR_SELECTOR, weekday_to_filter)
    ]

    # Grab taxi data of a specific zone
    taxi_data_of_zone: DataFrame = global_taxi_data[
        global_taxi_data["zone set"] == zone
    ].copy()

    # Keep all rows that have data with the particular day name
    taxi_data_of_zone = taxi_data_of_zone[
        taxi_data_of_zone["date"].isin(dates_to_match)
    ]

    # Sort data on datetime
    taxi_data_of_zone = taxi_data_of_zone.sort_values("date")

    grouped_zones_into_single_date = taxi_data_of_zone.groupby("date").sum()
    grouped_zones_into_single_date["zone set"] = zone
    grouped_zones_into_single_date.reset_index(inplace=True)

    return grouped_zones_into_single_date


@app.callback(
    [
        dash.dependencies.Output("outliers-plot", "figure"),
        dash.dependencies.Output("outliers-plot2", "figure"),
    ],
    [
        dash.dependencies.Input("zoneset-selector", "value"),
        dash.dependencies.Input("number_lof_neighbors_complaints", "value"),
        dash.dependencies.Input("number_lof_neighbors_traffic", "value"),
    ],
)
def adjusted_input_parameters(
    selected_zoneset, number_of_neighbors_complaints, number_of_neighbors_traffic
):
    if (
        selected_zoneset is None
        or number_of_neighbors_complaints is None
        or number_of_neighbors_traffic is None
    ):
        raise PreventUpdate()

    global NUMBER_NEIGHBORS_LOF_COMPLAINTS
    NUMBER_NEIGHBORS_LOF_COMPLAINTS = int(number_of_neighbors_complaints)

    global NUMBER_NEIGHBORS_LOF_TRAFFIC
    NUMBER_NEIGHBORS_LOF_TRAFFIC = int(number_of_neighbors_traffic)

    # Get data of traffic
    filtered_complaints = filter_complaint_data(zone_set=selected_zoneset)

    start_time = time.perf_counter()

    # Grab outliers of complaints for every day of the week
    complaint_outliers: List[str] = []

    print(f"\nComputing complaint outliers..\n")

    converted_dataframe = convert_filtered_complaint_data_to_lof_computable(
        filtered_complaints
    )

    for day in [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]:
        start_int_time = time.perf_counter()

        data_of_day = converted_dataframe[
            converted_dataframe["Row name"].str.contains(day)
        ]

        data_of_day.loc[:, "Date"] = pd.to_datetime(data_of_day["Date"])

        data_of_day_first_half = data_of_day[
            data_of_day["Date"].between(
                f"{YEAR_SELECTOR}-01-01", f"{YEAR_SELECTOR}-06-30", inclusive="both"
            )
        ]

        data_of_day_second_half = data_of_day[
            data_of_day["Date"].between(
                f"{YEAR_SELECTOR}-07-01", f"{YEAR_SELECTOR}-12-31", inclusive="both"
            )
        ]

        # Look at outliers of the whole year
        outliers_all_year = compute_complaint_outliers(data_of_day)
        complaint_outliers.extend(outliers_all_year)

    print(
        f"Took {time.perf_counter() - start_time} seconds to find outliers for complaint data"
    )

    # Grab outliers of traffic for every day of the week
    traffic_outliers: List[str] = []

    start_time = time.perf_counter()

    print(f"\nComputing traffic outliers..\n")

    for day in [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]:
        # Get data of traffic
        comparable_traffic_data = generate_comparable_taxi_data(
            day, zone=selected_zoneset
        )
        traffic_outliers.extend(compute_traffic_outliers(comparable_traffic_data))

    print(
        f"Took {time.perf_counter() - start_time} seconds to find outliers for traffic data"
    )

    start_time = time.perf_counter()

    dataframe_list = []

    start_date = date(YEAR_SELECTOR, 1, 1)
    end_date = date(YEAR_SELECTOR, 12, 31)

    while start_date <= end_date:

        is_outlier_complaint = False
        is_outlier_traffic = False

        date_to_string = start_date.strftime("%A %d %B %Y")

        if date_to_string in complaint_outliers:
            dataframe_list.append([date_to_string, "complaint", start_date, 1])

            is_outlier_complaint = True

        if date_to_string in traffic_outliers:
            dataframe_list.append([date_to_string, "traffic", start_date, 1])

            is_outlier_traffic = True

        if not is_outlier_complaint:
            dataframe_list.append([date_to_string, "complaint", start_date, 0])

        if not is_outlier_traffic:
            dataframe_list.append([date_to_string, "traffic", start_date, 0])

        start_date += timedelta(days=1)

    print(f"Took {time.perf_counter() - start_time} seconds to compute the dataframe")

    outliers_in_dataframe: DataFrame = pd.DataFrame(
        data=dataframe_list,
        columns=["Outlier", "Type", "Date", "Value"],
    )

    start_time = time.perf_counter()

    line_plot = px.line(
        outliers_in_dataframe,
        x="Date",
        y="Value",
        color="Type",
        color_discrete_map={"complaint": "orange", "traffic": "blue"},
        title=f"Timeline of outliers of complaints and traffic in {YEAR_SELECTOR}",
        line_shape="spline",
    )

    violin_plot = px.violin(
        outliers_in_dataframe[outliers_in_dataframe["Value"] == 1],
        x="Date",
        color="Type",
        color_discrete_map={"complaint": "orange", "traffic": "blue"},
        title=f"Outliers of complaints and traffic patterns in {YEAR_SELECTOR}",
        points="all",
    )

    print(f"Took {time.perf_counter() - start_time} seconds to compute figures")

    print("Done computing!")

    if SAVE_TRAFFIC_OUTLIERS:

        print(f"Writing traffic outliers to csv file")

        path_to_save_traffic = f"output/outliers/traffic/{YEAR_SELECTOR}/{selected_zoneset}/outliers_lof{number_of_neighbors_traffic}.csv"

        # Create dirs if they do not exist
        os.makedirs(os.path.dirname(path_to_save_traffic), exist_ok=True)

        # Write traffic outliers to a file
        outliers_in_dataframe[outliers_in_dataframe["Type"] == "traffic"].to_csv(
            path_to_save_traffic,
            sep=";",
            index=False,
            decimal=",",
        )

    if SAVE_COMPLAINT_OUTLIERS:
        print(f"Writing complaint outliers to csv file")
        path_to_save_complaints = f"output/outliers/complaints/{YEAR_SELECTOR}/{selected_zoneset}/outliers_lof{number_of_neighbors_complaints}.csv"

        # Create dirs if they do not exist
        os.makedirs(os.path.dirname(path_to_save_complaints), exist_ok=True)

        # Write complaint outliers to a file
        outliers_in_dataframe[outliers_in_dataframe["Type"] == "complaint"].to_csv(
            path_to_save_complaints,
            sep=";",
            index=False,
            decimal=",",
        )

    return line_plot, violin_plot


# Return a list of days that are outliers (based on a LOF method)
def compute_traffic_outliers(dataframe: DataFrame) -> List[str]:
    # Convert data to probability density function
    vectorized_data = dataframe.drop(labels=["zone set", "date"], axis=1)

    # Merge columns into smaller set of columns
    vectorized_data["0-4"] = (
        vectorized_data["1"]
        + vectorized_data["2"]
        + vectorized_data["3"]
        + vectorized_data["4"]
    )
    vectorized_data["5-8"] = (
        vectorized_data["5"]
        + vectorized_data["6"]
        + vectorized_data["7"]
        + vectorized_data["8"]
    )
    vectorized_data["9-12"] = (
        vectorized_data["9"]
        + vectorized_data["10"]
        + vectorized_data["11"]
        + vectorized_data["12"]
    )
    vectorized_data["13-16"] = (
        vectorized_data["13"]
        + vectorized_data["14"]
        + vectorized_data["15"]
        + vectorized_data["16"]
    )
    vectorized_data["17-20"] = (
        vectorized_data["17"]
        + vectorized_data["18"]
        + vectorized_data["19"]
        + vectorized_data["20"]
    )
    vectorized_data["21-24"] = (
        vectorized_data["21"]
        + vectorized_data["22"]
        + vectorized_data["23"]
        + vectorized_data["24"]
    )

    vectorized_data = vectorized_data.loc[:, "0-4":"21-24"]

    # Check if we should convert the data to a PDF or not
    if TRANSFORM_TO_PDF_BEFORE_DETECTING_OUTLIERS:
        vectorized_sum = vectorized_data.sum(axis=1)
        vectorized_data = vectorized_data.div(vectorized_sum, axis=0)

    # Perform LOF to detect outlier time patterns
    outlier_detector = LocalOutlierFactor(
        n_neighbors=NUMBER_NEIGHBORS_LOF_TRAFFIC, metric=LOF_METRIC
    )

    # Determine whether a time series is an outlier or not
    outlier_scores = outlier_detector.fit_predict(vectorized_data)

    # Store reference of outliers.
    outlier_names = []

    # Loop over outlier scores and see which ones are outliers
    for index, outlier_score in enumerate(outlier_scores):
        # We have an outlier!
        if outlier_score == -1:
            outlier_name = dataframe.iloc[index]["date"].strftime("%A %d %B %Y")
            outlier_names.append(outlier_name)

    return outlier_names


# Return a list of days that are outliers (based on a LOF method)
def compute_complaint_outliers(dataframe: DataFrame, day="Monday") -> List[str]:
    # Convert data to probability density function
    vectorized_data = dataframe.drop(labels=["Row name", "Date"], axis=1)

    if TRANSFORM_TO_PDF_BEFORE_DETECTING_OUTLIERS:
        vectorized_sum = vectorized_data.sum(axis=1)
        vectorized_data = vectorized_data.div(vectorized_sum, axis=0)

    # Perform LOF to detect outlier time patterns
    outlier_detector = LocalOutlierFactor(
        n_neighbors=NUMBER_NEIGHBORS_LOF_COMPLAINTS,
        metric=LOF_METRIC,
    )

    # Determine whether a time series is an outlier or not
    outlier_scores = outlier_detector.fit_predict(vectorized_data)

    # Store reference of outliers.
    outlier_names = []

    # Loop over outlier scores and see which ones are outliers
    for index, outlier_score in enumerate(outlier_scores):
        # We have an outlier!
        if outlier_score == -1:
            outlier_name = dataframe.iloc[index]["Row name"]
            outlier_names.append(outlier_name)

    return outlier_names


# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    px.set_mapbox_access_token(
        "pk.eyJ1Ijoic3RhYXJ0dmluIiwiYSI6ImNqcWRsNXRjZzQ2cDg0OG11YXpnNGh6eWQifQ._MVcI"
        "-FJIM3u0dPjaAj_lg"
    )

    print(f"\n\n --------- Importing complaint data --------- \n")
    import_complaint_data()

    print(f"\n\n --------- Importing traffic data --------- \n")
    import_taxi_data()

    app.layout = html.Div(
        children=[
            dcc.Graph(id="outliers-plot", figure={}),
            dcc.Graph(id="outliers-plot2", figure={}),
            html.H4(
                "Number of neighbors for LOF (for complaint outliers)",
                style={"display": "inline-block", "margin-right": 20},
            ),
            dcc.Input(
                id="number_lof_neighbors_complaints",
                type="number",
                placeholder=str(NUMBER_NEIGHBORS_LOF_COMPLAINTS),
                value=NUMBER_NEIGHBORS_LOF_COMPLAINTS,
                debounce=True,
            ),
            html.Br(),
            html.Br(),
            html.H4(
                "Number of neighbors for LOF (for traffic outliers)",
                style={"display": "inline-block", "margin-right": 20},
            ),
            dcc.Input(
                id="number_lof_neighbors_traffic",
                type="number",
                placeholder=str(NUMBER_NEIGHBORS_LOF_TRAFFIC),
                value=NUMBER_NEIGHBORS_LOF_TRAFFIC,
                debounce=True,
            ),
            html.Br(),
            html.Br(),
            dcc.Dropdown(
                id="zoneset-selector",
                options=[
                    {"label": "North Manhattan", "value": "North Manhattan"},
                    {
                        "label": "Upper Middle Manhattan",
                        "value": "Upper Middle Manhattan",
                    },
                    {
                        "label": "Lower Middle Manhattan",
                        "value": "Lower Middle Manhattan",
                    },
                    {
                        "label": "Upper South Manhattan",
                        "value": "Upper South Manhattan",
                    },
                    {
                        "label": "Lower South Manhattan",
                        "value": "Lower South Manhattan",
                    },
                ],
                value="North Manhattan",
            ),
            html.Br(),
            html.Br(),
        ]
    )

    app.run_server(debug=True, port=7999, use_reloader=False)

    print(f"Done!")
