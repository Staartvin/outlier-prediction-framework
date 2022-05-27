import math
import os
import time
from collections import defaultdict
from datetime import date, datetime, timedelta
from math import sqrt
from typing import List, Tuple

import dash
import numpy
import pandas as pd
import plotly.express as px
import sklearn.metrics
from dash import html, dcc
from dash.exceptions import PreventUpdate
from pandas import DataFrame, Timestamp
from plotly.graph_objs import Figure
from sklearn.neighbors import LocalOutlierFactor

"""
This file is used to display patterns of traffic and complaints in an interactive visualization using Dash and plotly. It shows the complaints and traffic from the ground truth values.
"""

app = dash.Dash(__name__)

# Keep reference of traffic data for each intersection (key = intersection, value = dataframe)
global_complaint_data: DataFrame = None

# Store a cached version of the figure if we don't want to redraw
global_cached_figure = None

# Number of neighbors to use during the LOF calculation
NUMBER_NEIGHBORS_LOF_COMPLAINTS = 20

# How many hours is one bin in a day
HOUR_SAMPLING = 4

# Whether we should convert the data to a probability density function before applying LOF
TRANSFORM_TO_PDF_BEFORE_DETECTING_OUTLIERS = False

# Which distance metric to use for LOF outlier detection
LOF_METRIC = "chebyshev"

ZONE_SET = "North Manhattan"
BATCH_SIZE = 30
SEQUENCE_LENGTH = 6

# Import all traffic data from the csvs
def import_complaint_data():
    global global_complaint_data

    start_time = time.perf_counter()

    global_complaint_data = pd.read_csv(
        f"output/prediction/baseline/complaints/2018/{ZONE_SET}/complaints_batch{BATCH_SIZE}_sequence{SEQUENCE_LENGTH}.csv",
        sep=";",
    )

    global_complaint_data["Created Date"] = pd.to_datetime(
        global_complaint_data["Created Date"], format="%Y-%m-%d %H:%M:%S"
    )

    print(f"Read all complaint data in {time.perf_counter() - start_time} seconds")


def convert_dataframe_to_displayable_dataframe(
    dataframe: DataFrame, day="Monday"
) -> DataFrame:
    column_names = ["zone set", "date", "name", "hour in day", "complaints"]

    temporary_data_list = []

    resampled = dataframe.resample(
        f"{HOUR_SAMPLING}H",
        on="Created Date",
    )

    zone_set = ZONE_SET

    name: Timestamp
    group: DataFrame
    for name, group in resampled:

        if name.isoweekday() != day_name_to_week_number(day):
            continue

        # Format name of row by time of group name
        row_name = name.strftime("%A %d %B %Y")
        count = group["Model forecast"].sum()

        # Add the last day to the list
        temporary_data_list.append(
            [
                zone_set,
                name,
                row_name,
                name.hour + HOUR_SAMPLING,
                count,
            ]
        )

    converted_dataframe = pd.DataFrame(temporary_data_list, columns=column_names)
    return converted_dataframe


# Draw a plot with several lines (each line indicates a different category)
def draw_line_plot(dataframe: DataFrame, day="Monday") -> Figure:

    zone_set = ZONE_SET

    start_time = time.perf_counter()

    transformed_dataframe = convert_dataframe_to_displayable_dataframe(
        dataframe, day=day
    )

    print(
        f"Took {time.perf_counter() - start_time} seconds to transform data into displayable format"
    )

    start_timestamp = transformed_dataframe["date"].min()
    end_timestamp = transformed_dataframe["date"].max()

    transformed_dataframe.sort_values("date", inplace=True)

    figure = px.line(
        transformed_dataframe,
        x="hour in day",
        y="complaints",
        height=1200,
        color="name",
        markers=True,
        title=f"Compare complaint patterns in {zone_set} ({day}) from {start_timestamp} to {end_timestamp}",
        custom_data=[
            "date",
            "name",
        ],  # Provide each line with the datetime and intersection (because we require it later)
    )

    return figure


def filter_complaint_data(zone_set="North Manhattan") -> DataFrame:
    # Grab taxi data of a specific zone
    complaint_data: DataFrame = global_complaint_data.copy()

    # Sort data on datetime
    complaint_data = complaint_data.sort_values("Created Date")

    return complaint_data


def convert_filtered_complaint_data_to_lof_computable(
    dataframe: DataFrame, day="Monday"
) -> DataFrame:
    temporary_data_list = []
    column_names = [
        "Row name",
        "Date",
    ]

    day_of_the_week = day_name_to_week_number(day)

    # Add appropriate column names based on hour sampling
    for i in range(HOUR_SAMPLING, 24 + 1, HOUR_SAMPLING):
        column_names.append(f"{i}")

    resampled = dataframe.resample(
        f"{HOUR_SAMPLING}H",
        on="Created Date",
    ).sum()[["Model forecast"]]

    summed_data_per_day = resampled.groupby(
        resampled["Model forecast"].index.day_of_year
    )

    name: Timestamp
    group: DataFrame
    for day_of_year, dataframe in summed_data_per_day:
        transposed_rows = dataframe.T

        date_of_row = transposed_rows.columns[0]

        # Skip days that are not interesting
        if date_of_row.isoweekday() != day_of_the_week:
            continue

        name_of_day = date_of_row.strftime("%A %d %B %Y")

        temp_list = [
            name_of_day,
            date_of_row.date(),
        ]

        data: List = list(transposed_rows.values.reshape(-1))

        if len(data) < 6:
            data.insert(0, 0)

        temp_list.extend(data)

        temporary_data_list.append(temp_list)

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


@app.callback(
    dash.dependencies.Output("complaint-pattern-plot", "figure"),
    [
        dash.dependencies.Input("selected-day", "value"),
        dash.dependencies.Input("zoneset-selector", "value"),
        dash.dependencies.Input("number_lof_neighbors", "value"),
        dash.dependencies.Input("complaint-pattern-plot", "clickData"),
    ],
)
def selected_day_of_week(
    selected_day, selected_zoneset, number_of_neighbors, selected_lines
):
    if selected_day is None or selected_zoneset is None or number_of_neighbors is None:
        raise PreventUpdate()

    ctx = dash.callback_context

    # Check if the callback was triggered by selecting a marker
    if (
        ctx.triggered
        and ctx.triggered[0]["prop_id"] == "complaint-pattern-plot.clickData"
    ):
        # We try to highlight the line if the user selects it

        # Check if a marker was indeed selected
        if selected_lines is None:
            raise PreventUpdate()

        global global_cached_figure

        if len(selected_lines["points"]) == 0:
            global_cached_figure.update_traces(line_width=2)
            return global_cached_figure

        selected_line = selected_lines["points"][0]
        selected_outlier = selected_line["customdata"][1]

        global_cached_figure.for_each_trace(
            lambda trace: trace.update(line_width=10)
            if trace.name == selected_outlier
            else trace.update(line_width=2)
        )
        return global_cached_figure

    global NUMBER_NEIGHBORS_LOF_COMPLAINTS
    NUMBER_NEIGHBORS_LOF_COMPLAINTS = int(number_of_neighbors)

    start_time = time.perf_counter()

    # Get data of traffic
    filtered_complaint_data = filter_complaint_data(zone_set=selected_zoneset)

    print(
        f"Took {time.perf_counter() - start_time} seconds to filter complaint data for '{selected_zoneset}'"
    )

    start_time = time.perf_counter()

    # Grab outliers
    outliers = determine_outliers(filtered_complaint_data, selected_day)

    print(f"Took {time.perf_counter() - start_time} seconds to find outliers")

    start_time = time.perf_counter()

    # Determine the line plot
    complaint_plot = draw_line_plot(filtered_complaint_data, day=selected_day)

    print(
        f"Took {time.perf_counter() - start_time} seconds to draw a time pattern plot"
    )

    # Update figure so that an outlier trace has low opacity
    complaint_plot.for_each_trace(
        lambda trace: trace.update(opacity=0.1, mode="lines")
        if trace.name
        not in outliers  # Make opacity of non-outliers lower so only outlier stand out
        else ()
    )

    # Update the cached version of the figure
    global_cached_figure = complaint_plot

    return complaint_plot


# Return a list of days that are outliers (based on a LOF method)
def determine_outliers(dataframe: DataFrame, day="Monday") -> List[str]:

    converted_dataframe = convert_filtered_complaint_data_to_lof_computable(
        dataframe, day=day
    )

    # Convert data to probability density function
    vectorized_data = converted_dataframe.drop(labels=["Row name", "Date"], axis=1)

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
            outlier_name = converted_dataframe.iloc[index]["Row name"]
            outlier_names.append(outlier_name)

    return outlier_names


# Distance metric used for calculating distances between probability density functions
def distance_metric_bhattacharryya(array1, array2) -> float:
    # In this case we are using the https://en.wikipedia.org/wiki/Bhattacharyya_distance
    return -numpy.log(numpy.sum(numpy.sqrt(array1 * array2)))


# Distance metric used for calculating distances between probability density functions
def distance_metric_earth_movers(array1, array2) -> float:
    # In this case we are using the https://en.wikipedia.org/wiki/Earth_mover's_distance

    sum = 0
    previous_emd = 0

    for e1, e2 in zip(array1, array2):
        previous_emd = e1 + previous_emd - e2
        sum += abs(previous_emd)

    return sum


# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    px.set_mapbox_access_token(
        "pk.eyJ1Ijoic3RhYXJ0dmluIiwiYSI6ImNqcWRsNXRjZzQ2cDg0OG11YXpnNGh6eWQifQ._MVcI"
        "-FJIM3u0dPjaAj_lg"
    )

    print(f"\n\n --------- Importing complaint data --------- \n")
    import_complaint_data()

    app.layout = html.Div(
        children=[
            dcc.Graph(id="complaint-pattern-plot", figure={}),
            html.H4(
                "Number of neighbors for LOF",
                style={"display": "inline-block", "margin-right": 20},
            ),
            dcc.Input(
                id="number_lof_neighbors",
                type="number",
                placeholder="5",
                value=NUMBER_NEIGHBORS_LOF_COMPLAINTS,
                debounce=True,
            ),
            html.Br(),
            html.Br(),
            dcc.RadioItems(
                id="selected-day",
                options=[
                    {"label": "Monday", "value": "Monday"},
                    {"label": "Tuesday", "value": "Tuesday"},
                    {"label": "Wednesday", "value": "Wednesday"},
                    {"label": "Thursday", "value": "Thursday"},
                    {"label": "Friday", "value": "Friday"},
                    {"label": "Saturday", "value": "Saturday"},
                    {"label": "Sunday", "value": "Sunday"},
                ],
                value="Monday",
                labelStyle={"display": "inline-block"},
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

    app.run_server(debug=True, port=7025, use_reloader=False)

    print(f"Done!")
