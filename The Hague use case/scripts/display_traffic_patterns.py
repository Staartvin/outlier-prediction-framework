import datetime
import glob
import os
import time
import warnings
from typing import List, Dict, Tuple

import dash
import numpy
import numpy as np
import pandas as pd
import plotly.express as px
from dash import html, dcc
from dash.exceptions import PreventUpdate
from pandas import DataFrame
from plotly.graph_objs import Figure
from sklearn.neighbors import LocalOutlierFactor

"""
This file launches a dash server that allows you to visualize outliers of a traffic intersection. It uses dash and plotly to show interactive visualizations.
"""


app = dash.Dash(__name__)

# Keep reference of traffic data for each intersection (key = intersection, value = dataframe)
traffic_data_per_intersection: Dict[str, DataFrame] = dict()

# Location of each intersection
# key is intersection name, value is pair of coordinates (lat, long)
intersection_coordinates: Dict[str, Tuple[float, float]] = dict()

# Store a cached version of the figure if we don't want to redraw
global_cached_figure = None

# Number of neighbors to use during the LOF calculation
NUMBER_NEIGHBORS_LOF = 5

YEAR_SELECTOR = 2020

# Whether we should convert the data to a probability density function before applying LOF
TRANSFORM_TO_PDF_BEFORE_DETECTING_OUTLIERS = False

# Which distance metric to use for LOF outlier detection
LOF_METRIC = "chebyshev"

HUMAN_FRIENDLY_OUTLIER_DATE_FORMAT = "%A %d %B, %Y"

SAVE_OUTLIERS = True

# In what window should we compare data to detect outliers
OUTLIER_TIME_WINDOW = datetime.timedelta(minutes=30)

# Whether we should generate the outliers of each interval of the day, or only generate the outliers for the selected interval
GENERATE_OUTLIERS_FOR_ALL_INTERVALS_OF_DAY = True


# Import all traffic data from the csvs
def import_traffic_data():
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

    # Read coordinates of intersections
    intersection_coord_data = pd.read_csv("input/traffic data/intersection_coordinates.csv", sep=";")

    global intersection_coordinates

    # Load coordinates of intersections
    for row in intersection_coord_data.itertuples():
        intersection_coordinates[row.intersection_name] = (float(row.latitude), float(row.longitude))


def convert_dataframe_to_displayable_dataframe(
        dataframe: DataFrame, day="Monday"
) -> DataFrame:
    converted_dataframe = dataframe.copy()
    converted_dataframe.reset_index(drop=False, inplace=True)
    converted_dataframe["window in hour"] = converted_dataframe["date"].dt.time
    converted_dataframe["date name"] = converted_dataframe["date"].dt.strftime(HUMAN_FRIENDLY_OUTLIER_DATE_FORMAT)
    converted_dataframe["detections"] = converted_dataframe.sum(numeric_only=True, axis=1)

    return converted_dataframe


# Draw a plot with several lines (each line indicates a different category)
def draw_line_plot(intersection_data: DataFrame, day: str = "Monday", time_window_begin: datetime.time = None,
                   time_window_end: datetime.time = None) -> Figure:
    if time_window_begin is None:
        time_window_begin = datetime.time(hour=0, minute=0)
    if time_window_end is None:
        time_window_end = (
                datetime.datetime.combine(datetime.date.today(), time_window_begin) + OUTLIER_TIME_WINDOW).time()

    # Filter intersection data on day and hour
    day_name_mask = intersection_data["date"].dt.day_of_week == day_name_to_day_of_week_number(day)

    # Grab rows that are on the specific day and hour
    intersection_data = intersection_data.loc[day_name_mask].copy()

    intersection_data = intersection_data.set_index("date")
    intersection_data = intersection_data.between_time(time_window_begin, time_window_end, inclusive="left")

    start_time = time.perf_counter()

    transformed_dataframe = convert_dataframe_to_displayable_dataframe(
        intersection_data, day=day
    )

    print(
        f"Took {time.perf_counter() - start_time} seconds to transform data into displayable format"
    )

    intersection_name = intersection_data["intersection_name"].iat[0]

    start_timestamp = transformed_dataframe["date"].min()
    end_timestamp = transformed_dataframe["date"].max()

    transformed_dataframe.sort_values("date", inplace=True, ascending=False)

    figure = px.line(
        transformed_dataframe,
        x="window in hour",
        y="detections",
        height=1200,
        color="date name",
        markers=True,
        title=f"Compare traffic patterns in {intersection_name} ({day}) from {start_timestamp} to {end_timestamp}",
        custom_data=[
            "date",
            "date name",
        ],  # Provide each line with the datetime and intersection (because we require it later)
    )

    return figure


def select_intersection_data(intersection=None) -> DataFrame:
    intersection_data: DataFrame = traffic_data_per_intersection[intersection]

    # Select only data from the year that we want to see
    intersection_data = intersection_data[intersection_data["date"].dt.year == YEAR_SELECTOR]

    return intersection_data


def select_lanes_of_intersection(intersection_data: DataFrame, lanes: List[str] = None):
    if lanes is None or len(lanes) == 0:
        return intersection_data

    # Only select columns of lanes we want to keep (and the date and intersection name)
    filtered_intersection_data = intersection_data.loc[:, ["date", "intersection_name"] + lanes]

    return filtered_intersection_data


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
        dataframe: DataFrame, time_window_begin: datetime.time = None, time_window_end: datetime.time = None
) -> DataFrame:
    temporary_data_list = []
    column_names = [
        "date",
    ]

    current_time = time_window_begin

    while current_time < time_window_end:
        next_time = (datetime.datetime.combine(datetime.date.today(), current_time) + datetime.timedelta(
            minutes=5)).time()

        column_names.append(f"{current_time.minute} - {next_time.minute}")

        if next_time < current_time:
            break

        current_time = next_time

    # Compute sum of all detections in the selected lanes
    dataframe["total_detections"] = dataframe.sum(numeric_only=True, axis=1)

    # Drop the intersection name
    dataframe.drop("intersection_name", inplace=True, axis=1)

    grouped_data = dataframe.groupby(by=dataframe.index.date)

    for name, group in grouped_data:
        transposed_group = group.transpose()

        filtered_row = transposed_group[(transposed_group.index == "total_detections")]

        date = filtered_row.columns[0].date()

        # Generate points that are zero from start
        list_of_row = [0] * (len(column_names) - 1)

        if len(list_of_row) == 0:
            continue

        # Loop over the number of detections per 5 minutes and put them in the right slot in the list
        for index, col_name in enumerate(filtered_row):
            date_of_detection = col_name
            number_of_detections = filtered_row[col_name][0]

            list_of_row[int((date_of_detection.minute - time_window_begin.minute) / 5)] = number_of_detections

        # Insert date at the beginning of the list
        list_of_row.insert(0, date)

        # Make sure to check whether the list has the correct length
        assert len(list_of_row) == len(column_names)

        temporary_data_list.append(list_of_row)

    converted_dataframe = pd.DataFrame(temporary_data_list, columns=column_names)

    return converted_dataframe


@app.callback(
    dash.dependencies.Output("traffic-pattern-plot", "figure"),
    [
        dash.dependencies.Input("selected-day", "value"),
        dash.dependencies.Input("intersection-selector", "value"),
        dash.dependencies.Input("hour-selector", "value"),
        dash.dependencies.Input("number_lof_neighbors", "value"),
        dash.dependencies.Input("traffic-pattern-plot", "clickData"),
    ],
)
def selected_day_of_week(
        selected_day: str, selected_intersection: str, selected_time_window: str, number_of_neighbors: str,
        selected_lines
):
    if selected_day is None or selected_intersection is None or number_of_neighbors is None or selected_time_window is None:
        raise PreventUpdate()

    ctx = dash.callback_context

    # Check if the callback was triggered by selecting a marker
    if (
            ctx.triggered
            and ctx.triggered[0]["prop_id"] == "traffic-pattern-plot.clickData"
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

    global NUMBER_NEIGHBORS_LOF
    NUMBER_NEIGHBORS_LOF = int(number_of_neighbors)

    start_time = time.perf_counter()

    # Get data of traffic
    intersection_data = select_intersection_data(intersection=selected_intersection)

    # Select proper lanes of intersection
    if selected_intersection == "K559":
        intersection_data = select_lanes_of_intersection(intersection_data, ["u02_1", "u02_2"])
    elif selected_intersection == "K557":
        intersection_data = select_lanes_of_intersection(intersection_data, ["02_5"])
    else:
        warnings.warn(f"Intersection {selected_intersection} has no lane filters!")

    print(
        f"Took {time.perf_counter() - start_time} seconds to select data for '{selected_intersection}'"
    )

    selected_time_window_begin = (datetime.datetime.combine(datetime.date.today(), datetime.time(hour=0, minute=0)) +
                                  datetime.timedelta(minutes=float(selected_time_window.split(" - ")[0]) * 60)).time()
    selected_time_window_end = (datetime.datetime.combine(datetime.date.today(), datetime.time(hour=0, minute=0)) +
                                datetime.timedelta(minutes=float(selected_time_window.split(" - ")[1]) * 60)).time()

    start_time = time.perf_counter()

    # Grab outliers
    outliers = determine_outliers(intersection_data, selected_day, time_window_begin=selected_time_window_begin,
                                  time_window_end=selected_time_window_end)

    outliers_human_friendly_names = [outlier.strftime(HUMAN_FRIENDLY_OUTLIER_DATE_FORMAT) for outlier in outliers]

    print(f"Took {time.perf_counter() - start_time} seconds to find outliers")

    start_time = time.perf_counter()

    # Determine the line plot
    pattern_plot = draw_line_plot(intersection_data, day=selected_day, time_window_begin=selected_time_window_begin,
                                  time_window_end=selected_time_window_end)

    print(
        f"Took {time.perf_counter() - start_time} seconds to draw a time pattern plot"
    )

    # Update figure so that an outlier trace has low opacity
    pattern_plot.for_each_trace(
        lambda trace: trace.update(opacity=0.1, mode="lines")
        if trace.name
           not in outliers_human_friendly_names  # Make opacity of non-outliers lower so only outlier stand out
        else ()
    )

    # Update the cached version of the figure
    global_cached_figure = pattern_plot

    if SAVE_OUTLIERS:
        save_outliers(outliers, selected_intersection, day=selected_day, time_window_begin=selected_time_window_begin,
                      time_window_end=selected_time_window_end)

    print("Showing plot now!")

    return pattern_plot


def save_outliers(outliers_dates: List[datetime.datetime], intersection: str, day: str,
                  time_window_begin: datetime.time, time_window_end: datetime.time):
    dataframe_list = []

    time_window_size: datetime.timedelta = datetime.datetime.combine(datetime.datetime.now(), time_window_end) \
                                           - datetime.datetime.combine(datetime.datetime.now(), time_window_begin)

    start_date = datetime.datetime(YEAR_SELECTOR, 1, 1, hour=0, minute=0)
    end_date = datetime.datetime(YEAR_SELECTOR, 12, 31, hour=23, minute=59)

    while start_date <= end_date:
        is_outlier = start_date in outliers_dates

        dataframe_list.append([start_date, 1 if is_outlier else 0])

        start_date += time_window_size

    outliers_in_dataframe: DataFrame = pd.DataFrame(
        data=dataframe_list,
        columns=["Date", "Is outlier"],
    )

    outlier_file_path = f"output/outliers/traffic/{YEAR_SELECTOR}/{intersection}/{day}/time_window{time_window_begin.hour}" \
                        f"-{time_window_begin.minute}_{time_window_end.hour}-{time_window_end.minute}" \
                        f"_outliers_lof{NUMBER_NEIGHBORS_LOF}.feather"

    # Create dirs if they do not exist
    os.makedirs(os.path.dirname(outlier_file_path), exist_ok=True)

    if outliers_in_dataframe.empty:
        open(outlier_file_path, "w").close()
    else:
        # Write outliers to a file
        outliers_in_dataframe.to_feather(outlier_file_path, compression="lz4")

    print(f"Wrote outliers of {intersection} to {day} ({time_window_begin}  - {time_window_end}) to a CSV file")


# Return a list of days that are outliers (based on a LOF method)
def determine_outliers(intersection_data: DataFrame, day: str = "Monday", time_window_begin: datetime.time = None,
                       time_window_end: datetime.time = None) -> List[datetime.datetime]:
    if time_window_begin is None:
        time_window_begin = datetime.time(hour=0, minute=0)
    if time_window_end is None:
        time_window_end = (
                datetime.datetime.combine(datetime.date.today(), time_window_begin) + OUTLIER_TIME_WINDOW).time()

    # Filter intersection data on day and hour
    day_name_mask = intersection_data["date"].dt.day_of_week == day_name_to_day_of_week_number(day)

    # Grab rows that are on the specific day and hour
    intersection_data = intersection_data.loc[day_name_mask].copy()

    intersection_data = intersection_data.set_index("date")
    intersection_data = intersection_data.between_time(time_window_begin, time_window_end, inclusive="left")

    converted_dataframe = convert_intersection_data_to_lof_computable(
        intersection_data, time_window_begin, time_window_end
    )

    # Convert data to probability density function
    vectorized_data = converted_dataframe.drop(labels=["date"], axis=1)

    if TRANSFORM_TO_PDF_BEFORE_DETECTING_OUTLIERS:
        vectorized_sum = vectorized_data.sum(axis=1)
        vectorized_data = vectorized_data.div(vectorized_sum, axis=0)

    # Perform LOF to detect outlier time patterns
    outlier_detector = LocalOutlierFactor(
        n_neighbors=NUMBER_NEIGHBORS_LOF,
        metric=LOF_METRIC,
    )

    # Store reference of outliers.
    outlier_names = []

    if vectorized_data.shape[0] == 0:
        return outlier_names

    # Determine whether a time series is an outlier or not
    outlier_scores = outlier_detector.fit_predict(vectorized_data)

    # Loop over outlier scores and see which ones are outliers
    for index, outlier_score in enumerate(outlier_scores):
        # We have an outlier!
        if outlier_score == -1:
            outlier_name = converted_dataframe.iloc[index]["date"]
            outlier_name = datetime.datetime.combine(outlier_name, time_window_begin)
            outlier_names.append(outlier_name)

    return outlier_names


# Distance metric used for calculating distances between probability density functions
def distance_metric_bhattacharryya(array1, array2) -> float:
    # In this case we are using the https://en.wikipedia.org/wiki/Bhattacharyya_distance
    return -numpy.log(numpy.sum(numpy.sqrt(array1 * array2)))


# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    px.set_mapbox_access_token(
        "pk.eyJ1Ijoic3RhYXJ0dmluIiwiYSI6ImNqcWRsNXRjZzQ2cDg0OG11YXpnNGh6eWQifQ._MVcI"
        "-FJIM3u0dPjaAj_lg"
    )

    print(f"\n\n --------- Importing traffic data --------- \n")
    import_traffic_data()

    app.layout = html.Div(
        children=[
            dcc.Graph(id="traffic-pattern-plot", figure={}),
            html.H4(
                "Number of neighbors for LOF",
                style={"display": "inline-block", "margin-right": 20},
            ),
            dcc.Input(
                id="number_lof_neighbors",
                type="number",
                placeholder="5",
                value=NUMBER_NEIGHBORS_LOF,
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
            dcc.Dropdown(
                id="intersection-selector",
                options=[
                    {
                        "label": intersection_location,
                        "value": intersection_location,
                    } for intersection_location in intersection_coordinates.keys()
                ],
                value="K559",
            ),
            html.Br(),
            html.Br(),
            dcc.Dropdown(
                id="hour-selector",
                options=[
                    {
                        "label": f"{hour} - {hour + 0.5}",
                        "value": f"{hour} - {hour + 0.5}",
                    } for hour in np.arange(0, 24, 0.5)
                ],
                value="0.0 - 0.5",
            ),
            html.Br(),
            html.Br(),
        ]
    )

    if GENERATE_OUTLIERS_FOR_ALL_INTERVALS_OF_DAY:
        selected_intersection = "K159"

        for selected_day in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]:
            # Get data of traffic
            intersection_data = select_intersection_data(intersection=selected_intersection)

            # Select proper lanes of intersection
            if selected_intersection == "K159":
                intersection_data = select_lanes_of_intersection(intersection_data, ["022", "082"])
            elif selected_intersection == "K557":
                intersection_data = select_lanes_of_intersection(intersection_data, ["02_5"])
            elif selected_intersection == "K504":
                intersection_data = select_lanes_of_intersection(intersection_data, ["02_5"])
            elif selected_intersection == "K206":
                intersection_data = select_lanes_of_intersection(intersection_data, ["S021"])
            elif selected_intersection == "K051":
                intersection_data = select_lanes_of_intersection(intersection_data, ["022", "021"])
            elif selected_intersection == "K101":
                intersection_data = select_lanes_of_intersection(intersection_data, ["u621", "u622"])
            else:
                warnings.warn(f"Intersection {selected_intersection} has no lane filters!")

            begin_index_minute = 0
            end_index_minute = 1440
            step_minute = 30

            for time_interval in np.arange(begin_index_minute, end_index_minute, step_minute):

                time_interval = int(time_interval)

                selected_time_window_begin = (
                        datetime.datetime.combine(datetime.date.today(), datetime.time(hour=0, minute=0)) +
                        datetime.timedelta(minutes=time_interval)).time()
                selected_time_window_end = (
                        datetime.datetime.combine(datetime.date.today(), datetime.time(hour=0, minute=0)) +
                        datetime.timedelta(minutes=time_interval + step_minute)).time()

                if (time_interval == (end_index_minute - step_minute)):
                    selected_time_window_end = datetime.time(hour=23, minute=59)

                outliers = determine_outliers(intersection_data, selected_day, selected_time_window_begin,
                                              selected_time_window_end)
                if SAVE_OUTLIERS:
                    save_outliers(outliers, selected_intersection, day=selected_day,
                                  time_window_begin=selected_time_window_begin,
                                  time_window_end=selected_time_window_end)

    app.run_server(debug=True, port=12346, use_reloader=False)

    print(f"Done!")
