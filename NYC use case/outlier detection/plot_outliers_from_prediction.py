import os
import time
from datetime import date, timedelta
from typing import List

import dash
import numpy
import numpy as np
import pandas as pd
import plotly.express as px
from dash import html, dcc
from dash.exceptions import PreventUpdate
from pandas import DataFrame, Timestamp
from sklearn.neighbors import LocalOutlierFactor

"""
This file is used to display outliers in an interactive visualization using Dash and plotly. It shows the outliers of complaints and traffic from the predictions.
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

# How many hours do we want in one bin of data?
HOUR_SAMPLING = 4

# Whether we should convert the data to a probability density function before applying LOF
TRANSFORM_TO_PDF_BEFORE_DETECTING_OUTLIERS = False

# Which distance metric to use for LOF outlier detection
LOF_METRIC = "chebyshev"

# Which year to use for selecting complaints
YEAR_SELECTOR = 2018

# Whether to use the baseline or non-baseline model data
USE_NON_BASELINE_MODEL = True

BATCH_SIZE = 30
SEQUENCE_SIZE = 6

SAVE_COMPLAINT_OUTLIERS_TO_FILE = False

# Import all complaint data from the csvs
def import_complaint_data():
    global global_complaint_data

    start_time = time.perf_counter()

    predicted_complaint_data = []

    for zone_set in [
        "Lower South Manhattan",
        "Upper South Manhattan",
        "Lower Middle Manhattan",
        "Upper Middle Manhattan",
        "North Manhattan",
    ]:
        try:
            data = pd.read_csv(
                f"output/prediction/{'baseline' if not USE_NON_BASELINE_MODEL else 'non-baseline'}/complaints/{YEAR_SELECTOR}/{zone_set}/complaints_batch{BATCH_SIZE}_sequence{SEQUENCE_SIZE}.csv",
                sep=";",
            )
        except FileNotFoundError:
            print(
                f"Could not find prediction of complaints for {YEAR_SELECTOR} in zone {zone_set}."
            )
            continue

        data["Created Date"] = pd.to_datetime(
            data["Created Date"], format="%Y-%m-%d %H:%M:%S"
        )

        data["zone set"] = zone_set

        # Store dataframe in a temporary list
        predicted_complaint_data.append(data)

    # Create complaint dataframe from all small subsets
    global_complaint_data = pd.concat(predicted_complaint_data)

    print(f"Read all complaint data in {time.perf_counter() - start_time} seconds")


def filter_complaint_data(zone_set="North Manhattan") -> DataFrame:
    # Grab complaint data of a specific zone
    complaint_data: DataFrame = global_complaint_data[
        global_complaint_data["zone set"] == zone_set
    ].copy()

    # Sort data on datetime
    complaint_data = complaint_data.sort_values("Created Date")

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
        on="Created Date",
        origin="start_day",
    ).sum()[["Model forecast"]]

    summed_data_per_day = resampled.groupby(
        resampled["Model forecast"].index.day_of_year
    )

    name: Timestamp
    group: DataFrame
    for day_of_year, dataframe in summed_data_per_day:

        transposed_rows = dataframe.T

        date_of_row = transposed_rows.columns[0]
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


@app.callback(
    [
        dash.dependencies.Output("outliers-plot", "figure"),
        dash.dependencies.Output("outliers-plot2", "figure"),
    ],
    [
        dash.dependencies.Input("zoneset-selector", "value"),
        dash.dependencies.Input("number_lof_neighbors_complaints", "value"),
    ],
)
def adjusted_input_parameters(selected_zoneset, number_of_neighbors_complaints):
    if selected_zoneset is None or number_of_neighbors_complaints is None:
        raise PreventUpdate()

    global NUMBER_NEIGHBORS_LOF_COMPLAINTS
    NUMBER_NEIGHBORS_LOF_COMPLAINTS = int(number_of_neighbors_complaints)

    # Get data of complaints
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

        data_of_day = converted_dataframe[
            converted_dataframe["Row name"].str.contains(day)
        ].copy()

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
        outliers_all_year = compute_complaint_outliers(data_of_day, day)
        complaint_outliers.extend(outliers_all_year)

    print(
        f"Took {time.perf_counter() - start_time} seconds to find outliers for complaint data"
    )

    # # Update the cached version of the figure
    #     # global_cached_figure = complaint_plot

    start_time = time.perf_counter()

    dataframe_list = []

    start_date = date(YEAR_SELECTOR, 1, 1)
    end_date = date(YEAR_SELECTOR, 12, 31)

    while start_date <= end_date:

        date_to_string = start_date.strftime("%A %d %B %Y")
        is_outlier_complaint = date_to_string in complaint_outliers

        if is_outlier_complaint:
            dataframe_list.append([date_to_string, "complaint", start_date, 1])
        else:
            dataframe_list.append([date_to_string, "complaint", start_date, 0])

        start_date += timedelta(days=1)

    print(f"Took {time.perf_counter() - start_time} seconds to compute the dataframe")

    outliers_in_dataframe: DataFrame = pd.DataFrame(
        data=dataframe_list,
        columns=["Outlier", "Type", "Date", "Value"],
    )

    line_plot = px.line(
        outliers_in_dataframe,
        x="Date",
        y="Value",
        color="Type",
        color_discrete_map={"complaint": "orange", "traffic": "blue"},
        title=f"Timeline of outliers of complaints and traffic in {YEAR_SELECTOR}",
        line_shape="spline",
        hover_name="Outlier",
    )

    violin_plot = px.violin(
        outliers_in_dataframe[outliers_in_dataframe["Value"] == 1],
        x="Date",
        color="Type",
        color_discrete_map={"complaint": "orange", "traffic": "blue"},
        title=f"Outliers of complaints and traffic patterns in {YEAR_SELECTOR}",
        points="all",
        hover_name="Outlier",
    )

    print("Done computing!")

    path_to_save_complaints = f"output/outliers/complaints/{YEAR_SELECTOR}/{selected_zoneset}/predicted_outliers_lof{number_of_neighbors_complaints}.csv"

    # Create dirs if they do not exist
    os.makedirs(os.path.dirname(path_to_save_complaints), exist_ok=True)

    if SAVE_COMPLAINT_OUTLIERS_TO_FILE:
        print(f"Saving complaint outliers to a file")
        # Write complaint outliers to a file
        outliers_in_dataframe[outliers_in_dataframe["Type"] == "complaint"].to_csv(
            path_to_save_complaints,
            sep=";",
            index=False,
            decimal=",",
        )

    return line_plot, violin_plot


# Return a list of days that are outliers (based on a LOF method)
def compute_complaint_outliers(dataframe: DataFrame, day: str = "Monday") -> List[str]:
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
    outlier_scores = outlier_detector.fit_predict(
        vectorized_data.reset_index(drop=True)
    )

    df_to_display = dataframe.copy()
    df_to_display["Outlier score"] = outlier_detector.negative_outlier_factor_

    print(f"Found {len(np.where(outlier_scores == -1)[0])} outliers for {day}.")

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

    app.run_server(debug=True, port=5679, use_reloader=False)

    print(f"Done!")
