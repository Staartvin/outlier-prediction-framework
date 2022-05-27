import datetime
import os
import re
import sys
from typing import Union, List, Sequence, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn.metrics
import torch
import tqdm
from pandas import DataFrame, Series
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import Dataset, DataLoader

"""
This file is used to run the enriched model. Like the baseline model, it predicts the traffic of the target intersection (also known as dependent intersection).
It uses input for N other source intersections (a.k.a the independent intersection).   

"""

# The target intersection for which we would like to predict the traffic
dependent_intersection = "K183"
# The lanes that we want to predict
dependent_intersection_lanes = ["U081", "U082", "U021", "U022"]

# The intersections that we use as additional input (with their respective lanes)
independent_intersections = {
    "K139": ["021", "022", "101"],
}

# Offset for each day per intersection (in regard to the dependent intersection) in steps of 5 minutes
# Each intersection has an offset for each day (0 = Monday, 6 = Sunday)
# A negative offset indicates that the dependent intersection occurs AFTER the independent intersection
offset_per_intersection: Dict[str, Dict[int, int]] = {
    "K139": {0: 0, 1: -1, 2: 0, 3: -1, 4: -1, 5: -1, 6: 0},
}

# Years to use as training data
train_years = [2018, 2019]

# Year to use for test data
test_year = 2020

# Parameters of the neural network
batch_size = 100
sequence_length = 2
number_of_features_per_element = len(independent_intersections) + 1
learning_rate = 0.001
num_hidden_features = 48
number_epochs = 5

# Scaler used for standardization - one for the train data and one for the test data
dependent_intersection_scaler_train = StandardScaler()
dependent_intersection_scaler_test = StandardScaler()

# Each independent intersection also has a scaler.
# We want to scale the data of each intersection independently, so we store a scaler for each intersection
# Key is an intersection name, value is a scaler.
independent_intersection_scalers: Dict[str, StandardScaler] = {}

# By default, the predictions are only evaluated on the ground truth of the test set.
# If you enable this boolean, it will also try to compare the predictions of the model on particular outlier dates
COMPARE_PERFORMANCE_ON_OUTLIER_DATES = False

# LOF value to use for selection of outliers
compare_to_outliers_lof = 5


# --------------------------------------- #
# This code is used to load hyperparameters from any command line arguments.
# If no command line arguments are used, we use the default hyperparameters
# --------------------------------------- #


def parse_str(num: str):
    """
    Parse a string that is expected to contain a number.
    :param num: str. the number in string.
    :return: float or int. Parsed num.
    """
    if not isinstance(num, str):  # optional - check type
        raise TypeError("num should be a str. Got {}.".format(type(num)))
    if re.compile("^\s*\d+\s*$").search(num):
        return int(num)
    if re.compile("^\s*(\d*\.\d+)|(\d+\.\d*)\s*$").search(num):
        return float(num)
    return str(num)


# Hyperparameters that we understand from the command line
keys = {
    "l=": "sequence_length",
    "b=": "batch_size",
    "l_r=": "learning_rate",
    "e=": "number_epochs",
    "h=": "num_hidden_features",
}

# Loop over input arguments
for i in range(1, len(sys.argv)):
    for key in keys:
        if sys.argv[i].find(key) == 0:
            value = sys.argv[i][len(key) :]

            print(f"Found input '{keys[key]} = {value}'")
            globals()[keys[key]] = parse_str(value)

# Let user know what hyperparameters we are going to use
print("---------------")
print(f"Input parameters:")
print(f"Intersection: {dependent_intersection}")
print(f"Using data of other intersections {list(independent_intersections.keys())}")
print(
    f"b = {batch_size}, l = {sequence_length}, l_r = {learning_rate}, |h| = {num_hidden_features}, epochs = {number_epochs}"
)
print("---------------")

# Check if we have support for cuda.
# Running on the GPU is faster
if torch.cuda.is_available():
    print(f"GPU training is available")
    device = "cuda:0"
else:
    print(f"GPU training is NOT available")
    device = "cpu"

# String used for displaying in figures
subtitle_string = f"$b={batch_size}, e={number_epochs}, l={sequence_length}, |h| ={num_hidden_features}, l_r = {learning_rate}$"


# Convert a day number to a day in the week
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


def load_intersection_data(
    years_to_load: Union[List[int], int],
    intersection: str,
    intersection_lanes: List[str],
) -> DataFrame:
    """
    Load data of an intersection for a particular date range and selected lanes
    :param years_to_load: Years to load data for
    :param intersection: Intersection to load data of
    :param intersection_lanes: The lanes of the intersection to keep
    :return: a dataframe with the selected lanes and date range
    """
    years_to_check = []

    try:
        # Check to see if year is an iterable
        years_to_check = list(iter(years_to_load))

    except TypeError:
        # Only check one year
        years_to_check = [years_to_load]

    # Load the intersection data as a feather file.
    # You will need to create it if it doesn't exist yet.
    path_to_load_from = f"input/traffic data/processed/cache/{intersection}.feather"
    intersection_data: DataFrame = pd.read_feather(
        path_to_load_from,
    )

    # Only select columns that contain data of the lanes we are interested in
    intersection_data = intersection_data.loc[:, ["date"] + intersection_lanes]

    # Keep only data of the years we want to keep
    intersection_data = intersection_data[
        intersection_data["date"].dt.year.isin(years_to_check)
    ]

    # Compute sum of all detections in the selected lanes
    intersection_data["total_detections"] = intersection_data.loc[
        :, intersection_lanes
    ].sum(axis=1)

    # Remove data per lane and only use the total instead
    intersection_data = intersection_data.loc[:, ["date", "total_detections"]]

    # Convert total detections to int type
    intersection_data["total_detections"] = intersection_data[
        "total_detections"
    ].astype(int)

    # Resample data to fill gaps in time with forward-filling
    resampled_intersection_data = (
        intersection_data.resample("5min", on="date")
        .min()
        .drop("date", axis=1)
        .reset_index()
        .fillna(method="ffill")
    )

    return resampled_intersection_data


# Define custom dataset so we can easily load our data in the model
# Each item in this dataset is a sequence of minutes (the sequence length determines how many minutes there are in one sequence)
# The label of an item is the traffic volume at the next timestamp.
# There is a particular structure to the Dataset object, as it is a PyTorch class.
class SequenceMinuteDataset(Dataset):
    def __init__(
        self,
        dependent_intersection_dataframe: DataFrame,
        independent_intersections_dataframe: DataFrame,
        offset_per_intersection: Dict[str, Dict[int, int]],
        number_of_elements_per_sequence: int,
        dependent_intersection_scaler: StandardScaler,
        timestamps_of_data_points: DataFrame,
    ):
        """
        Create dataset of traffic data
        :param dependent_intersection_dataframe: Dataframe containing traffic data of the target intersection
        :param independent_intersections_dataframe: Dataframe containing traffic of the source intersection
        :param offset_per_intersection: Dictionary containing the offsets per intersection
        :param number_of_elements_per_sequence: The length of a single sequence
        :param dependent_intersection_scaler: The scaler used for scaling the target intersection
        :param timestamps_of_data_points: The timestamps that indicate the time for each record in both dataframes. This is used to detect on what data the index falls.
        """
        # How many elements are there in a sequence?
        self.sequence_length: int = number_of_elements_per_sequence

        # Standardize data to have a mean of 0 and a variance of 1
        self.dependent_data = dependent_intersection_scaler.fit_transform(
            dependent_intersection_dataframe.values.reshape(-1, 1)
        )

        # Keep track of the data in tensors, per intersection
        self.independent_tensors: Dict[Tuple[str, int], torch.tensor] = {}

        independent_intersection_name: str
        intersection_data: Series

        # For each independent intersection, scale the data and store it locally.
        for (
            independent_intersection_name,
            intersection_data,
        ) in independent_intersections_dataframe.iteritems():
            # Build a new scaler for this intersection
            scaler_for_intersection = StandardScaler()

            # Check if we have an offset defined for the intersection
            if independent_intersection_name not in offset_per_intersection:
                print(
                    f"There is no offset defined for intersection {independent_intersection_name}!"
                )
                exit()

            # Grab offset of this intersection
            offset_for_intersection_per_day = offset_per_intersection[
                independent_intersection_name
            ]

            # Each day of the week has a different shift, so we account for that.
            for day_in_week in range(0, 7):
                # Shift the data using offset
                shifted_intersection_data = intersection_data.shift(
                    offset_for_intersection_per_day[day_in_week], fill_value=0
                )

                # Scale the data of the intersection
                independent_intersection_scaled_data = (
                    scaler_for_intersection.fit_transform(
                        shifted_intersection_data.values.reshape(-1, 1)
                    )
                )

                # Store the scaler somewhere so we can use it later on
                independent_intersection_scalers[
                    independent_intersection_name
                ] = scaler_for_intersection

                # Convert the scaled data into a tensor and store it
                self.independent_tensors[
                    (independent_intersection_name, day_in_week)
                ] = (
                    torch.tensor(independent_intersection_scaled_data.reshape(-1))
                    .float()
                    .cuda()
                )

        # Store the dependent intersection data as a tensor as well
        self.dependent_tensor: torch.tensor = (
            torch.tensor(self.dependent_data.reshape(-1)).float().cuda()
        )

        # Store date for each data record
        self.timesteps = timestamps_of_data_points

        # Store cached computation in here
        self.cached_computation = {}

        # Cache the data so we only load it once - takes a bit of time at the start
        self.cache_data()

    def __len__(self):
        return self.dependent_tensor.shape[0] - 1

    def fetch_intersection_tensor(
        self, index: int, intersection_name: str
    ) -> torch.tensor:
        """
        Fetch the tensor with traffic data for a particular index and intersection name
        :param index: Index to fetch the data of
        :param intersection_name: Name of the index
        :return: tensor containing the data of the requested index
        """
        # If the intersection is the dependent intersection, it's fairly straight-forward
        if intersection_name == dependent_intersection:
            applicable_tensor = self.dependent_tensor
        else:
            # We need to look up which day of the week it is
            date = self.timesteps[index]

            # Grab the tensor that is shifted according to the day of the week
            applicable_tensor = self.independent_tensors[
                (intersection_name, date.weekday())
            ]

        # If we have a sequence that fits in the dataset (without padding), we just return that sequence
        if index >= self.sequence_length - 1:
            index_start = index - self.sequence_length + 1
            sequence: torch.tensor = applicable_tensor[index_start : (index + 1)]
        else:
            # If the sequence does not fit (because it requests data with index < 0),
            # we pad the sequence with the first value
            padding = applicable_tensor[0].repeat(self.sequence_length - index - 1)
            # Grab sequence that is valid
            sequence: torch.tensor = applicable_tensor[0 : (index + 1)]
            # Add padding to the sequence
            sequence = torch.cat((padding, sequence), 0)

        return sequence

    def cache_data(self) -> None:
        """
        Cache data in a member variable so we can grab it later
        :return:
        """
        for index in tqdm.tqdm(
            range(0, len(self)),
            desc="Caching indices of dataset for improved training speed",
        ):
            self.cached_computation[index] = self.__getitem__(index)

    def __getitem__(self, index):
        # Check if we have a cached value for this index available
        if index in self.cached_computation:
            return self.cached_computation[index]

        # Grab tensors of target and source intersections.
        dependent_tensor = self.fetch_intersection_tensor(
            index, dependent_intersection
        ).reshape(-1, 1)
        independent_tensors = [
            self.fetch_intersection_tensor(
                index, independent_intersection_name
            ).reshape(-1, 1)
            for independent_intersection_name in independent_intersections.keys()
        ]

        # Merge all sequences together
        # We obtain a sequence of elements, where each element consists of multiple elements again
        final_sequence: torch.tensor = torch.cat(
            (dependent_tensor, *independent_tensors), dim=1
        )

        # The length of this dataset is one fewer than the number of days in the dataframe (as the last day does not
        # have a next day).
        return final_sequence, self.dependent_tensor[index + 1]


# Load the train data of the target intersection
train_data_dependent_intersection = load_intersection_data(
    train_years, dependent_intersection, intersection_lanes=dependent_intersection_lanes
)

# Load the test data of the target intersection
test_data_dependent_intersection = load_intersection_data(
    test_year, dependent_intersection, intersection_lanes=dependent_intersection_lanes
)


def load_independent_intersections(years: Union[Sequence[int], int]) -> DataFrame:
    """
    Load the data of all source intersections.
    :param years: Years to load data for
    :return: dataframe containing a column for each source intersection.
    """
    all_independent_intersection_data: Optional[DataFrame] = None

    # Iterate over every intersection that we want to add and merge them together into one single dataframe
    for (
        independent_intersection_name,
        intersection_lanes,
    ) in independent_intersections.items():
        # Grab the data of the intersection
        intersection_data = load_intersection_data(
            years, independent_intersection_name, intersection_lanes
        )

        # If there was no intersection data loaded before, we set the loaded intersection data to the current
        # intersection data
        if all_independent_intersection_data is None:
            intersection_data.rename(
                {"total_detections": f"{independent_intersection_name}"},
                axis=1,
                inplace=True,
            )
            all_independent_intersection_data = intersection_data
            continue

        # There is already loaded intersection data, so we must merge this intersection data with the loaded data.
        all_independent_intersection_data = all_independent_intersection_data.merge(
            intersection_data, on="date"
        )

        # Rename the added column to the intersection we just added
        all_independent_intersection_data.rename(
            {"total_detections": f"{independent_intersection_name}"},
            axis=1,
            inplace=True,
        )

    return all_independent_intersection_data


# Load the train and test data of the source intersections
train_data_independent_intersections = load_independent_intersections(train_years)
test_data_independent_intersections = load_independent_intersections(test_year)

# We need to merge the data (train AND test) of the target and source intersections so they have records on the same dates.
# If they do not match on the timestamps, the model is going to train on non-matching data.
all_train_data = train_data_dependent_intersection.merge(
    train_data_independent_intersections, on="date"
).rename({"total_detections": dependent_intersection}, axis=1)

all_test_data = test_data_dependent_intersection.merge(
    test_data_independent_intersections, on="date"
).rename({"total_detections": dependent_intersection}, axis=1)

# Remove columns with duplicate names.
all_train_data = all_train_data.loc[:, ~all_train_data.columns.duplicated()]
all_test_data = all_test_data.loc[:, ~all_test_data.columns.duplicated()]

# Load train dataset into PyTorch dataset object
dataset_train = SequenceMinuteDataset(
    dependent_intersection_dataframe=all_train_data[dependent_intersection],
    independent_intersections_dataframe=all_train_data[
        independent_intersections.keys()
    ],
    number_of_elements_per_sequence=sequence_length,
    offset_per_intersection=offset_per_intersection,
    dependent_intersection_scaler=dependent_intersection_scaler_train,
    timestamps_of_data_points=all_train_data["date"],
)

# Load test dataset into PyTorch dataset object
dataset_test = SequenceMinuteDataset(
    dependent_intersection_dataframe=all_test_data[dependent_intersection],
    independent_intersections_dataframe=all_test_data[independent_intersections.keys()],
    number_of_elements_per_sequence=sequence_length,
    offset_per_intersection=offset_per_intersection,
    dependent_intersection_scaler=dependent_intersection_scaler_test,
    timestamps_of_data_points=all_test_data["date"],
)

# Make a train and test dataloader (from Pytorch)
# We shuffle the train dataset so the model does not learn the data in chronological order. We want it to predict data
# without know the date of the year.
loader_train_set = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)

# Since the model we be used to predict the data in sequence, we want the test data to be in sequence (hence shuffle=false).
loader_test_set = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)


# Set up model for LSTM
class SampleLSTM(nn.Module):
    def __init__(
        self,
        sequence_length: int,
        num_features_in_hidden_state: int,
        num_features_per_element: int,
    ):
        super().__init__()
        self.sequence_length = sequence_length
        self.num_features_per_element = num_features_per_element
        self.num_features_in_hidden_state = num_features_in_hidden_state

        self.lstm = nn.LSTM(
            input_size=self.num_features_per_element,
            hidden_size=num_features_in_hidden_state,
            batch_first=True,
            num_layers=1,
        )

        self.linear = nn.Linear(
            in_features=self.num_features_in_hidden_state, out_features=1
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size = x.shape[0]

        output, (hn, cn) = self.lstm(
            x.reshape(batch_size, self.sequence_length, self.num_features_per_element)
        )

        out = self.linear(
            hn[0]
        ).flatten()  # First dim of Hn is num_layers, which is set to 1 above.

        return out


# ------------ Define the model -------------

model = SampleLSTM(
    sequence_length=sequence_length,
    num_features_in_hidden_state=num_hidden_features,
    num_features_per_element=number_of_features_per_element,
)
model.to(device)

# Define a loss function and move the model to the correct device (either CPU or GPU)
loss_function = nn.MSELoss()
loss_function.to(device)

# Define an optimizer (Adam seems to work fine)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)


# Train the model on the input
def train_model(data_loader, model, loss_function, optimizer) -> float:
    num_batches = len(data_loader)
    total_loss = 0
    model.train()

    for X, y in data_loader:
        output = model(X)
        loss = loss_function(output, y.float().cuda())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / num_batches

    return avg_loss


# Test the model to see how it's working on the test data
def test_model(data_loader, model, loss_function) -> float:
    num_batches = len(data_loader)
    total_loss = 0

    model.eval()
    with torch.no_grad():
        for X, y in data_loader:
            output = model(X)
            total_loss += loss_function(output, y.cuda()).item()

    avg_loss = total_loss / num_batches

    return avg_loss


# Store the loss per epoch
train_loss_per_epoch = []
test_loss_per_epoch = []

# Do some training for a few epochs
for epoch_index in tqdm.tqdm(range(number_epochs), desc="Training model"):
    train_loss = train_model(
        loader_train_set, model, loss_function, optimizer=optimizer
    )
    test_loss = test_model(loader_test_set, model, loss_function)

    train_loss_per_epoch.append(train_loss)
    test_loss_per_epoch.append(test_loss)

# Make sure to use latex math formatting, so it shows nicely in the figure
plt.rcParams.update({"text.usetex": True, "font.family": "Helvetica"})
# Make sure all labels fit on the drawing
plt.rcParams["figure.constrained_layout.use"] = True
# Plot loss over time for epoch
plt.plot(
    train_loss_per_epoch,
    label="Train loss",
)
plt.plot(
    test_loss_per_epoch,
    label="Test loss",
)
plt.margins(x=0)

# Rotate the ticks so they are more readable
plt.xticks(rotation=45, ha="right")
plt.xlabel("Epoch")
plt.ylabel("Loss")

# Set the super title and title
plt.suptitle(f"Enriched - Loss over time on {train_years} + {test_year}")
plt.title(
    subtitle_string,
    fontsize=10,
)

plt.legend()
plt.show()


# Predict the unseen data using a data loader and model
def predict(data_loader, model):
    output = torch.tensor([]).cuda()
    model.eval()
    with torch.no_grad():
        for X, _ in data_loader:
            y_star = model(X)
            output = torch.cat((output, y_star), 0)

    return output


# Name of the column used for storing the forecast of the model
forecast_column_name = "Model forecast"

# Make a dataframe that stores the test data and the predicted data
test_data_converted_to_forecasted = test_data_dependent_intersection.reset_index(
    drop=True
)

# Add the predictions to the dataframe
test_data_converted_to_forecasted[forecast_column_name] = pd.Series(
    data=predict(loader_test_set, model).cpu().numpy(),
)
# Remove NA values
test_data_converted_to_forecasted.dropna(inplace=True)
# Make sure to transform the standardized output back to non-standardized output so we can compare it
test_data_converted_to_forecasted[
    forecast_column_name
] = dependent_intersection_scaler_test.inverse_transform(
    test_data_converted_to_forecasted[[forecast_column_name]]
)

# Build dataframe that shows the label and the predicted label
dataframe_to_compare = test_data_converted_to_forecasted[
    ["date", "total_detections", forecast_column_name]
]


# Draw predictions and ground truth to be able to compare them
def plot_prediction_results(predictions, ground_truth, timestamps):
    MSE = sklearn.metrics.mean_squared_error(ground_truth, predictions)
    RMSE = np.sqrt(MSE)
    R2 = sklearn.metrics.r2_score(
        ground_truth, predictions, multioutput="variance_weighted"
    )
    MAE = sklearn.metrics.mean_absolute_error(ground_truth, predictions)
    MEDAE = sklearn.metrics.median_absolute_error(ground_truth, predictions)
    mean_prediction_value = Series.mean(predictions)
    stdev_prediction_value = Series.std(
        predictions, ddof=0
    )  # Obtain the population stdev (not sample)
    print(f"Average forecast: {mean_prediction_value}")
    print(f"Population stdev: {stdev_prediction_value}")
    print(f"RMSE of model: {RMSE}")
    print(f"R^2 of model: {R2}")
    print(f"MAE: {MAE}")
    print(f"MEDAE: {MEDAE}")

    sns.kdeplot(ground_truth, shade=True, label="Ground truth")
    sns.kdeplot(predictions, shade=True, label="Predictions")
    plt.suptitle(
        f"Enriched - Distribution of predictions vs ground truth - {dependent_intersection}"
    )
    plt.title(
        subtitle_string,
        fontsize=10,
    )
    plt.legend()
    plt.show()

    sns.histplot(abs(ground_truth - predictions))
    plt.title("Enriched - distribution of error")
    plt.show()

    timestamps_part = timestamps.loc[:6000]
    ground_truth_part = ground_truth.loc[:6000]
    predictions_part = predictions.loc[:6000]
    plt.figure(figsize=(24, 12))
    # Plot ground truth values vs. prediction
    plt.plot(
        timestamps_part, ground_truth_part.reset_index(drop=True), label="Ground truth"
    )
    plt.plot(
        timestamps_part, predictions_part.reset_index(drop=True), label="Prediction"
    )

    plt.xticks(rotation=45, ha="right")
    plt.title("Predictions and ground truth (partly)")
    plt.show()

    return RMSE, MAE, MEDAE


# EVALUATION

print(f"Train dataset ({train_years}) has {len(dataset_train)} items")
print(f"Test dataset ({test_year}) has {len(dataset_test)} items")

print(f"-----------")
print(f"Enriched - evaluation for test data")

filtered_predictions = dataframe_to_compare[
    dataframe_to_compare["date"].dt.year.isin([test_year])
].copy()

# Grab the predictions, the ground truth and the timestamps
predicted_values_filtered = filtered_predictions[forecast_column_name]
ground_truth_filtered = filtered_predictions["total_detections"]
timestamps_filtered = filtered_predictions["date"]

# Compute the results
results = plot_prediction_results(
    predicted_values_filtered,
    ground_truth_filtered,
    timestamps_filtered,
    title=f"Predictions of test year {test_year}",
)
print(f"-----------")

# ---------
# Write output of predictions to file
output_dataframe = test_data_converted_to_forecasted.copy()[
    ["date", forecast_column_name]
]

# Round predictions to whole number
output_dataframe[forecast_column_name] = (
    output_dataframe[forecast_column_name].astype("float").round(0).astype(int)
)

path_to_output_to = f"output/prediction/enriched/traffic/{test_year}/{dependent_intersection}/traffic_prediction.csv"

# Create directories if they do not exist yet
os.makedirs(os.path.dirname(path_to_output_to), exist_ok=True)

# Write predicted data to a CSV
output_dataframe.to_csv(
    path_to_output_to,
    sep=";",
    index=False,
    decimal=",",
)

# ---------
# We also want to determine the quality of the predictions at the position of the outliers
# So, what is the difference in prediction vs ground truth only at the dates that we predict are outliers
#
if COMPARE_PERFORMANCE_ON_OUTLIER_DATES:
    outlier_dates = []

    print(
        f"Grabbing outliers from {test_year} to compare the performance at outlier points.."
    )
    for selected_day in [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]:
        begin_index_minute = 0
        end_index_minute = 1440
        step_minute = 30

        for time_interval in np.arange(
            begin_index_minute, end_index_minute, step_minute
        ):
            time_interval = int(time_interval)

            selected_time_window_begin = (
                datetime.datetime.combine(
                    datetime.date.today(), datetime.time(hour=0, minute=0)
                )
                + datetime.timedelta(minutes=time_interval)
            ).time()
            selected_time_window_end = (
                datetime.datetime.combine(
                    datetime.date.today(), datetime.time(hour=0, minute=0)
                )
                + datetime.timedelta(minutes=time_interval + step_minute)
            ).time()

            if time_interval == (end_index_minute - step_minute):
                selected_time_window_end = datetime.time(hour=23, minute=59)

            outlier_file = (
                f"output/outliers/traffic/{test_year}/{dependent_intersection}/{selected_day}/time_window{selected_time_window_begin.hour}"
                f"-{selected_time_window_begin.minute}_{selected_time_window_end.hour}-{selected_time_window_end.minute}"
                f"_outliers_lof{compare_to_outliers_lof}.feather"
            )

            # Load outliers as determined by the LOF algorithm
            determined_outliers_dataframe = pd.read_feather(outlier_file)

            # Filter only datetime that are outliers
            determined_outliers_dataframe = determined_outliers_dataframe[
                determined_outliers_dataframe["Is outlier"] == 1
            ]

            # Store outlier dates
            outlier_dates.append(determined_outliers_dataframe["Date"])

    # Make a series of the outlier dates
    outlier_dates_in_series = pd.concat(outlier_dates).reset_index(drop=True)

    # Only keep the datapoints that are at the outlier dates
    mask = filtered_predictions["date"].isin(outlier_dates_in_series)

    # Grab the predictions that are made at the outlier dates
    data_at_outliers = filtered_predictions[mask]

    print(f"-----------")
    print(f"Enriched - evaluation for test data at outlier points")

    predicted_values_at_outliers = data_at_outliers[forecast_column_name]
    ground_truth_at_outliers = data_at_outliers["total_detections"]
    timestamps_at_outliers = data_at_outliers["date"]

    plot_prediction_results(
        predicted_values_at_outliers,
        ground_truth_at_outliers,
        timestamps_at_outliers,
        title=f"Predictions of enriched at outlier dates",
    )
    print(f"-----------")
