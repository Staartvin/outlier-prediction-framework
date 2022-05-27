import datetime
import random
import re
import sys
from typing import Union, List

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn.metrics
import torch
import tqdm
from pandas import DataFrame, Timestamp, Series, Timedelta
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import Dataset, DataLoader

"""
This file is used to generate predictions for complaints based on historical data of a certain zone and traffic data of the same zone. It uses PyTorch to generate a model and train it.
"""

zone = "Lower South Manhattan"
compare_to_outliers_lof = 40

train_years = [2015, 2016, 2017]
test_year = 2018

# How many days should we shift the traffic data to?
# A positive integer will shift the traffic down, thus providing complaints with historical traffic data
# A negative integer will shift the traffic upward, providing complaints with future traffic data
traffic_offset = +5

# Shifts used in the thesis
# North Manhattan -> Offset = 7 days
# Upper Middle Manhattan -> Offset = 4 days
# Lower Middle Manhattan -> Offset = 6 days
# Upper South Manhattan -> Offset = 0 days
# Lower South Manhattan -> Offset = 5 days

# Hyperparameters for the neural network
batch_size = 60
sequence_length = 6
number_of_features_per_element = 2
learning_rate = 0.01
num_hidden_features = 128
number_epochs = 10

# Scaler used for standardization
complaint_scaler = StandardScaler()
traffic_scaler = StandardScaler()


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


keys = {
    "l=": "sequence_length",
    "b=": "batch_size",
    "l_r=": "learning_rate",
    "e=": "number_epochs",
    "h=": "num_hidden_features",
}
for i in range(1, len(sys.argv)):
    for key in keys:
        if sys.argv[i].find(key) == 0:
            value = sys.argv[i][len(key):]

            print(f"Found input '{keys[key]} = {value}'")
            globals()[keys[key]] = parse_str(value)

if torch.cuda.is_available():
    print(f"GPU training is available")
    device = "cuda:0"
else:
    print(f"GPU training is NOT available")
    device = "cpu"

print("---------------")
print(f"Input parameters:")
print(f"Zone: {zone}")
print(
    f"b = {batch_size}, l = {sequence_length}, l_r = {learning_rate}, |h| = {num_hidden_features}, epochs = {number_epochs}"
)
print("---------------")

subtitle_string = f"$b={batch_size}, e={number_epochs}, l={sequence_length}, |h| ={num_hidden_features}, l_r = {learning_rate}, offset = {traffic_offset} days$"


def load_complaint_data(years_to_load: Union[List[int], int], zone: str) -> DataFrame:
    years_to_check = []

    try:
        # Check to see if year is an iterable
        years_to_check = list(iter(years_to_load))

    except TypeError:
        # Only check one year
        years_to_check = [years_to_load]

    path_to_load_from = f"input/complaints/preprocessed complaints manhattan with zones and zonesets.csv"
    all_complaints: DataFrame = pd.read_csv(
        path_to_load_from,
        sep=";",
        decimal=",",
        usecols=[
            "Created Date",
            "Borough",
            "Latitude",
            "Longitude",
            "zone",
            "zone set",
        ],
    )

    # Convert date column to Pandas DateTime
    all_complaints["Created Date"] = pd.to_datetime(
        all_complaints["Created Date"], format="%d-%m-%Y %H:%M:%S"
    )

    # Keep only complaints of the years we want to keep
    all_complaints = all_complaints[
        all_complaints["Created Date"].dt.year.isin(years_to_check)
    ]

    # Only keep complaints from a particular zone.
    all_complaints = all_complaints[all_complaints["zone set"] == zone]

    # Group the complaints by day (count the number of complaints per day)
    resampled_data = (
        all_complaints.resample("1H", on="Created Date", label="right")
            .count()[["Created Date"]]
            .rename({"Created Date": "Complaints"}, axis=1)
    )

    return resampled_data


def load_traffic_data(years_to_load: Union[List[int], int], zone: str) -> DataFrame:
    years_to_check = []

    try:
        # Check to see if year is an iterable
        years_to_check = list(iter(years_to_load))

    except TypeError:
        # Only check one year
        years_to_check = [years_to_load]

    all_traffic_data: DataFrame = None

    column_names = ["zone", "zone set", "date", "value"]

    for year_to_check in years_to_check:

        temporary_data_storage = []

        path_to_dropoff_data = f"input/taxi data/{year_to_check}/preprocessed/dropoff_per_zone_{year_to_check}_with_zonesets.csv"
        dropoff_data: DataFrame = pd.read_csv(
            path_to_dropoff_data,
            sep=";",
            decimal=",",
        )

        # Only keep data of selected zone
        dropoff_data = dropoff_data[dropoff_data["zone set"] == zone]

        # Convert date column to Pandas DateTime
        dropoff_data["date"] = pd.to_datetime(dropoff_data["date"], format="%d-%m-%Y")

        def add_dataframe_row_to_temp_storage(row):
            zone: str = row[0]
            zone_set: str = row[1]
            date: Timestamp = row[2]

            for hour_in_day in range(1, 25):
                movements_in_hour = row[str(hour_in_day)]
                adjusted_datetime = date + Timedelta(hours=hour_in_day)

                # Store the row in a new data list
                temporary_data_storage.append(
                    [zone, zone_set, adjusted_datetime, movements_in_hour]
                )

        dropoff_data.apply(lambda row: add_dataframe_row_to_temp_storage(row), axis=1)

        path_to_pickup_data = f"input/taxi data/{year_to_check}/preprocessed/pickups_per_zone_{year_to_check}_with_zonesets.csv"
        pickup_data: DataFrame = pd.read_csv(
            path_to_pickup_data,
            sep=";",
            decimal=",",
        )

        # Only keep data of selected zone
        pickup_data = pickup_data[pickup_data["zone set"] == zone]

        # Convert date column to Pandas DateTime
        pickup_data["date"] = pd.to_datetime(pickup_data["date"], format="%d-%m-%Y")

        pickup_data.apply(lambda row: add_dataframe_row_to_temp_storage(row), axis=1)

        # Create pandas dataframe
        traffic_data = pd.DataFrame(temporary_data_storage, columns=column_names)

        # Because and pickup and dropoff events can happen on the same moment, we want to merge events if they occur
        # simultaneously.
        traffic_data_duplicates_merged = (
            traffic_data.groupby(["zone", "zone set", "date"])
                .agg(
                {"zone": "first", "zone set": "first", "date": "first", "value": "sum"}
            )
                .reset_index(drop=True)
        )

        # Sum all data of a zone set per date time
        traffic_data_summed_per_date = (
            traffic_data_duplicates_merged.groupby("date")
                .agg({"zone set": "first", "date": "first", "value": "sum"})
                .reset_index(drop=True)
        )

        # Store the data of this year
        if all_traffic_data is None:
            # If there was no data yet, we initialize the dataframe
            all_traffic_data = traffic_data_summed_per_date
        else:
            # Otherwise, we add to the existing dataframe
            all_traffic_data = pd.concat(
                [all_traffic_data, traffic_data_summed_per_date]
            )

    all_traffic_data = all_traffic_data[["date", "value"]].rename(
        {"value": "Traffic"}, axis=1
    )

    return all_traffic_data


train_complaint_data = load_complaint_data(train_years, zone)
train_traffic_data = load_traffic_data(train_years, zone)

test_complaint_data = load_complaint_data(test_year, zone)
test_traffic_data = load_traffic_data(test_year, zone)


# Define custom dataset so we can easily load our data in the model
# Each item in this dataset is a sequence of hours (the sequence length determines how many hours there are in one sequence)
# The label of an item is the number of complaints in the next hour
class SequenceHourDataset(Dataset):
    def __init__(
            self,
            complaints_dataframe: DataFrame,
            traffic_dataframe: DataFrame,
            number_of_elements_per_sequence: int,
            number_of_features_per_element: int,
            offset_traffic_in_days: int = 0,
    ):
        self.sequence_length: int = number_of_elements_per_sequence

        self.joined_data = (
            complaints_dataframe.reset_index()
                .merge(traffic_dataframe, left_on="Created Date", right_on="date")
                .drop("date", axis=1)
        )

        # Shift the traffic data
        if offset_traffic_in_days != 0:
            self.joined_data["Traffic"] = self.joined_data["Traffic"].shift(
                periods=offset_traffic_in_days * 24, fill_value=0
            )

        self.complaints_tensor: torch.tensor = (
            torch.tensor(
                complaint_scaler.fit_transform(
                    self.joined_data["Complaints"].values.reshape(-1, 1)
                )
            )
                .flatten()
                .float()
                .cuda()
        )
        self.traffic_tensor: torch.tensor = (
            torch.tensor(
                traffic_scaler.fit_transform(
                    self.joined_data["Traffic"].values.reshape(-1, 1)
                )
            )
                .flatten()
                .float()
                .cuda()
        )

    def __len__(self):
        return self.complaints_tensor.shape[0] - 1

    def __getitem__(self, index):
        # If we have a sequence that fits in the dataset (without padding), we just return that sequence
        if index >= self.sequence_length - 1:
            index_start = index - self.sequence_length + 1
            complaint_sequence: torch.tensor = self.complaints_tensor[
                                               index_start: (index + 1)
                                               ]
            traffic_sequence: torch.tensor = self.traffic_tensor[
                                             index_start: (index + 1)
                                             ]
        else:
            # If the sequence does not fit (because it requests data with index < 0),
            # we pad the sequence with the first value
            complaint_padding = self.complaints_tensor[0].repeat(
                self.sequence_length - index - 1
            )
            traffic_padding = self.traffic_tensor[0].repeat(
                self.sequence_length - index - 1
            )

            # Grab sequence that is valid
            complaint_sequence: torch.tensor = self.complaints_tensor[0: (index + 1)]
            traffic_sequence: torch.tensor = self.traffic_tensor[0: (index + 1)]
            # Add padding to the sequence
            complaint_sequence = torch.cat((complaint_padding, complaint_sequence), 0)
            traffic_sequence = torch.cat((traffic_padding, traffic_sequence), 0)

        # Merge complaints and traffic sequence together
        # We obtain a sequence of elements, where each element consists of two elements again
        # (the first is the # of complaints of a day, the second the # of traffic of the same day)
        # Something like this: [[10,100][0,110][1,20]]
        final_sequence: torch.tensor = torch.cat(
            (complaint_sequence.reshape(-1, 1), traffic_sequence.reshape(-1, 1)), dim=1
        )

        # The goal is to predict the number of complaints on the next day.

        # The length of this dataset is one fewer than the number of days in the dataframe (as the last day does not
        # have a next day).
        return final_sequence, self.complaints_tensor[index + 1]


dataset_train = SequenceHourDataset(
    complaints_dataframe=train_complaint_data,
    traffic_dataframe=train_traffic_data,
    number_of_elements_per_sequence=sequence_length,
    number_of_features_per_element=number_of_features_per_element,
    offset_traffic_in_days=traffic_offset,
)

dataset_test = SequenceHourDataset(
    complaints_dataframe=test_complaint_data,
    traffic_dataframe=test_traffic_data,
    number_of_elements_per_sequence=sequence_length,
    number_of_features_per_element=number_of_features_per_element,
    offset_traffic_in_days=traffic_offset,
)

loader_train_set = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
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

    def forward(self, x):
        batch_size = x.shape[0]

        x.to(device)

        output, (hn, cn) = self.lstm(
            x.reshape(batch_size, self.sequence_length, self.num_features_per_element)
        )
        out = self.linear(
            hn[0]
        ).flatten()  # First dim of Hn is num_layers, which is set to 1 above.

        out.to(device)

        return out


# ------------ Define the model -------------

model = SampleLSTM(
    sequence_length=sequence_length,
    num_features_in_hidden_state=num_hidden_features,
    num_features_per_element=number_of_features_per_element,
)
model.to(device)

loss_function = nn.L1Loss()
loss_function.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


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
    # print(f"Train loss: {avg_loss}")

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
    # print(f"Test loss: {avg_loss}")

    return avg_loss


# Store the loss per epoch
train_loss_per_epoch = []
test_loss_per_epoch = []

# Do some training for a few epochs
for epoch_index in tqdm.tqdm(range(number_epochs)):
    train_loss = train_model(
        loader_train_set, model, loss_function, optimizer=optimizer
    )
    test_loss = test_model(loader_test_set, model, loss_function)

    train_loss_per_epoch.append(train_loss)
    test_loss_per_epoch.append(test_loss)

plt.rcParams.update({"text.usetex": True, "font.family": "Helvetica"})
# Make sure all labels fit on the drawing
plt.rcParams["figure.constrained_layout.use"] = True

# Plot loss over time for epoch
plt.plot(train_loss_per_epoch, label="Train loss")
plt.plot(test_loss_per_epoch, label="Test loss")
plt.margins(x=0)

plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.suptitle(f"Loss over time on {train_years} + {test_year}")
plt.title(
    subtitle_string,
    fontsize=10,
)

plt.legend()
plt.show()


# Evaluate
def predict(data_loader, model):
    output = torch.tensor([]).cuda()
    model.eval()
    with torch.no_grad():
        for X, _ in data_loader:
            y_star = model(X)
            output = torch.cat((output, y_star), 0)

    return output


loader_train_evaluation = DataLoader(
    dataset_train, batch_size=batch_size, shuffle=False
)

forecast_column_name = "Model forecast"

train_data_converted_to_forecasted = train_complaint_data.reset_index()

test_data_converted_to_forecasted = test_complaint_data.reset_index()

train_data_converted_to_forecasted[forecast_column_name] = pd.Series(
    data=predict(loader_train_evaluation, model).cpu().numpy(),
)

train_data_converted_to_forecasted.dropna(inplace=True)
# Make sure to transform the standardized output back to non-standardized output so we can compare it
train_data_converted_to_forecasted[
    forecast_column_name
] = complaint_scaler.inverse_transform(
    train_data_converted_to_forecasted[[forecast_column_name]]
)

test_data_converted_to_forecasted[forecast_column_name] = pd.Series(
    data=predict(loader_test_set, model).cpu().numpy(),
)

test_data_converted_to_forecasted.dropna(inplace=True)
# Make sure to transform the standardized output back to non-standardized output so we can compare it
test_data_converted_to_forecasted[
    forecast_column_name
] = complaint_scaler.inverse_transform(
    test_data_converted_to_forecasted[[forecast_column_name]]
)

# Build dataframe that shows the label and the predicted label
dataframe_to_compare = pd.concat(
    (train_data_converted_to_forecasted, test_data_converted_to_forecasted)
)[["Created Date", "Complaints", forecast_column_name]]


# Draw predictions and ground truth to be able to compare them
def plot_prediction_results(
        predictions,
        ground_truth,
        timestamps,
        draw_year_lines=False,
        draw_mean_line=True,
        title=None,
        subtitle=None,
):
    # Plot ground truth values vs. prediction
    plt.plot(timestamps, ground_truth.reset_index(drop=True), label="Ground truth")
    plt.plot(timestamps, predictions.reset_index(drop=True), label="Prediction")

    plt.xticks(rotation=45, ha="right")

    # Put a super title (if applicable)
    if subtitle is not None:
        plt.title(subtitle, fontsize=10)
    else:
        plt.title(
            subtitle_string,
            fontsize=10,
        )

    # Put a subtitle (if applicable)
    if title is not None:
        plt.suptitle(title)
    else:
        plt.suptitle("Predictions vs ground truth")

    plt.ylabel("Value to predict")

    if draw_year_lines:
        # Add vertical lines for each year of the training data
        for train_year in train_years:
            plt.axvline(
                datetime.datetime(train_year, 1, 1),
                color=(random.random(), random.random(), random.random()),
                linestyle="dashed",
                label=f"Start of a {train_year}",
            )

        plt.axvline(
            datetime.datetime(test_year, 1, 1),
            color="r",
            linestyle="dotted",
            label=f"Start of test data",
        )

    mean_prediction_value = Series.mean(predictions)
    stdev_prediction_value = Series.std(
        predictions, ddof=0
    )  # Obtain the population stdev (not sample)
    mean_ground_truth_value = Series.mean(ground_truth)

    if draw_mean_line:
        # Plot a horizontal line of the mean prediction
        plt.axhline(
            y=mean_prediction_value,
            color="r",
            linestyle="dashed",
            label="Mean (predictions)",
        )
        plt.axhline(
            y=mean_ground_truth_value,
            color="g",
            linestyle="dashed",
            label="Mean (ground truth)",
        )

    plt.legend()
    plt.show()

    MSE = sklearn.metrics.mean_squared_error(ground_truth, predictions)
    RMSE = np.sqrt(MSE)
    R2 = sklearn.metrics.r2_score(
        ground_truth, predictions, multioutput="variance_weighted"
    )
    MAE = sklearn.metrics.mean_absolute_error(ground_truth, predictions)
    MEDAE = sklearn.metrics.median_absolute_error(ground_truth, predictions)
    print(f"Average forecast: {mean_prediction_value}")
    print(f"Population stdev: {stdev_prediction_value}")
    print(f"RMSE of model: {RMSE}")
    print(f"R^2 of model: {R2}")
    print(f"MAE: {MAE}")
    print(f"MEDAE: {MEDAE}")

    sns.kdeplot(ground_truth, shade=True, label="Ground truth", clip=(0, 10))
    sns.kdeplot(predictions, shade=True, label="Predictions", clip=(0, 10))
    plt.suptitle(f"Enriched - Distribution of predictions vs ground truth - {zone}")
    plt.title(
        subtitle_string,
        fontsize=10,
    )
    plt.legend()
    plt.show()

    return RMSE, MAE, MEDAE


# EVALUATION

print(f"Train dataset ({train_years}) has {len(dataset_train)} items")
print(f"Test dataset ({test_year}) has {len(dataset_test)} items")

print(f"-----------")
print(f"Evaluation for test data")

filtered_predictions = dataframe_to_compare[
    dataframe_to_compare["Created Date"].dt.year.isin([test_year])
].copy()

predicted_values_filtered = filtered_predictions[forecast_column_name]
ground_truth_filtered = filtered_predictions["Complaints"]
timestamps_filtered = filtered_predictions["Created Date"]

results = plot_prediction_results(
    predicted_values_filtered,
    ground_truth_filtered,
    timestamps_filtered,
    title=f"Predictions of test year {test_year}",
)
print(f"-----------")

# # Write output of predictions to file
# output_dataframe = test_data_converted_to_forecasted.copy()[
#     ["Created Date", forecast_column_name]
# ]
#
# # Round predictions to whole number
# output_dataframe[forecast_column_name] = (
#     output_dataframe[forecast_column_name].astype("float").round(0).astype(int)
# )
#
# path_to_output_to = f"output/prediction/non-baseline/complaints/{test_year}/{zone}/complaints_batch{batch_size}_sequence{sequence_length}.csv"
#
# # Create directories if they do not exist yet
# os.makedirs(os.path.dirname(path_to_output_to), exist_ok=True)
#
# output_dataframe.to_csv(
#     path_to_output_to,
#     sep=";",
#     index=False,
#     decimal=",",
# )
#
# ---------
# We also want to determine the quality of the predictions at the position of the outliers
# So, what is the difference in prediction vs ground truth only at the dates that we predict are outliers

outlier_file = f"output/outliers/complaints/{test_year}/{zone}/outliers_lof{compare_to_outliers_lof}.csv"

# Load outliers as determined by the LOF algorithm
determined_outliers_dataframe = pd.read_csv(
    outlier_file,
    sep=";",
    decimal=",",
)

# Only keep the records that are deemed to be an outlier
determined_outliers_dataframe = determined_outliers_dataframe[
    determined_outliers_dataframe["Value"] == 1
    ]

# Read the dates of the outliers
determined_outliers_dataframe["Date"] = pd.to_datetime(
    determined_outliers_dataframe["Date"]
)

# Grab the predictions that are made at the outlier dates
data_at_outliers = filtered_predictions[
    filtered_predictions["Created Date"].isin(
        determined_outliers_dataframe["Date"].dt.date
    )
]

print(f"-----------")
print(f"Evaluation for test data at outlier points")

predicted_values_at_outliers = data_at_outliers[forecast_column_name]
ground_truth_at_outliers = data_at_outliers["Complaints"]
timestamps_at_outliers = data_at_outliers["Created Date"]

plot_prediction_results(
    predicted_values_at_outliers,
    ground_truth_at_outliers,
    timestamps_at_outliers,
    draw_year_lines=False,
    draw_mean_line=False,
    title=f"Predictions at outlier dates",
)
print(f"-----------")
