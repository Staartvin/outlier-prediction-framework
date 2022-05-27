import datetime
import os
import random
import re
import sys
from typing import Union, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.metrics
import torch
import tqdm
from pandas import DataFrame, Series
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import Dataset, DataLoader

import seaborn as sns

"""
This file is used to generate predictions for complaints based on historical data of a certain zone. It uses PyTorch to generate a model and train it.
"""

# Zone to train for
zone = "Lower South Manhattan"

# Data to use as train years
train_years = [2015, 2016, 2017]

# Data to use as test year
test_year = 2018

# Neural network hyperparameters
batch_size = 60
sequence_length = 6
number_of_features_per_element = 1
learning_rate = 0.001
num_hidden_features = 48
number_epochs = 20

compare_to_outliers_lof = 40

# Scaler used for standardization
scaler = StandardScaler()


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

            value = sys.argv[i][len(key) :]

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

subtitle_string = f"$b={batch_size}, e={number_epochs}, l={sequence_length}, |h| ={num_hidden_features}, l_r = {learning_rate}$"


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


# Define custom dataset so we can easily load our data in the model
# Each item in this dataset is a sequence of hours (the sequence length determines how many hours there are in one sequence)
# The label of an item is the number of complaints in the next hour
class SequenceHourDataset(Dataset):
    def __init__(
        self,
        dataframe: DataFrame,
        number_of_elements_per_sequence: int,
        number_of_features_per_element: int,
    ):
        self.sequence_length: int = number_of_elements_per_sequence

        # Standardize data to have a mean of 0 and a variance of 1
        self.data = scaler.fit_transform(dataframe.values.reshape(-1, 1))

        self.x: torch.tensor = torch.tensor(self.data.reshape(-1)).float().cuda()

    def __len__(self):
        return self.x.shape[0] - 1

    def __getitem__(self, index):
        # If we have a sequence that fits in the dataset (without padding), we just return that sequence
        if index >= self.sequence_length - 1:
            index_start = index - self.sequence_length + 1
            sequence: torch.tensor = self.x[index_start : (index + 1)]
        else:
            # If the sequence does not fit (because it requests data with index < 0),
            # we pad the sequence with the first value
            padding = self.x[0].repeat(self.sequence_length - index - 1)
            # Grab sequence that is valid
            sequence: torch.tensor = self.x[0 : (index + 1)]
            # Add padding to the sequence
            sequence = torch.cat((padding, sequence), 0)

        # The goal is to predict the number of complaints on the next day.

        # The length of this dataset is one fewer than the number of days in the dataframe (as the last day does not
        # have a next day).
        return sequence, self.x[index + 1]


train_complaints = load_complaint_data(train_years, zone)
test_complaints = load_complaint_data(test_year, zone)

dataset_train = SequenceHourDataset(
    dataframe=train_complaints["Complaints"],
    number_of_elements_per_sequence=sequence_length,
    number_of_features_per_element=number_of_features_per_element,
)

dataset_test = SequenceHourDataset(
    dataframe=test_complaints["Complaints"],
    number_of_elements_per_sequence=sequence_length,
    number_of_features_per_element=number_of_features_per_element,
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

loss_function = nn.MSELoss()
loss_function.to(device)

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
plt.plot(
    train_loss_per_epoch,
    label="Train loss",
)
plt.plot(
    test_loss_per_epoch,
    label="Test loss",
)
plt.margins(x=0)
# plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
plt.xticks(rotation=45, ha="right")
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

train_data_converted_to_forecasted = train_complaints.reset_index()


test_data_converted_to_forecasted = test_complaints.reset_index()

train_data_converted_to_forecasted[forecast_column_name] = pd.Series(
    data=predict(loader_train_evaluation, model).cpu().numpy(),
    index=list(
        range(
            number_of_features_per_element - 1,
            len(train_complaints) - 1,
            number_of_features_per_element,
        )
    ),
)

train_data_converted_to_forecasted.dropna(inplace=True)
# Make sure to transform the standardized output back to non-standardized output so we can compare it
train_data_converted_to_forecasted[forecast_column_name] = scaler.inverse_transform(
    train_data_converted_to_forecasted[[forecast_column_name]]
)

test_data_converted_to_forecasted[forecast_column_name] = pd.Series(
    data=predict(loader_test_set, model).cpu().numpy(),
    index=list(
        range(
            number_of_features_per_element - 1,
            len(test_complaints) - 1,
            number_of_features_per_element,
        )
    ),
)

test_data_converted_to_forecasted.dropna(inplace=True)
# Make sure to transform the standardized output back to non-standardized output so we can compare it
test_data_converted_to_forecasted[forecast_column_name] = scaler.inverse_transform(
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
    plt.suptitle(f"Baseline - Distribution of predictions vs ground truth - {zone}")
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

# ---------
# Write output of predictions to file
output_dataframe = test_data_converted_to_forecasted.copy()[
    ["Created Date", forecast_column_name]
]

# Round predictions to whole number
output_dataframe[forecast_column_name] = (
    output_dataframe[forecast_column_name].astype("float").round(0).astype(int)
)

path_to_output_to = f"output/prediction/baseline/complaints/{test_year}/{zone}/complaints_batch{batch_size}_sequence{sequence_length}.csv"

# Create directories if they do not exist yet
os.makedirs(os.path.dirname(path_to_output_to), exist_ok=True)

output_dataframe.to_csv(
    path_to_output_to,
    sep=";",
    index=False,
    decimal=",",
)

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
