import subprocess

from matplotlib import pyplot as plt

"""
This file is used for running a hyperparameter search. It runs each model with a different hyperparameter set. It feeds
the hyperparameters via a command line argument. 
"""

# Hyperparameter to test
parameter_name = "h"

# Title and subtitle of the figure
title = f"Effect of ${parameter_name}$ on evaluation metrics"
subtitle = "$l = 6, b = 30, l_r = 0.001, e = 10$"

# Parameter values
parameter_values = [
    20,
    30,
    40,
    50,
    60,
    70,
    80,
    90,
    100,
    110,
    120,
    150,
    175,
    200,
    225,
    250,
    275,
]

# Whether we are using the enriched or the baseline model
use_enriched_model = True

# Determine the script to run
if use_enriched_model:
    script_to_run = "prediction/predict_complaints_by_hour_enriched.py"
else:
    script_to_run = "prediction/predict_complaints_by_hour_baseline.py"

baseline_model_rmse = []
baseline_model_mae = []
baseline_model_medae = []

RUN_BASELINE_MODELS = True

if RUN_BASELINE_MODELS:
    for parameter_value in parameter_values:
        print(f"Testing model with {subtitle} and {parameter_name}={parameter_value}")
        output = subprocess.check_output(
            f"python {script_to_run} {parameter_name}={parameter_value}",
            shell=True,
        )

        string_output = output.decode("utf-8")
        output_lines = string_output.splitlines()

        try:
            index_of_start_test_data_results = output_lines.index(
                "Evaluation for test data"
            )
        except ValueError:
            print(
                f"Could not determine performance of model with {parameter_name}={parameter_value}!"
            )
            continue

        performance_on_test_data = output_lines[
                                   index_of_start_test_data_results + 1: index_of_start_test_data_results + 10
                                   ]

        for performance_line in performance_on_test_data:
            if "RMSE" in performance_line:
                rmse = float(performance_line.split(":")[1])
                baseline_model_rmse.append(rmse)
            elif "MAE" in performance_line:
                mae = float(performance_line.split(":")[1])
                baseline_model_mae.append(mae)
            elif "MEDAE" in performance_line:
                medae = float(performance_line.split(":")[1])
                baseline_model_medae.append(medae)

        print(f"Performance of {parameter_name}={parameter_value}:")
        print(f"RMSE: {baseline_model_rmse[-1]}")
        print(f"MAE: {baseline_model_mae[-1]}")
        print(f"MEDAE: {baseline_model_medae[-1]}")

plt.rcParams.update({"text.usetex": True, "font.family": "Helvetica"})

plt.plot(parameter_values, baseline_model_rmse, label="RMSE")
plt.plot(parameter_values, baseline_model_mae, label="MAE")
plt.plot(parameter_values, baseline_model_medae, label="MEDAE")
plt.gca().set_ylim(bottom=0)

plt.suptitle(title)
plt.title(subtitle)

plt.ylabel("Evaluation metric score")
plt.xlabel(f"${parameter_name}$")

plt.legend()
plt.show()
