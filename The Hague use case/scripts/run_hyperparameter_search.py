import subprocess

from matplotlib import pyplot as plt

"""
This file is used for running a hyperparameter search. It runs each model with a different hyperparameter set. It feeds
the hyperparameters via a command line argument. 
"""

# Hyperparameter to test
parameter_name = "l_r"

# Title and subtitle of the figure
title = f"Effect of ${parameter_name}$ on evaluation metrics"
subtitle = "$l = 6, h = 48, b = 100, e = 5$"

# Parameter values
parameter_values = [0.1, 0.01, 0.001, 0.0001]

# Whether we are using the enriched or the baseline model
use_enriched_model = True

# Determine the script to run
if use_enriched_model:
    script_to_run = "scripts/predict_traffic_enriched.py"
else:
    script_to_run = "scripts/predict_traffic_baseline.py"

baseline_model_rmse = []
baseline_model_mae = []
baseline_model_medae = []

RUN_BASELINE_MODELS = True

# You may wish to scale the x-axis using a logarithm
USE_LOG_SCALE_FOR_XAXIS = True

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
                f"{'Enriched' if use_enriched_model else 'Baseline'} - evaluation for test data"
            )
        except ValueError:
            print(
                f"Could not determine performance of model with {parameter_name}={parameter_value}!"
            )
            continue

        performance_on_test_data = output_lines[
            index_of_start_test_data_results + 1 : index_of_start_test_data_results + 10
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

if USE_LOG_SCALE_FOR_XAXIS:
    plt.xscale("log")

plt.suptitle(title)
plt.title(subtitle)

plt.ylabel("Evaluation metric score")
plt.xlabel(f"${parameter_name}$")

plt.legend()
plt.show()
