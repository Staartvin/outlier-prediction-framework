import subprocess

import pandas as pd
import seaborn as sns

from matplotlib import pyplot as plt

"""
This file is used for running a model several times.
"""

# Whether we are using the enriched or the baseline model
use_enriched_model = True

# Determine the script to run
if use_enriched_model:
    script_to_run = "scripts/predict_traffic_enriched.py"
else:
    script_to_run = "scripts/predict_traffic_baseline.py"

# Store the evaluation metrics for each run
model_rmse = []
model_mae = []
model_medae = []

# How many iterations are we going to run?
iterations_to_repeat = 10

# Run each iteration of the model and store the results
for iteration in range(0, iterations_to_repeat):
    # Run the script
    output = subprocess.check_output(
        f"python {script_to_run}",
        shell=True,
    )

    # Grab the output of the model (it will print strings)
    string_output = output.decode("utf-8")
    output_lines = string_output.splitlines()

    # Try to parse the output in a meaningful way
    try:
        index_of_start_test_data_results = output_lines.index(
            f"{'Enriched' if use_enriched_model else 'Baseline'} - evaluation for test data"
        )
    except ValueError:
        print(f"Could not determine performance of model!")
        break

    # Read the lines that have performance information in them
    performance_on_test_data = output_lines[
        index_of_start_test_data_results + 1 : index_of_start_test_data_results + 10
    ]

    # Go over the performance lines and read the RMSE, MAE and MEDAE
    # Store these values in the corresponding lists
    for performance_line in performance_on_test_data:
        if "RMSE" in performance_line:
            rmse = float(performance_line.split(":")[1])
            model_rmse.append(rmse)
        elif "MAE" in performance_line:
            mae = float(performance_line.split(":")[1])
            model_mae.append(mae)
        elif "MEDAE" in performance_line:
            medae = float(performance_line.split(":")[1])
            model_medae.append(medae)

    # Output the performance of this iteration to the user
    print(f"Performance @ iteration {iteration}:")
    print(f"RMSE: {model_rmse[-1]}")
    print(f"MAE: {model_mae[-1]}")
    print(f"MEDAE: {model_medae[-1]}")


# Let matplotlib know we want to use latex for formatting math in figures
plt.rcParams.update({"text.usetex": True, "font.family": "Helvetica"})

print(f"RMSE scores: {model_rmse}")
print(f"MAE scores: {model_mae}")
print(f"MEDAE scores: {model_medae}")

# Add three lists as columns into a dataframe
performance_data = pd.DataFrame(
    {"RMSE": model_rmse, "MAE": model_mae, "MEDAE": model_medae}
)

# Generate barplot based on the data
ax = sns.barplot(
    data=pd.melt(performance_data), x="variable", y="value", capsize=0.2, errwidth=1.0
)

# Add values of each bar chart to the figure
ax.bar_label(ax.containers[-1], fmt="\n%.2f", label_type="center")

plt.ylabel("Evaluation metric score")
plt.xlabel(f"Evaluation metric")
plt.title(
    f"Average scores of {'enriched' if use_enriched_model else 'baseline'} model for {len(model_rmse)} iterations"
)

plt.show()
