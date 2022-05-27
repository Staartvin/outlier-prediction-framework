This is the directory of the second use case. There is a directory hierarchy that can be explained as follows:
- Input -> All raw and processed data of The Hague traffic is stored here. Since the data used in the thesis is confidential, I cannot share it with you.
- Output -> Most scripts perform computations and store the results in the output directory. Think of figures, CSVs with detected outliers or predictions of unseen data.
- Scripts -> Contains a list of scripts that you can run. At the top of each script, the goal of the script is explained. Some scripts create the required directories themselves,
while other scripts do not. Be on the lookout for errors!

All of the scripts are useful, but generally these stand out:
- create_preprocessed_traffic_files -> parses the raw The Hague dataset and stores aggregated files in the input folder.
- predict_traffic_baseline -> Predict traffic of an intersection using the baseline model.
- predict_traffic_enriched -> Predict traffic of an intersection using source intersection with the enriched model.
- run_model_several_times -> Runs either the baseline model or enriched model multiple times (for averaged results).
- run_hyperparameter_search -> Performs a hyperparameter search using either the baseline or enriched model.