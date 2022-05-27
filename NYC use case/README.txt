This is the directory of the first use case. There is a directory hierarchy that can be explained as follows:
- Input -> All raw and processed data of NYC is stored here. The complaint data is about 1Gb large so too large to share conveniently with git. 
            The taxi data is over 400Gb, so trivially not stored in this repo. Look up the data on the NYC opendata platform.
- Output -> Most scripts perform computations and store the results in the output directory. Think of figures, CSVs with detected outliers or predictions of unseen data.
- Scripts -> Contains a list of scripts that you can run. At the top of each script, the goal of the script is explained. Some scripts create the required directories themselves,
while other scripts do not. Be on the lookout for errors!

Note that preprocessing was done in Tableau as the datasets are too large to be processed easily using Python. Unfortunately, that means they are not easily shared in this repo.

All of the scripts are useful, but generally these stand out:
- predict_complaints_by_hour_baseline -> Train a neural network to predict complaints based on historical complaints of a particular zone
- predict_complaints_by_hour_enriched -> Train a neural network to predict complaints based on historical complaints AND traffic data of a particular zone
- run_hyperparameter_search -> Run a hyperparameter search on either the baseline or enriched model to inspect the effect of hyperparameter on the model.
- plot_outliers_from_ground_truth -> Opens an interactive visualization with which you can inspect the outlier of the ground truth complaints
- plot_outliers_from_predictions -> Opens an interactive visualization with which you can inspect the outlier of the predicted complaints