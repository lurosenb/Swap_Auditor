# Swap Auditor
NELS empirical data analysis fairness.

## Notebook
Run through vscode or jupyter to walk through some examples of Stability measures on the original dataset using 

## Running Metrics and Generating Plots
Below are instructions to replicate the plots we generated. In order to generate our plots, we leveraged the infrastructure built by Friedler et. al. (https://github.com/algofairness/fairness-comparison) for their paper "A comparative study of fairness-enhancing interventions in machine learning." We added our stability metric to their existing metrics, and modified their framework to run with the NELS:88 education dataset. We are thankful that they shared their code, which allowed us to perform many metric comparisons with relatively low effort.

### Run Algorithms and Track Metrics
From the top directory, run python fairness/runner.py.
Results from the run will appear in: /fairness/data/results/results/

### Correlation Plot
To recreate correlation, cd into analysis/correlation-vis. Make sure the proper data files are included in the directory. Then, run: python combine_results_files.py education_Race_original.csv education_Race-Sex_original.csv education_Sex_original.csv education_cor.csv 

### Other plots
Move the the data files from /fairness/data/results/results/ to /fairness/
Then run: python fairness/runner_analysis.py