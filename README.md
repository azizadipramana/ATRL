
# Adaptive Temporal Region Learning (ATRL)

### Submitted to The Indonesia Journal of Science and Technology (Under Editor Review)

## Requirements

Python >= 3.10

Install dependencies:

```bash
pip install -r requirements.txt
````

Main libraries:

* numpy
* pandas
* scikit-learn
* matplotlib
* xgboost
* lightgbm
* openpyxl

## Run Experiment

Run notebooks sequentially:

```
01_data_eda.ipynb
02_data_preprocessing.ipynb
03_data_description.ipynb
04_data_splitting.ipynb
05_data_benchmark.ipynb
06_temporal_anchor_discovery.ipynb
07_candidate_lag_pruning.ipynb
08_ensemble_lag_scoring.ipynb
09_contribution_temporal_compression.ipynb
10_evaluation_determine_optimal_lag.ipynb
11_hyperparameter_tuning.ipynb
12_k-fold.ipynb
13_test.ipynb
14_benchmarking.ipynb
```

## Proposed Method

* **Temporal Anchor Discovery**: Finds important historical time points.
* **Candidate Lag Pruning**: Removes irrelevant lag features.
* **Ensemble Lag Scoring**: Ranks lag importance using multiple scoring methods.
* **Contribution Temporal Compression**: Selects compact lag features based on contribution.

## Project Structure

```
datasets/
├── raw/          # Original PV and wind datasets
└── processed/    # Preprocessed datasets

notebooks/
├── 01-05         # Data preparation and benchmark
├── 06-09         # Proposed temporal compression method
└── 10-14         # Evaluation and benchmarking

src/
├── data_preprocessing.py   # Data preprocessing functions
├── data_splitting.py       # Dataset splitting functions
├── feature_engineering.py  # Feature generation
├── feature_ranking.py      # Feature ranking methods
└── feature_selection.py    # Feature selection methods

result/
├── csv files      # Feature selection results
├── figure/        # Experimental figures
└── images_paper/  # Paper figures
```
