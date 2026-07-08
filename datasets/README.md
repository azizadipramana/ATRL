# Dataset
This directory contains the datasets used for the research experiments.

## Directory Structure

```text
.
│── README.md
│
├── benchmark/
│   ├── data_train_benchmark_lag.csv
│   └── data_test_benchmark_lag.csv
│
├── processed/
│   ├── dataset_hybrid.csv
│   ├── dataset_pv.csv
│   ├── dataset_wind.csv
│   ├── train.csv
│   ├── train_lag.csv
│   ├── test.csv
│   ├── test_lag.csv
│   ├── train_combine.csv
│   ├── train_combine_lag.csv
│   ├── eval.csv
│   └── eval_lag.csv
│
└── raw/
    ├── pv_musim_dingin.xlsx
    ├── pv_musim_gugur.xlsx
    ├── pv_musim_panas.xlsx
    ├── pv_musim_semi.xlsx
    ├── wind_musim_dingin.xlsx
    ├── wind_musim_gugur.xlsx
    ├── wind_musim_panas.xlsx
    └── wind_musim_semi.xlsx
```

## Folder Description

* **raw/** – Original seasonal PV and wind datasets.
* **processed/** – Cleaned and processed datasets used for model training, validation, and testing.
* **benchmark/** – Benchmark datasets with lag features for performance comparison.

## Data Source

The original dataset is available from the Mendeley Data repository:

https://data.mendeley.com/datasets/gxc6j5btrx/1

Please download the original dataset from the Mendeley Data repository and place the files in the raw/ directory before running the preprocessing scripts.
