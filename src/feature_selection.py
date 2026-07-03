import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_regression
from collections import defaultdict

def calculate_mi_scores(df: pd.DataFrame, target_column: str) -> pd.DataFrame:
    """
    Calculate Mutual Information scores between lag features
    and the target variable.
    """

    lag_features = [col for col in df.columns if "_lag_" in col]
    X = df[lag_features]
    y = df[target_column]

    mi_scores = mutual_info_regression(X, y, random_state=42)
    mi_df = pd.DataFrame({"feature": X.columns, "MI Score": mi_scores})

    return (mi_df.sort_values("MI Score", ascending=False).reset_index(drop=True))

def select_features_by_median(mi_df: pd.DataFrame):
    """
    Select features whose MI score is greater than or equal
    to the median MI score.
    """
    median_score = (mi_df["MI Score"].median())
    selected_df = (mi_df[mi_df["MI Score"] >= median_score].reset_index(drop=True))
    return selected_df, median_score

def group_selected_lags(selected_features: list) -> dict:
    """
    Group selected lag features by their base feature.
    """
    grouped = defaultdict(list)
    for feature in selected_features:
        feature_clean = (feature.replace("_lag", ""))
        parts = (feature_clean.rsplit("_", 1))

        if len(parts) == 2:
            feature_name = parts[0]
            lag = int(parts[1])
            grouped[feature_name].append(lag)
    return dict(grouped)


def calculate_cumulative_contribution(ensemble_df: pd.DataFrame, score_column: str = "Ensemble_Score") -> pd.DataFrame:
    """
    Calculate cumulative contribution based on feature importance scores.

    Parameters
    ----------
    ensemble_df : pd.DataFrame
        DataFrame containing feature importance scores.
    score_column : str, default="Ensemble_Score"
        Column used to calculate cumulative contribution.

    Returns
    -------
    pd.DataFrame
        DataFrame with an additional
        'Cumulative_Contribution' column.
    """

    df = ensemble_df.copy()
    if score_column not in df.columns:
        raise ValueError(
            f"Column '{score_column}' not found in DataFrame."
        )
    total_score = df[score_column].sum()
    if total_score == 0:
        raise ValueError("Total feature importance score is zero.")
    df["Cumulative_Contribution"] = (df[score_column].cumsum() / total_score)
    return df


def select_features_by_contribution(ensemble_df, threshold):
    idx_cutoff = (
        ensemble_df["Cumulative_Contribution"]
        >= threshold
    ).idxmax()

    selected_df = ensemble_df.iloc[:idx_cutoff + 1]
    selected_features = selected_df["Feature"].tolist()
    return selected_df, selected_features