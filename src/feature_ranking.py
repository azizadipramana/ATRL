import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from xgboost import XGBRegressor

def calculate_rf_importance(X, y, n_estimators=200, random_state=42):
    """
    Train a Random Forest model and calculate feature importance scores.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target variable.

    Returns
    -------
    tuple[pd.DataFrame, RandomForestRegressor]
        Feature importance dataframe and trained model.
    """
    
    rf_model = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1
    )
    rf_model.fit(X, y)

    rf_dict = {feature: score for feature, score in zip(X.columns, rf_model.feature_importances_)}

    rf_scores_df = pd.DataFrame({'Feature': list(rf_dict.keys()), 'RF_Importance': list(rf_dict.values())})
    rf_scores_df = rf_scores_df.sort_values(by='RF_Importance', ascending=False).reset_index(drop=True)

    return rf_scores_df, rf_model

def calculate_xgb_importance(X, y, n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42):
    xgb_model = XGBRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1
    )
    xgb_model.fit(X, y)

    xgb_dict = {feature: score for feature, score in zip(X.columns, xgb_model.feature_importances_)}

    xgb_scores_df = pd.DataFrame({'Feature': list(xgb_dict.keys()), 'XGB_Importance': list(xgb_dict.values())})
    xgb_scores_df = xgb_scores_df.sort_values(by='XGB_Importance', ascending=False).reset_index(drop=True)

    return xgb_scores_df, xgb_model

def calculate_permutation_importance(model, X, y, n_repeats=10, random_state=42):
    perm = permutation_importance(model, X, y, n_repeats=n_repeats, random_state=random_state)

    perm_dict = {feature: score for feature, score in zip(X.columns, perm.importances_mean)}
    perm_scores_df = pd.DataFrame({'Feature': list(perm_dict.keys()), 'Permutation_Importance': list(perm_dict.values())})
    perm_scores_df = perm_scores_df.sort_values(by='Permutation_Importance', ascending=False).reset_index(drop=True)

    return perm_scores_df

def min_max_normalize(series):
    denominator = (series.max() - series.min())
    if denominator == 0:
        return pd.Series(0, index=series.index)
    return (series - series.min()) / denominator

def calculate_ensemble_score(rf_df: pd.DataFrame, xgb_df: pd.DataFrame, perm_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate ensemble feature importance by averaging
    normalized RF, XGB, and Permutation importance scores.
    """

    ensemble_df = (rf_df.merge(xgb_df, on="Feature").merge(perm_df, on="Feature"))

    ensemble_df["RF_Norm"] = min_max_normalize(ensemble_df["RF_Importance"])
    ensemble_df["XGB_Norm"] = min_max_normalize(ensemble_df["XGB_Importance"])
    ensemble_df["PERM_Norm"] = min_max_normalize(ensemble_df["Permutation_Importance"])

    ensemble_df["Ensemble_Score"] = (ensemble_df["RF_Norm"] + ensemble_df["XGB_Norm"] + ensemble_df["PERM_Norm"]) / 3

    return (ensemble_df.sort_values(by="Ensemble_Score", ascending=False).reset_index(drop=True))