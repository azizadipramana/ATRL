import pandas as pd    

def generate_selected_lags(df: pd.DataFrame, lag_mapping: dict) -> pd.DataFrame:
    df = df.copy()
    lag_data = {}

    for feature, lags in lag_mapping.items():
        if feature not in df.columns:
            raise ValueError(
                f"Feature '{feature}' tidak ditemukan dalam dataset."
            )
        for lag in lags:
            lag_data[f"{feature}_lag_{lag}"] = df[feature].shift(lag)
    
    lag_df = pd.DataFrame(lag_data, index=df.index)
    return pd.concat([df, lag_df], axis=1)

def finalize_lag_dataset(df: pd.DataFrame, drop_columns=None) -> pd.DataFrame:
    df = df.copy()
    if drop_columns:
        df = df.drop(columns=drop_columns, errors="ignore")
    return (df.dropna().reset_index(drop=True))

def create_lag_dataset(df: pd.DataFrame, lag_mapping: dict, drop_columns: list = None) -> pd.DataFrame:
    df = df.copy()
    if drop_columns:
        df = df.drop(columns=drop_columns, errors="ignore")    
    lag_df = generate_selected_lags(df=df, lag_mapping=lag_mapping)
    final_df = (lag_df.dropna().reset_index(drop=True))
    return final_df

def feature_summary(df, dataset_name="Dataset"):
    lag_features = [col for col in df.columns if "_lag_" in col]
    base_features = [col for col in df.columns if "_lag_" not in col]

    print(f"\n=== {dataset_name.upper()} ===")
    print(f"Rows                : {df.shape[0]}")
    print(f"Total Features      : {df.shape[1]}")
    print(f"Original Features   : {len(base_features)}")
    print(f"Lag Features        : {len(lag_features)}")


def generate_all_lags(df: pd.DataFrame, max_lag: int, lag_features: list = None) -> pd.DataFrame:
    df = df.copy()
    lag_data = {}
    
    features_to_process = lag_features if lag_features is not None else df.columns

    for feature in features_to_process:
        if feature in df.columns: 
            for lag in range(1, max_lag + 1):
                lag_data[f"{feature}_lag_{lag}"] = df[feature].shift(lag)

    lag_df = pd.DataFrame(lag_data, index=df.index)
    return pd.concat([df, lag_df], axis=1)

def create_full_lag_dataset(df: pd.DataFrame, max_lag: int, lag_features: list = None, drop_columns: list = None) -> pd.DataFrame:
    lag_df = generate_all_lags(df=df, max_lag=max_lag, lag_features=lag_features)
    lag_df = finalize_lag_dataset(df=lag_df, drop_columns=drop_columns)

    return lag_df