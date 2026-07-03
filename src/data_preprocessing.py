from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent

def load_pv_dataset():
    files = [
        BASE_DIR / "datasets/raw/pv_musim_dingin.xlsx",
        BASE_DIR / "datasets/raw/pv_musim_semi.xlsx",
        BASE_DIR / "datasets/raw/pv_musim_panas.xlsx",
        BASE_DIR / "datasets/raw/pv_musim_gugur.xlsx"
    ]
    dfs = [pd.read_excel(file) for file in files]
    return pd.concat(dfs, axis=0)


def preprocess_pv_dataset(df):
    columns = [
        'timestamp',
        'time',
        'temperature',
        'humidity',
        'surface_radiation',
        'upper_atmospheric_radiation',
        'output_pv'
    ]
    df = df.copy()
    df.columns = columns
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour'] = df['timestamp'].dt.hour
    df['month'] = df['timestamp'].dt.month
    df = df.sort_values(by='timestamp').reset_index(drop=True)
    return df

def load_wind_dataset():
    files = [
        BASE_DIR / "datasets/raw/wind_musim_dingin.xlsx",
        BASE_DIR / "datasets/raw/wind_musim_semi.xlsx",
        BASE_DIR / "datasets/raw/wind_musim_panas.xlsx",
        BASE_DIR / "datasets/raw/wind_musim_gugur.xlsx"
    ]
    dfs = [pd.read_excel(file) for file in files]
    return pd.concat(dfs, axis=0)

def preprocess_wind_dataset(df):
    columns = [
        'timestamp',
        'time',
        'air_density',
        'wind_velocity',
        'output_wind', 
    ]

    df = df.copy()
    df.columns = columns
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour'] = df['timestamp'].dt.hour
    df['month'] = df['timestamp'].dt.month
    df = df.sort_values(by='timestamp').reset_index(drop=True)
    return df


def merge_hybrid_dataset(df_pv,df_wind):
    df_wind = df_wind.drop(columns=['hour','month','time'])
    df = pd.merge(df_pv,df_wind,on='timestamp',how='inner')
    df['output_hybrid'] = (df['output_pv'] + df['output_wind'])
    df.drop(columns=['time'], inplace=True, errors='ignore')
    return df

def save_dataset(df, path):
    df.to_csv(path, index=False)


def load_dataset(file_path):
    df = pd.read_csv(file_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return (
        df
        .sort_values("timestamp")
        .reset_index(drop=True)
    )