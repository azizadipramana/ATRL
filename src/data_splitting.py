import pandas as pd 

def load_dataset(path):
    df = pd.read_csv(path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return (df.sort_values('timestamp').reset_index(drop=True))

def split_dataset(df, train_ratio=0.70, eval_ratio=0.15):
    n = len(df)
    train_end = int(n * train_ratio)
    eval_end = int(n * (train_ratio + eval_ratio))

    train = (df.iloc[:train_end].copy())
    eval = (df.iloc[train_end:eval_end].copy())
    test = (df.iloc[eval_end:].copy())
    return train, eval, test

def print_split_summary(train, eval, test):
    total = (len(train) + len(eval) + len(test))
    print(f"Total: {total}")
    print(f"Train: {len(train)} ({len(train)/total:.2%})")
    print(f"eValidation: {len(eval)} ({len(eval)/total:.2%})")
    print(f"Test: {len(test)} ({len(test)/total:.2%})")

def save_split_dataset(train,eval,test,dir_path):
    train.to_csv(f"{dir_path}/train.csv", index=False)
    eval.to_csv(f"{dir_path}/eval.csv", index=False)
    test.to_csv(f"{dir_path}/test.csv",index=False)
