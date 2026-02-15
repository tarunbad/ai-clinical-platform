import pandas as pd
from sklearn.model_selection import train_test_split

DROP_COLS = [
    "encounter_id", "patient_nbr"  # IDs leak & add noise
]

NA_TOKENS = {"?": pd.NA, "Unknown/Invalid": pd.NA}

def load_raw(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # standardize missing tokens
    df = df.replace(NA_TOKENS)
    return df

def make_target(df: pd.DataFrame) -> pd.DataFrame:
    # readmitted: '<30' positive, else negative
    df = df.copy()
    df["target_readmit_30"] = (df["readmitted"] == "<30").astype(int)
    return df

def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in DROP_COLS:
        if c in df.columns:
            df = df.drop(columns=c)
    # Drop rows with missing label
    df = df.dropna(subset=["readmitted"])
    return df

def split(df: pd.DataFrame, test_size=0.2, val_size=0.1, seed=42):
    # First split off test
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=seed, stratify=df["target_readmit_30"]
    )
    # Then split train into train/val
    val_ratio = val_size / (1 - test_size)
    train_df, val_df = train_test_split(
        train_df, test_size=val_ratio, random_state=seed, stratify=train_df["target_readmit_30"]
    )
    return train_df, val_df, test_df
