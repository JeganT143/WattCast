import pandas as pd
from config.paths import RAW_DATA_PATH


def load_raw_data(path=RAW_DATA_PATH):
    df = pd.read_csv(path)

    df["date"] = pd.to_datetime(df["date"])

    df = df.sort_values("date").set_index("date")

    return df
