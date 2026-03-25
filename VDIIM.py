import pandas as pd
import numpy as np

def load_and_process_data():
    df = pd.read_csv("Nifty50dataset.csv")

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.set_index("timestamp", inplace=True)
    df.sort_index(inplace=True)

    df = df[["open", "high", "low", "close"]]
    df = df.between_time("09:15", "15:30")

    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    df["rolling_vol_15"] = df["log_return"].rolling(window=15).std()

    percentile_threshold = df["rolling_vol_15"].quantile(0.95)
    df["vol_spike"] = df["rolling_vol_15"] > percentile_threshold

    df.dropna(inplace=True)

    # DAILY SUMMARY
    daily = df.resample("D").agg({
        "rolling_vol_15": ["mean", "max"],
        "vol_spike": "sum"
    })

    daily.columns = ["avg_vol", "max_vol", "spike_count"]
    daily = daily.dropna()

    # REGIME CLASSIFICATION
    low_thresh = daily["spike_count"].quantile(0.2)
    high_thresh = daily["spike_count"].quantile(0.8)

    def classify(x):
        if x <= low_thresh:
            return "Low Vol"
        elif x >= high_thresh:
            return "High Vol"
        else:
            return "Normal"

    daily["regime"] = daily["spike_count"].apply(classify)

    return df, daily