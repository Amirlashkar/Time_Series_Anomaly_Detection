import pandas as pd
import pickle
import os

data_path = os.path.join(os.getcwd(), "timeseries")

all_timeseries = []
for filename in os.listdir(data_path):
    if "DS" not in filename:
        print(f"ON {filename}")
        df = pd.read_csv(os.path.join(data_path, filename))
        all_timeseries.append(df)
else:
    all_timeseries = pd.concat(all_timeseries)

with open("timeseries.pkl", "wb") as f:
    pickle.dump(all_timeseries, f)