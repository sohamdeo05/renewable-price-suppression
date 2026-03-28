import pandas as pd
import os
from functools import reduce

RAW_DIR = "data/raw"
output_path = "data/merged_data.xlsx"

def merge_raw():
    day_ahead = pd.read_excel(os.path.join(RAW_DIR, "day_ahead_prices.xlsx"))
    day_ahead['timestamp'] = pd.to_datetime(day_ahead['timestamp'], utc=True)
    day_ahead.rename(columns={'value': 'dap'}, inplace=True)

    grid_load = pd.read_excel(os.path.join(RAW_DIR, "grid_load.xlsx"))
    grid_load['timestamp'] = pd.to_datetime(grid_load['timestamp'], utc=True)
    grid_load.rename(columns={'value': 'gl'}, inplace=True)

    solar = pd.read_excel(os.path.join(RAW_DIR, "solar.xlsx"))
    solar['timestamp'] = pd.to_datetime(solar['timestamp'], utc=True)
    solar.rename(columns={'value': 'slr'}, inplace=True)

    wind_offshore = pd.read_excel(os.path.join(RAW_DIR, "wind_offshore.xlsx"))
    wind_offshore['timestamp'] = pd.to_datetime(wind_offshore['timestamp'], utc=True)
    wind_offshore.rename(columns={'value': 'wof'}, inplace=True)

    wind_onshore = pd.read_excel(os.path.join(RAW_DIR, "wind_onshore.xlsx"))
    wind_onshore['timestamp'] = pd.to_datetime(wind_onshore['timestamp'], utc=True)
    wind_onshore.rename(columns={'value': 'won'}, inplace=True) 

    dataframes = [day_ahead, grid_load, solar, wind_offshore, wind_onshore]

    merged_df = reduce(lambda left, right: pd.merge(left, right, on='timestamp', how='inner'), dataframes)

    # Since inner only keeps values for timestamps present in ALL df we check how many datapoints we lost
    print(f"Rows before merge: {len(day_ahead)} DAP rows")
    print(f"Rows after inner merge: {len(merged_df)}")

    merged_df = merged_df.sort_values('timestamp').reset_index(drop=True)
    merged_df['timestamp'] = merged_df['timestamp'].dt.tz_localize(None)
    merged_df.to_excel(output_path, index=False)

merge_raw()
