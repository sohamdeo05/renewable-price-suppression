import requests
import os
import time
import pandas as pd

START_MS = 1538344800000  # Jan 1 2016 UTC in milliseconds

SERIES = [
    (4169, "day_ahead_prices"), # 1538344800000 Sep 30 2018 22:00:00 GMT+0000
    (410,  "grid_load"), 	    # 1419807600000 Dec 28 2014 23:00:00 GMT+0000
    (4068, "solar"),            # 1419807600000 Dec 28 2014 23:00:00 GMT+0000
    (1225, "wind_offshore"),    # 1419807600000 Dec 28 2014 23:00:00 GMT+0000
    (4067, "wind_onshore"),     # 1419807600000 Dec 28 2014 23:00:00 GMT+0000
]

RAW_DIR    = "data/raw"
CHUNKS_DIR = "data/chunks"

def get_timestamps(filter_id, start_ms):
    """
    Fetches the index of available data chunks for a given SMARD filter_id
    and returns timestamps that are greater than or equal to start_ms.
    """
    url = f"https://www.smard.de/app/chart_data/{filter_id}/DE/index_hour.json"

    response = requests.get(url)
    response.raise_for_status()

    data = response.json()
    all_timestamps = data.get("timestamps", [])

    filtered_timestamps = [ts for ts in all_timestamps if ts >= start_ms]
    
    return filtered_timestamps

def get_data(filter_id, timestamp):
    url = f"https://www.smard.de/app/chart_data/{filter_id}/DE/{filter_id}_DE_hour_{timestamp}.json"
    
    for attempt in range(5):
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            series = response.json().get("series", [])
            df = pd.DataFrame(series, columns=["timestamp", "value"])
            df["timestamp"] = (
                pd.to_datetime(df["timestamp"], unit="ms")
                .dt.tz_localize("UTC")
                .dt.tz_convert("Europe/Berlin")
            )
            return df
        except Exception as e:
            print(f"  Attempt {attempt+1} failed: {e}")
            time.sleep(3)
    
    raise Exception(f"Failed to fetch {url} after 5 attempts")

def fetch_series(filter_id, name, start_ms):
    series_chunk_dir = os.path.join(CHUNKS_DIR, name)
    os.makedirs(series_chunk_dir, exist_ok=True)

    final_path = os.path.join(RAW_DIR, f"{name}.xlsx")
    if os.path.exists(final_path):
        print(f"[SKIP] {name} — final Excel already exists.")
        return

    timestamps = get_timestamps(filter_id, start_ms)
    total = len(timestamps)

    for i, ts in enumerate(timestamps):
        chunk_path = os.path.join(series_chunk_dir, f"{ts}.csv")

        if os.path.exists(chunk_path):
            print(f"[{name}] {i+1}/{total} chunk {ts} already cached, skipping.")
            continue

        print(f"[{name}] {i+1}/{total} fetching {ts}...")
        df_chunk = get_data(filter_id, ts)
        df_chunk.to_csv(chunk_path, index=False)

    print(f"[{name}] All chunks fetched. Combining...")
    all_chunks = []
    for ts in timestamps:
        chunk_path = os.path.join(series_chunk_dir, f"{ts}.csv")
        all_chunks.append(pd.read_csv(chunk_path))

    df_final = pd.concat(all_chunks, ignore_index=True)
    df_final = df_final.dropna(subset=["value"])
    df_final = df_final.reset_index(drop=True)

    os.makedirs(RAW_DIR, exist_ok=True)
    df_final.to_excel(final_path, index=False)
    print(f"[{name}] Done. {len(df_final)} rows saved to {final_path}")

if __name__ == "__main__":
    for filter_id, name in SERIES:
        fetch_series(filter_id, name, START_MS)
