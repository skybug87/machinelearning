import os
import pandas as pd
import requests

RAW_DIR = os.path.join("dataset", "raw", "A-M")
META_PATH = os.path.join("dataset", "raw", "train_extended.csv")
OUTPUT_META = os.path.join("dataset", "raw", "raw_filtered.csv")

def filter_metadata(df):
    df = df[(df["rating"] >= 3) & (df["duration"] <= 20)]
    species_counts = df["ebird_code"].value_counts()
    valid_species = species_counts[species_counts >= 100].index
    df = df[df["ebird_code"].isin(valid_species)]
    return df

def download_and_save(row):
    ebird_code = row["ebird_code"]
    filename = row["filename"]
    url = row["url"]
    subfolder = os.path.join(RAW_DIR, ebird_code)
    os.makedirs(subfolder, exist_ok=True)
    out_path = os.path.join(subfolder, filename)
    if not os.path.exists(out_path):
        try:
            r = requests.get(url, stream=True, timeout=30)
            r.raise_for_status()
            with open(out_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        except Exception as e:
            print(f"Failed to download {url}: {e}")

import time

def main():
    df = pd.read_csv(META_PATH)
    df_filtered = filter_metadata(df)

    # Exclude files that already exist
    missing_rows = []
    for row in df_filtered.itertuples(index=False):
        ebird_code = getattr(row, "ebird_code")
        filename = getattr(row, "filename")
        out_path = os.path.join(RAW_DIR, ebird_code, filename)
        if not os.path.exists(out_path):
            missing_rows.append(row)

    total = len(missing_rows)
    print(f"Number of files to download: {total}")

    start_time = time.time()
    for idx, row in enumerate(missing_rows, 1):
        ebird_code = getattr(row, "ebird_code")
        filename = getattr(row, "filename")
        out_path = os.path.join(RAW_DIR, ebird_code, filename)
        print(f"Downloading {idx}/{total}: {filename} to {out_path}")
        t0 = time.time()
        download_and_save(row._asdict())
        elapsed = time.time() - start_time
        avg_time = elapsed / idx
        remaining = total - idx
        est_remaining = avg_time * remaining
        print(f"Estimated time remaining: {est_remaining:.1f} seconds")

    df_filtered.to_csv(OUTPUT_META, index=False)

if __name__ == "__main__":
    main()