import os
import sqlite3

import pandas as pd
from config import settings
from google.cloud import storage


def main():
    # print(settings)
    bucket, path = extract(settings.DATA_INPUT_PATH)
    bucket = storage.Client(project=settings.GCP_PROJECT).bucket(bucket)
    blobs = bucket.list_blobs(match_glob=f"{path}/*.parquet")

    for blob in blobs:
        # Read from GS
        print("Loading", blob.path)
        df = pd.read_parquet(f"gs://{bucket.name}/{blob.name}")

        # Write to SQLite DB
        with sqlite3.connect(settings.MOA_DB_PATH) as conn:
            # Extract table name from blob path
            table_name = os.path.splitext(os.path.basename(blob.name))[0]
            df.to_sql(table_name, conn, if_exists="replace", index=False)


def extract(gs_path: str):
    if not gs_path.startswith("gs://"):
        raise ValueError("Not a GCS path!")

    return gs_path[len("gs://") :].split("/", 1)


if __name__ == "__main__":
    main()
