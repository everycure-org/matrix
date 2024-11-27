import pandas as pd
import sqlite3
import os
from google.cloud import storage

DATA_INPUT_PATH = os.environ.get("DATA_INPUT_PATH", "gs://silc-storage/moa_vis/")
MOA_DB_PATH = os.environ.get("MOA_DB_PATH", "data/moa_extraction.db")
GCP_PROJECT = os.environ.get("GCP_PROJECT", "project-silc")


def main():
    client = storage.Client(project=GCP_PROJECT)  # define project here - where is this from
    blobs = client.list_blobs(DATA_INPUT_PATH)
    for blob in blobs:
        # Read from GS
        df = pd.read_parquet(blob.path)

        # Write to SQLite DB
        with sqlite3.connect(MOA_DB_PATH) as conn:
            # Extract table name from blob path
            table_name = os.path.splitext(os.path.basename(blob.name))[0]
            df.to_sql(table_name, conn, if_exists="replace", index=False)


if __name__ == "__main__":
    main()
