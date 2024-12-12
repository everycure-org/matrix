import pandas as pd
import sqlite3
import os
from google.cloud import storage

from config import settings


def main():
    bucket, path = extract(settings.data_input_path)
    bucket = storage.Client(project=settings.gcp_project).bucket(bucket)
    blobs = bucket.list_blobs(match_glob=f"{path}/*.parquet")

    for blob in blobs:
        # Read from GS
        print("Loading", blob.path)
        df = pd.read_parquet(f"gs://{bucket.name}/{blob.name}")

        # Write to SQLite DB
        with sqlite3.connect(settings.moa_db_path) as conn:
            # Extract table name from blob path
            table_name = os.path.splitext(os.path.basename(blob.name))[0]
            df.to_sql(table_name, conn, if_exists="replace", index=False)


def extract(gs_path: str):
    if not gs_path.startswith("gs://"):
        raise ValueError("Not a GCS path!")

    return gs_path[len("gs://") :].split("/", 1)


if __name__ == "__main__":
    main()
