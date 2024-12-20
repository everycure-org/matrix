import pandas as pd
import os

# directory = "/Users/Kathleen/Documents/GitHub/matrix/pipelines/matrix/data/qc/int/robokop/qc_results"
# directory = "/Users/Kathleen/Documents/GitHub/matrix/pipelines/matrix/data/releases/local-release/datasets/integration/int/robokop/nodes_norm_mapping"
# directory = "/Users/Kathleen/Documents/GitHub/matrix/pipelines/matrix/data/releases/local-release/datasets/integration/int/robokop/nodes"
# directory = "/Users/Kathleen/Documents/GitHub/matrix/pipelines/matrix/data/ingestion/int/robokop-kg/c5ec1f282158182f/edges"
directory = "/Users/Kathleen/Documents/GitHub/matrix/pipelines/matrix/data/qc/int/robokop"

dataframes = []

for filename in os.listdir(directory):
    filepath = os.path.join(directory, filename)

    if filename.endswith(".parquet"):
        df = pd.read_parquet(filepath)
        print(df)
        dataframes.append(df)

# Optionally, concatenate all DataFrames into a single DataFrame
if dataframes:
    combined_df = pd.concat(dataframes, ignore_index=True)
    print("All Parquet files have been loaded and concatenated.")
else:
    print("No Parquet files found in the directory.")

# combined_df.to_csv('test_nodes.csv')
