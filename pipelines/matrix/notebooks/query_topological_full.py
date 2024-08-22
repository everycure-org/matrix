"""for querying."""

from google.cloud import bigquery
import pandas as pd

# Initialize the BigQuery client.
client = bigquery.Client()

categories = [
    "biolink:Drug",
    "biolink:SmallMolecule",
    "biolink:Disease",
    "biolink:PhenotypicFeature",
    "biolink:BehavioralFeature",
    "biolink:DiseaseOrPhenotypicFeature",
]
categories = str(categories).strip("[]")
query = f"SELECT id, embedding FROM `mtrx-hub-dev-3of.data_api.nodes_20240821` where CATEGORY in ({categories})"
query_job = client.query(query)
df = query_job.to_dataframe()
print("query done")
df.to_csv("node_embeddings_20240821.csv")
# Now df contains all your query results
print(df.head())
