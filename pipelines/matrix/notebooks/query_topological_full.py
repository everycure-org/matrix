"""for querying."""

from google.cloud import bigquery
import pandas as pd

# Initialize the BigQuery client.
client = bigquery.Client()

embed = pd.read_csv("topological_embeddings_20240807.csv", index_col=0)
drug_ids = embed.id.values

df_list = []
for i in range(0, len(drug_ids), 1000):
    batch_ids = drug_ids[i : i + 1000]
    ids_str = '", "'.join(batch_ids)
    print(ids_str)
    query = f'SELECT id, topological_embedding FROM `mtrx-hub-dev-3of.data_api.nodes_20240820` WHERE id IN ("{ids_str}")'
    query_job = client.query(query)
    results = query_job.to_dataframe()
    df_list.append(results)
df = pd.concat(df_list, ignore_index=True)
print("query done")
df.to_csv("topological_embeddings_20240820.csv")
# Now df contains all your query results
print(df.head())
