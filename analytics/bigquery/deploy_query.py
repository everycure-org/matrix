from google.cloud import bigquery

client = bigquery.Client()


def deploy_view(view_name, sql_path, dataset_id):
    with open(sql_path, "r") as file:
        query = file.read()

    view_id = f"{client.project}.{dataset_id}.{view_name}"
    view = bigquery.Table(view_id)
    view.view_query = query
    view = client.update_table(view, ["view_query"])

    print(f"✅ Deployed {view_id}")


# Example usage:
deploy_view("test_view", "./analytics/bigquery/amy_test_query.sql", "amy_test")
