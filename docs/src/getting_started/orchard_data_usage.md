# Using Orchard Data in Matrix Pipelines

## Quick Start

The Matrix platform provides seamless access to orchard datasets from the production environments. This guide will show you how to configure and use orchard data in your Matrix pipelines.

## Accessing Orchard Data

### In Kedro Pipelines

Orchard datasets can be accessed through the existing BigQuery dataset configuration in your `catalog.yml`:

```yaml
# Access orchard production data
orchard_latest_status:
  type: matrix.datasets.gcp.SparkBigQueryDataset
  project: ec-orchard-prod
  dataset: orchard_us
  table: latest_status_20250801
  load_args:
    format: bigquery

# Access orchard development data
orchard_dev_data:
  type: matrix.datasets.gcp.SparkBigQueryDataset  
  project: ec-orchard-dev
  dataset: orchard_us
  table: your_table_name
  load_args:
    format: bigquery
```

### In Jupyter Notebooks

You can query orchard data directly using standard BigQuery client libraries:

```python
from google.cloud import bigquery

# Initialize BigQuery client
client = bigquery.Client()

# Query orchard production data
query = """
SELECT *
FROM `ec-orchard-prod.orchard_us.latest_status_20250801`
WHERE condition = 'your_filter'
LIMIT 1000
"""

# Execute query
df = client.query(query).to_dataframe()
print(df.head())
```

### Using Spark DataFrames

For large-scale data processing, you can use Spark to read orchard data:

```python
from pyspark.sql import SparkSession

# Initialize Spark session (configured automatically in Matrix environment)
spark = SparkSession.builder.getOrCreate()

# Read orchard data using Spark BigQuery connector
orchard_df = spark.read \
  .format("bigquery") \
  .option("project", "ec-orchard-prod") \
  .option("dataset", "orchard_us") \
  .option("table", "latest_status_20250801") \
  .load()

# Process the data
orchard_df.show(10)
```

## Environment Configuration

### Kubernetes Workloads

When running in the Matrix Kubernetes cluster, your workloads automatically have access to orchard datasets through the cluster's service account. No additional configuration is required.

### Local Development

For local development, ensure you have the appropriate service account access:

1. **Set up service account impersonation** based on your user group:

   ```bash
   # For internal data science team
   export SPARK_IMPERSONATION_SERVICE_ACCOUNT=sa-internal-data-science@mtrx-hub-prod-sms.iam.gserviceaccount.com
   
   # For standard contractors  
   export SPARK_IMPERSONATION_SERVICE_ACCOUNT=sa-subcon-standard@mtrx-hub-prod-sms.iam.gserviceaccount.com
   
   # For embiology contractors
   export SPARK_IMPERSONATION_SERVICE_ACCOUNT=sa-subcon-embiology@mtrx-hub-prod-sms.iam.gserviceaccount.com
   ```

2. **Authenticate with Google Cloud**:
   ```bash
   gcloud auth application-default login
   ```

3. **Verify access**:
   ```bash
   gcloud projects get-iam-policy ec-orchard-prod --flatten="bindings[].members" --format="table(bindings.role)"
   ```

## Available Orchard Projects

- **Development**: `ec-orchard-dev` - Use for testing and development
- **Production**: `ec-orchard-prod` - Production orchard datasets

## Common Datasets

### Latest Status Data

The most commonly used orchard dataset contains the latest status information:

- **Project**: `ec-orchard-prod`
- **Dataset**: `orchard_us`
- **Table**: `latest_status_20250801` (table names may include date suffixes)

Example query structure:
```sql
SELECT 
  field1,
  field2,
  field3
FROM `ec-orchard-prod.orchard_us.latest_status_20250801`
WHERE your_conditions
```

## Best Practices

### Data Access Patterns

1. **Use appropriate environments**: Query development data (`ec-orchard-dev`) during pipeline development and testing
2. **Optimize queries**: Use proper WHERE clauses and LIMIT statements to avoid excessive data transfer
3. **Monitor costs**: Be mindful of BigQuery usage as orchard projects have their own quotas and billing

### Security Considerations

1. **Read-only access**: All Matrix access to orchard data is read-only - you cannot modify orchard data
2. **Audit compliance**: All data access is logged and auditable through GCP audit logs
3. **Data governance**: Follow your organization's data governance policies when working with orchard data

### Error Handling

Common issues and solutions:

```python
from google.cloud import bigquery
from google.cloud.exceptions import NotFound, Forbidden

try:
    client = bigquery.Client()
    query_job = client.query(your_query)
    results = query_job.result()
except Forbidden as e:
    print(f"Access denied: {e}")
    print("Check your service account permissions and IAM bindings")
except NotFound as e:
    print(f"Resource not found: {e}")  
    print("Verify the project, dataset, and table names")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Example Pipeline Node

Here's a complete example of a Kedro node that processes orchard data:

```python
from typing import Dict, Any
import pandas as pd
from kedro.pipeline import node, Pipeline

def process_orchard_data(orchard_data: pd.DataFrame) -> Dict[str, Any]:
    """Process orchard latest status data.
    
    Args:
        orchard_data: DataFrame containing orchard data
        
    Returns:
        Dictionary with processed results
    """
    # Your processing logic here
    processed_count = len(orchard_data)
    
    # Example analysis
    status_summary = orchard_data.groupby('status_field').size().to_dict()
    
    return {
        "total_records": processed_count,
        "status_summary": status_summary,
        "sample_data": orchard_data.head(10).to_dict('records')
    }

def create_orchard_processing_pipeline() -> Pipeline:
    """Create a pipeline that processes orchard data."""
    return Pipeline([
        node(
            func=process_orchard_data,
            inputs="orchard_latest_status",  # References catalog.yml entry
            outputs="orchard_processed_results",
            name="process_orchard_data_node"
        )
    ])
```

## Troubleshooting

### Permission Denied Errors

If you encounter permission denied errors:

1. Verify your service account has the correct orchard access roles
2. Check that service account impersonation is configured correctly
3. Ensure you're a member of the appropriate Google Groups

### Project Not Found

If you get "project not found" errors:

1. Verify the project ID is correct (`ec-orchard-prod` or `ec-orchard-dev`)
2. Check that the service account has access to the specific orchard project
3. Ensure the BigQuery API is enabled

### Quota Exceeded

If you hit quota limits:

1. Optimize your queries to reduce data transfer
2. Use appropriate sampling techniques for large datasets
3. Coordinate with the orchard team if you need higher quotas

For more detailed troubleshooting, see the [Orchard Data Access Infrastructure Documentation](../infrastructure/orchard_data_access.md).

## Support

If you need help with orchard data access:

1. Check the infrastructure documentation for detailed implementation details
2. Review audit logs in the GCP console for access attempts
3. Contact the infrastructure team for IAM or permission issues
4. Reach out to the orchard team for data-specific questions
