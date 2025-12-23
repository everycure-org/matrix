# Data Publication Pipeline

This pipeline publishes MATRIX datasets to Hugging Face Hub for public access and sharing.

## Overview

The data publication pipeline takes processed datasets from the MATRIX integration pipeline and publishes them to Hugging Face Hub as Parquet datasets. Currently, the pipeline publishes:

- **Knowledge Graph Edges** → [everycure/kg-edges](https://huggingface.co/datasets/everycure/kg-edges)
- **Knowledge Graph Nodes** → [everycure/kg-nodes](https://huggingface.co/datasets/everycure/kg-nodes)

## How It Works

### Pipeline Structure

The pipeline is intentionally simple: apart from the pipeline definition (which passes through data (`lambda x: x`) and has nearly no custom code), the actual upload logic is handled by the `HFIterableDataset` custom Kedro dataset type.

### Upload Process

When a dataset is saved, `HFIterableDataset` performs the following:

1. **Convert DataFrame** - Converts Spark/Pandas/Polars DataFrame to a Hugging Face `Dataset`
2. **Push to Hub** - Uploads the dataset to the specified repository as Parquet files
3. **Verify Upload** - Runs a 3-step verification:
   - Confirms the commit SHA matches
   - Lists files on the Hub to ensure they exist
   - Attempts a streaming load to verify the dataset is readable

### Authentication

The dataset authenticates with Hugging Face using a token resolved in this order:

1. Explicit `token` parameter in catalog config (not recommended)
2. Kedro credentials mapping (via `credentials:` key)
3. `HF_TOKEN` environment variable (RECOMMENDED)

## Configuration

### Catalog Configuration

Datasets are configured in `conf/base/data_publication/catalog.yml`:

```yaml
data_publication.prm.kg_edges_hf_published:
  type: matrix_gcp_datasets.huggingface.HFIterableDataset
  repo_id: everycure/kg-edges
  split: train
  dataframe_type: spark
  data_dir: data/edges

data_publication.prm.kg_nodes_hf_published:
  type: matrix_gcp_datasets.huggingface.HFIterableDataset
  repo_id: everycure/kg-nodes
  split: train
  dataframe_type: spark
  data_dir: data/nodes
```

### Configuration Options

| Parameter | Required | Default | Description |
|-----------|----------|---------|-------------|
| `repo_id` | Yes | - | HF Hub repository (e.g., `everycure/kg-edges`) |
| `split` | No | `train` | Dataset split name |
| `dataframe_type` | No | `spark` | Input type: `spark`, `pandas`, or `polars` |
| `data_dir` | No | - | Subdirectory in the repo for data files |
| `config_name` | No | - | HF dataset config/subset name |
| `private` | No | `false` | Whether to create a private repository |
| `credentials` | No | - | Kedro credentials key for token |
| `token_key` | No | `HF_TOKEN` | Key within credentials mapping |
| `token` | No | - | Direct token (prefer credentials or env var) |

### Credentials Setup

**Option 1: Environment Variable (Recommended)**

```bash
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxx
```

**Option 2: Kedro Credentials**

Add to `conf/local/credentials.yml`:

```yaml
hf:
  HF_TOKEN: ${oc.env:HF_TOKEN}
```

Then reference in catalog:

```yaml
data_publication.prm.kg_edges_hf_published:
  type: matrix_gcp_datasets.huggingface.HFIterableDataset
  repo_id: everycure/kg-edges
  credentials: hf
  # ... other options
```

Get a token from: https://huggingface.co/settings/tokens

**Token Permissions Required:**
- `write` access to the target repository
- For private repos, the token must have access to read/write private datasets

## Running the Pipeline

```bash
# Run the data publication pipeline
kedro run --pipeline data_publication

# Run with a specific environment
kedro run --pipeline data_publication --env prod
```

## Adding a New Dataset

To add a new dataset to the pipeline:

### Step 1: Create the Repository on Hugging Face Hub

1. Go to https://huggingface.co/new-dataset
2. Choose the organization (e.g., `everycure`)
3. Enter the dataset name (e.g., `disease-list`)
4. Select visibility (public or private)
5. Click "Create dataset"
6. Update your TOKEN to be able to write to this new dataset (important!)

**Important:** The repository must exist before running the pipeline. The `HFIterableDataset` pushes to an existing repo; it does not create new repositories.

### Step 2: Add Catalog Entry

Add the dataset to `conf/base/data_publication/catalog.yml`:

```yaml
data_publication.prm.disease_list_hf_published:
  type: matrix_gcp_datasets.huggingface.HFIterableDataset
  repo_id: everycure/disease-list
  split: train
  dataframe_type: spark  # or pandas/polars depending on input
  data_dir: data/diseases  # optional subdirectory
```

### Step 3: Add Pipeline Node

Add a node in `pipeline.py`:

```python
node(
    func=lambda x: x,
    inputs="disease_list.prm.disease_list",  # upstream dataset
    outputs="data_publication.prm.disease_list_hf_published",
    name="publish_disease_list_node",
),
```

### Step 4: Test the Publication

```bash
# Run just the new node
kedro run --nodes publish_disease_list_node
```

## Troubleshooting

### Token Not Found

```
ValueError: Hugging Face token not found. Provide via catalog credentials,
explicit dataset `token`, or HF_TOKEN environment variable.
```

**Solution:** Set the `HF_TOKEN` environment variable or configure credentials.

### Permission Denied (403/401)

```
requests.exceptions.HTTPError: 403 Client Error: Forbidden
```

**Solutions:**

- Ensure your token has write access to the repository
- Verify you have access to the organization
- Check that the repository exists on Hugging Face Hub

### Repository Not Found (404)

```
requests.exceptions.HTTPError: 404 Client Error: Not Found
```

**Solution:** Create the repository on Hugging Face Hub first (see "Adding a New Dataset" above).

### Verification Timeout

```
TimeoutError: Timed out waiting for SHA to update
```

**Possible causes:**

- Hugging Face Hub is experiencing delays
- Network issues
- Very large dataset taking longer to process

The upload may still have succeeded. Check the repository on Hugging Face Hub directly.
