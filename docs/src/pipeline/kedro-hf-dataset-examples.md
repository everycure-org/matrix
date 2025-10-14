# Kedro HuggingFace Dataset Examples

This document provides practical examples of using the Kedro HuggingFace datasets for data distribution and consumption.

## Table of Contents

- [Basic Configuration](#basic-configuration)
- [Authentication Setup](#authentication-setup)
- [Dataset Types](#dataset-types)
- [Pipeline Integration](#pipeline-integration)
- [Advanced Usage](#advanced-usage)
- [Troubleshooting](#troubleshooting)

## Basic Configuration

### Catalog Configuration

Add HuggingFace datasets to your `catalog.yml`:

```yaml
# Knowledge Graph Nodes
kg_nodes:
  type: matrix.datasets.huggingface.HuggingFaceParquetDataset
  repo_id: "everycure/matrix-kg-v1.0"
  filename: "nodes.parquet"
  revision: "main"
  credentials: huggingface_token
  load_args:
    pandas_args:
      columns: ["id", "name", "category", "is_drug", "is_disease"]
  save_args:
    commit_message: "Update knowledge graph nodes"
    commit_description: "Updated with latest processing pipeline"
    pandas_args:
      compression: "snappy"
  metadata:
    description: "Knowledge graph nodes for drug repurposing"
    tags: ["knowledge-graph", "nodes", "biomedical"]

# Knowledge Graph Edges
kg_edges:
  type: matrix.datasets.huggingface.HuggingFaceParquetDataset
  repo_id: "everycure/matrix-kg-v1.0"
  filename: "edges.parquet"
  revision: "main"
  credentials: huggingface_token
  metadata:
    description: "Knowledge graph edges and relationships"
    tags: ["knowledge-graph", "edges", "relationships"]

# Drug-Disease Predictions
predictions:
  type: matrix.datasets.huggingface.HuggingFaceCSVDataset
  repo_id: "everycure/matrix-predictions-v1.0"
  filename: "drug_disease_predictions.csv"
  revision: "main"
  credentials: huggingface_token
  save_args:
    commit_message: "Update predictions"
    pandas_args:
      float_format: "%.6f"

# Large Embeddings (using Xet for efficiency)
node_embeddings:
  type: matrix.datasets.huggingface.HuggingFaceXetDataset
  repo_id: "everycure/matrix-embeddings-v1.0"
  filename: "node_embeddings.xet"
  revision: "main"
  credentials: huggingface_token
  xet_args:
    chunk_size: "64MB"
    compression: "lz4"
  load_args:
    lazy: true
```

## Authentication Setup

### Environment Variable (Recommended)

```bash
# Set HuggingFace token in environment
export HF_TOKEN="hf_your_token_here"

# Or use the alternative variable name
export HUGGINGFACE_HUB_TOKEN="hf_your_token_here"
```

### Credentials File

```yaml
# credentials.yml
huggingface_token:
  token: "${oc.env:HF_TOKEN}"

# Or use file-based token storage
huggingface_token:
  token_file: "~/.cache/huggingface/token"
```

### Programmatic Login

```python
from huggingface_hub import login

# Login programmatically (for notebooks/scripts)
login(token="hf_your_token_here")
```

## Dataset Types

### Parquet Datasets (Recommended for Structured Data)

```yaml
structured_data:
  type: matrix.datasets.huggingface.HuggingFaceParquetDataset
  repo_id: "everycure/my-dataset"
  filename: "data.parquet"
  load_args:
    pandas_args:
      columns: ["id", "value"]  # Load specific columns
      filters: [("category", "==", "drug")]  # Apply filters
  save_args:
    pandas_args:
      compression: "snappy"
      index: false
```

### CSV Datasets (For Compatibility)

```yaml
tabular_data:
  type: matrix.datasets.huggingface.HuggingFaceCSVDataset
  repo_id: "everycure/my-dataset"
  filename: "data.csv"
  load_args:
    pandas_args:
      sep: ","
      encoding: "utf-8"
      dtype: {"id": str, "score": float}
  save_args:
    pandas_args:
      index: false
      float_format: "%.4f"
```

### JSON Datasets (For Semi-structured Data)

```yaml
metadata:
  type: matrix.datasets.huggingface.HuggingFaceJSONDataset
  repo_id: "everycure/my-dataset"
  filename: "metadata.json"
  load_args:
    json_args:
      object_pairs_hook: collections.OrderedDict
  save_args:
    json_args:
      indent: 2
      sort_keys: true
```

### Xet Datasets (For Large Files)

```yaml
large_dataset:
  type: matrix.datasets.huggingface.HuggingFaceXetDataset
  repo_id: "everycure/large-dataset"
  filename: "embeddings.xet"
  xet_args:
    chunk_size: "128MB"
    compression: "gzip"
  load_args:
    lazy: true  # Enable lazy loading
```

## Pipeline Integration

### Basic Pipeline Node

```python
from kedro import pipeline, node
import pandas as pd

def process_knowledge_graph(kg_nodes: pd.DataFrame) -> pd.DataFrame:
    """Process knowledge graph nodes from HuggingFace Hub.
    
    Args:
        kg_nodes: DataFrame loaded from HuggingFace repository
        
    Returns:
        Processed DataFrame to be saved back to HuggingFace
    """
    # Remove duplicates
    processed = kg_nodes.drop_duplicates(subset=['id'])
    
    # Add processing timestamp
    processed['processed_at'] = pd.Timestamp.now()
    
    # Filter for valid entries
    processed = processed[processed['name'].notna()]
    
    return processed

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=process_knowledge_graph,
            inputs="kg_nodes",  # From HuggingFace Hub
            outputs="processed_kg_nodes",  # To HuggingFace Hub
            name="process_kg_nodes"
        )
    ])
```

### Data Release Pipeline

```python
def publish_dataset_release(
    kg_nodes: pd.DataFrame,
    kg_edges: pd.DataFrame,
    predictions: pd.DataFrame
) -> dict:
    """Create a complete dataset release.
    
    This function processes multiple datasets and prepares them
    for publication to HuggingFace Hub.
    """
    release_info = {
        "version": "v1.2.0",
        "release_date": pd.Timestamp.now().isoformat(),
        "datasets": {
            "nodes": len(kg_nodes),
            "edges": len(kg_edges),
            "predictions": len(predictions)
        }
    }
    
    return release_info

def create_release_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=publish_dataset_release,
            inputs=["kg_nodes", "kg_edges", "predictions"],
            outputs="release_metadata",
            name="create_release"
        )
    ])
```

### Programmatic Usage

```python
from matrix.datasets.huggingface import HuggingFaceParquetDataset
import pandas as pd

# Direct dataset usage
dataset = HuggingFaceParquetDataset(
    repo_id="everycure/matrix-kg-v1.0",
    filename="nodes.parquet",
    credentials={"token": "hf_your_token"}
)

# Load data
kg_nodes = dataset.load()
print(f"Loaded {len(kg_nodes)} nodes")

# Process data
processed_nodes = kg_nodes.drop_duplicates()

# Save back to repository
dataset.save(processed_nodes)
```

## Advanced Usage

### Versioned Datasets

```yaml
# Load specific version
kg_nodes_v1:
  type: matrix.datasets.huggingface.HuggingFaceParquetDataset
  repo_id: "everycure/matrix-kg"
  filename: "nodes.parquet"
  revision: "v1.0.0"  # Specific version tag

# Load from development branch
kg_nodes_dev:
  type: matrix.datasets.huggingface.HuggingFaceParquetDataset
  repo_id: "everycure/matrix-kg"
  filename: "nodes.parquet"
  revision: "development"  # Branch name

# Load from specific commit
kg_nodes_commit:
  type: matrix.datasets.huggingface.HuggingFaceParquetDataset
  repo_id: "everycure/matrix-kg"
  filename: "nodes.parquet"
  revision: "a1b2c3d4"  # Commit hash
```

### Private Repositories

```yaml
private_dataset:
  type: matrix.datasets.huggingface.HuggingFaceParquetDataset
  repo_id: "everycure/private-research-data"
  filename: "sensitive_data.parquet"
  credentials: huggingface_token  # Token with repo access
  save_args:
    private: true  # Ensure repository remains private
```

### Batch Operations

```python
def batch_upload_datasets():
    """Upload multiple datasets in batch."""
    datasets = [
        ("nodes.parquet", nodes_df),
        ("edges.parquet", edges_df),
        ("embeddings.parquet", embeddings_df)
    ]
    
    for filename, data in datasets:
        dataset = HuggingFaceParquetDataset(
            repo_id="everycure/matrix-kg-batch",
            filename=filename,
            save_args={
                "commit_message": f"Batch update: {filename}",
                "commit_description": "Automated batch upload"
            }
        )
        dataset.save(data)
        print(f"Uploaded {filename}")
```

### Custom Metadata and Tags

```yaml
research_dataset:
  type: matrix.datasets.huggingface.HuggingFaceParquetDataset
  repo_id: "everycure/research-v2"
  filename: "results.parquet"
  metadata:
    description: |
      Research dataset containing drug repurposing predictions
      generated using the MATRIX pipeline v2.0.
      
      This dataset includes:
      - Drug-disease association scores
      - Confidence intervals
      - Supporting evidence links
      
      For more information, see: https://docs.everycure.org
    tags:
      - drug-repurposing
      - biomedical
      - machine-learning
      - predictions
      - everycure
    license: "CC-BY-4.0"
    citation: |
      @dataset{everycure_matrix_2024,
        title={MATRIX Drug Repurposing Dataset v2.0},
        author={Every Cure Research Team},
        year={2024},
        publisher={HuggingFace Hub},
        url={https://huggingface.co/datasets/everycure/research-v2}
      }
```

## Integration with Existing Pipelines

### GCS to HuggingFace Migration

```python
def migrate_gcs_to_huggingface(gcs_data: pd.DataFrame) -> pd.DataFrame:
    """Migrate data from GCS to HuggingFace Hub.
    
    This function can be used in a pipeline to gradually migrate
    datasets from Google Cloud Storage to HuggingFace Hub.
    """
    # Process data for HuggingFace format
    processed_data = gcs_data.copy()
    
    # Add metadata for HuggingFace
    processed_data.attrs['source'] = 'GCS Migration'
    processed_data.attrs['migration_date'] = pd.Timestamp.now().isoformat()
    
    return processed_data

# Pipeline configuration
migration_pipeline = pipeline([
    node(
        func=migrate_gcs_to_huggingface,
        inputs="gcs_dataset",  # Existing GCS dataset
        outputs="hf_dataset",  # New HuggingFace dataset
        name="migrate_to_hf"
    )
])
```

### Hybrid Storage Strategy

```yaml
# Use both GCS and HuggingFace for different purposes
internal_data:
  type: matrix_gcp_datasets.gcp.SparkBigQueryDataset
  # Internal processing data stays in GCS
  
public_data:
  type: matrix.datasets.huggingface.HuggingFaceParquetDataset
  repo_id: "everycure/public-dataset"
  filename: "public_data.parquet"
  # Public data goes to HuggingFace for sharing
```

## Troubleshooting

### Common Issues and Solutions

#### Authentication Problems

```python
# Check if token is valid
from huggingface_hub import whoami

try:
    user_info = whoami()
    print(f"Authenticated as: {user_info['name']}")
except Exception as e:
    print(f"Authentication failed: {e}")
    print("Please check your HF_TOKEN environment variable")
```

#### Repository Access Issues

```python
# Check repository access
from huggingface_hub import HfApi

api = HfApi()
try:
    repo_info = api.repo_info("everycure/matrix-kg-v1.0")
    print(f"Repository accessible: {repo_info.id}")
except Exception as e:
    print(f"Cannot access repository: {e}")
```

#### Large File Upload Issues

```yaml
# For large files, use Xet dataset or configure Git LFS
large_file:
  type: matrix.datasets.huggingface.HuggingFaceXetDataset
  repo_id: "everycure/large-dataset"
  filename: "large_file.xet"
  xet_args:
    chunk_size: "32MB"  # Smaller chunks for better reliability
```

#### Network and Timeout Issues

```python
# Configure retry behavior
dataset = HuggingFaceParquetDataset(
    repo_id="everycure/dataset",
    filename="data.parquet",
    load_args={
        "download_args": {
            "resume_download": True,  # Resume interrupted downloads
            "local_files_only": False,  # Allow network access
        }
    }
)
```

### Debug Mode

```python
import logging

# Enable debug logging for HuggingFace operations
logging.getLogger("huggingface_hub").setLevel(logging.DEBUG)
logging.getLogger("matrix.datasets.huggingface").setLevel(logging.DEBUG)

# Run your dataset operations with detailed logging
dataset = HuggingFaceParquetDataset(...)
data = dataset.load()  # Will show detailed debug information
```

### Performance Optimization

```yaml
# Optimize for performance
optimized_dataset:
  type: matrix.datasets.huggingface.HuggingFaceParquetDataset
  repo_id: "everycure/dataset"
  filename: "data.parquet"
  load_args:
    pandas_args:
      columns: ["id", "score"]  # Load only needed columns
      filters: [("category", "in", ["drug", "disease"])]  # Filter early
    download_args:
      cache_dir: "/tmp/hf_cache"  # Use fast local storage for cache
```

## Best Practices

### Repository Organization

```
everycure/matrix-kg-v1.0/
├── README.md                    # Dataset description
├── nodes.parquet               # Node data
├── edges.parquet               # Edge data
├── metadata.json               # Dataset metadata
└── embeddings/
    ├── node_embeddings.xet     # Large embedding files
    └── edge_embeddings.xet
```

### Commit Messages

```yaml
save_args:
  commit_message: "feat: add drug-target interactions v2.1"
  commit_description: |
    - Added 15,000 new drug-target interactions
    - Updated confidence scores using improved model
    - Fixed data quality issues in previous version
    - Includes validation results and metrics
```

### Version Management

```python
# Use semantic versioning for datasets
def create_versioned_dataset(version: str):
    return HuggingFaceParquetDataset(
        repo_id=f"everycure/matrix-kg-{version}",
        filename="nodes.parquet",
        save_args={
            "commit_message": f"Release {version}",
            "commit_description": f"Official release of MATRIX KG {version}"
        }
    )

# Create releases
v1_dataset = create_versioned_dataset("v1.0.0")
v2_dataset = create_versioned_dataset("v2.0.0")
```

---

*This documentation was partially generated using AI assistance.*