#!/usr/bin/env python3
# NOTE: This file was partially generated using AI assistance.

"""
Prototype script for HuggingFace dataset integration with MATRIX.

This script demonstrates how to:
1. Load data from existing GCS releases
2. Upload to Hugging Face Hub using the new HuggingFaceDataset
3. Load data back from Hugging Face

Usage:
    python examples/huggingface_prototype.py --help
"""

import os
import sys
from pathlib import Path
import pandas as pd
import typer
from typing import Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from matrix.datasets.huggingface import HuggingFaceDataset

app = typer.Typer()


def create_sample_data(limit: int = 100) -> dict:
    """Create sample knowledge graph data for testing."""
    
    # Sample nodes data
    nodes_data = {
        "id": [f"node_{i}" for i in range(limit)],
        "name": [f"Entity {i}" for i in range(limit)],
        "type": ["drug" if i % 2 == 0 else "disease" for i in range(limit)],
        "is_drug": [i % 2 == 0 for i in range(limit)],
        "is_disease": [i % 2 == 1 for i in range(limit)],
        "topological_embedding": [[0.1 * i, 0.2 * i, 0.3 * i] for i in range(limit)]
    }
    
    # Sample edges data  
    edges_data = {
        "source": [f"node_{i}" for i in range(0, limit-1, 2)],
        "target": [f"node_{i}" for i in range(1, limit, 2)],
        "relation": ["treats"] * (limit // 2),
        "weight": [0.5 + (i * 0.01) for i in range(limit // 2)]
    }
    
    return {
        "nodes": pd.DataFrame(nodes_data),
        "edges": pd.DataFrame(edges_data)
    }


@app.command()
def test_single_dataframe(
    repo_id: str = "test-user/matrix-kg-nodes-test",
    token: Optional[str] = None,
    limit: int = 100,
    private: bool = True
):
    """Test uploading and downloading a single DataFrame."""
    
    typer.echo(f"🧪 Testing single DataFrame upload/download with {limit} rows...")
    
    # Create sample data
    sample_data = create_sample_data(limit)
    nodes_df = sample_data["nodes"]
    
    typer.echo(f"📊 Created sample nodes DataFrame: {len(nodes_df)} rows, {len(nodes_df.columns)} columns")
    
    # Initialize dataset
    dataset = HuggingFaceDataset(
        repo_id=repo_id,
        token=token or os.getenv("HF_TOKEN"),
        private=private,
        save_args={
            "commit_message": f"Test upload: {limit} sample nodes",
            "commit_description": "Testing MATRIX HuggingFace dataset integration"
        }
    )
    
    try:
        # Upload
        typer.echo(f"⬆️  Uploading to {repo_id}...")
        dataset.save(nodes_df)
        typer.echo("✅ Upload successful!")
        
        # Download
        typer.echo("⬇️  Downloading back...")
        loaded_df = dataset.load()
        typer.echo(f"✅ Download successful: {len(loaded_df)} rows, {len(loaded_df.columns)} columns")
        
        # Verify
        if len(loaded_df) == len(nodes_df) and len(loaded_df.columns) == len(nodes_df.columns):
            typer.echo("✅ Data integrity verified!")
        else:
            typer.echo("❌ Data integrity check failed!")
            
    except Exception as e:
        typer.echo(f"❌ Error: {e}")
        raise typer.Exit(1)


@app.command()
def test_multiple_dataframes(
    repo_id: str = "test-user/matrix-kg-complete-test",
    token: Optional[str] = None,
    limit: int = 100,
    private: bool = True
):
    """Test uploading and downloading multiple DataFrames."""
    
    typer.echo(f"🧪 Testing multiple DataFrame upload/download with {limit} rows...")
    
    # Create sample data
    sample_data = create_sample_data(limit)
    
    typer.echo(f"📊 Created sample data:")
    typer.echo(f"  - Nodes: {len(sample_data['nodes'])} rows")
    typer.echo(f"  - Edges: {len(sample_data['edges'])} rows")
    
    # Initialize dataset
    dataset = HuggingFaceDataset(
        repo_id=repo_id,
        token=token or os.getenv("HF_TOKEN"),
        private=private,
        save_args={
            "commit_message": f"Test upload: complete KG with {limit} nodes",
            "commit_description": "Testing MATRIX HuggingFace dataset with nodes and edges"
        }
    )
    
    try:
        # Upload
        typer.echo(f"⬆️  Uploading to {repo_id}...")
        dataset.save(sample_data)
        typer.echo("✅ Upload successful!")
        
        # Download
        typer.echo("⬇️  Downloading back...")
        loaded_data = dataset.load()
        typer.echo(f"✅ Download successful:")
        typer.echo(f"  - Loaded splits: {list(loaded_data.keys())}")
        
        for split_name, df in loaded_data.items():
            typer.echo(f"  - {split_name}: {len(df)} rows, {len(df.columns)} columns")
        
        # Verify
        if ("nodes" in loaded_data and "edges" in loaded_data and
            len(loaded_data["nodes"]) == len(sample_data["nodes"]) and
            len(loaded_data["edges"]) == len(sample_data["edges"])):
            typer.echo("✅ Data integrity verified!")
        else:
            typer.echo("❌ Data integrity check failed!")
            
    except Exception as e:
        typer.echo(f"❌ Error: {e}")
        raise typer.Exit(1)


@app.command()
def test_load_existing(
    repo_id: str,
    token: Optional[str] = None,
    split: Optional[str] = None
):
    """Test loading from an existing HuggingFace dataset."""
    
    typer.echo(f"🧪 Testing load from existing dataset: {repo_id}")
    
    # Initialize dataset
    dataset = HuggingFaceDataset(
        repo_id=repo_id,
        token=token or os.getenv("HF_TOKEN"),
        load_args={"split": split} if split else {}
    )
    
    try:
        # Check if exists
        if dataset.exists():
            typer.echo("✅ Dataset exists on Hugging Face Hub")
        else:
            typer.echo("❌ Dataset not found on Hugging Face Hub")
            raise typer.Exit(1)
        
        # Load
        typer.echo("⬇️  Loading dataset...")
        data = dataset.load()
        
        if isinstance(data, pd.DataFrame):
            typer.echo(f"✅ Loaded single DataFrame: {len(data)} rows, {len(data.columns)} columns")
            typer.echo(f"📊 Columns: {list(data.columns)}")
        elif isinstance(data, dict):
            typer.echo(f"✅ Loaded DatasetDict with {len(data)} splits:")
            for split_name, df in data.items():
                typer.echo(f"  - {split_name}: {len(df)} rows, {len(df.columns)} columns")
        else:
            typer.echo(f"❓ Unexpected data type: {type(data)}")
            
    except Exception as e:
        typer.echo(f"❌ Error: {e}")
        raise typer.Exit(1)


@app.command()
def prototype_real_data(
    nodes_path: str = "gs://mtrx-us-central1-hub-dev-storage/kedro/data/releases/v0.10.0/datasets/integration/prm/unified/nodes/*.parquet",
    edges_path: str = "gs://mtrx-us-central1-hub-dev-storage/kedro/data/releases/v0.10.0/datasets/integration/prm/unified/edges/*.parquet",
    repo_id: str = "everycure/matrix-kg-prototype",
    token: Optional[str] = None,
    limit: int = 1000,
    private: bool = True
):
    """Prototype with real MATRIX data (requires GCS access and Spark)."""
    
    typer.echo(f"🧪 Prototyping with real MATRIX data (limited to {limit} rows)...")
    
    try:
        # This would require Spark and GCS access
        typer.echo("⚠️  This command requires:")
        typer.echo("  - PySpark environment")
        typer.echo("  - GCS credentials")
        typer.echo("  - Access to MATRIX data bucket")
        typer.echo("")
        typer.echo("Example implementation:")
        typer.echo(f"""
from kedro_datasets.spark import SparkDataset
from matrix.datasets.huggingface import HuggingFaceDataset

# Load from GCS
nodes_sds = SparkDataset("{nodes_path}")
edges_sds = SparkDataset("{edges_path}")

nodes_df = nodes_sds.load().limit({limit}).toPandas()
edges_df = edges_sds.load().limit({limit}).toPandas()

# Upload to HuggingFace
hfds = HuggingFaceDataset(
    repo_id="{repo_id}",
    token="{token or '${HF_TOKEN}'}",
    private={private}
)

data_dict = {{"nodes": nodes_df, "edges": edges_df}}
hfds.save(data_dict)
""")
        
    except Exception as e:
        typer.echo(f"❌ Error: {e}")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()