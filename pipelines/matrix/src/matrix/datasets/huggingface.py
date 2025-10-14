# NOTE: This file was partially generated using AI assistance.

import logging
from typing import Any, Dict, Optional, Union
from pathlib import Path
import tempfile
import shutil

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import Dataset, DatasetDict, load_dataset
from huggingface_hub import HfApi, create_repo, upload_folder
from kedro.io.core import (
    AbstractDataset,
    DatasetError,
    Version,
)

logger = logging.getLogger(__name__)


class HuggingFaceDataset(AbstractDataset[Union[pd.DataFrame, Dict[str, pd.DataFrame]], Union[pd.DataFrame, Dict[str, pd.DataFrame]]]):
    """Kedro dataset for uploading and downloading data to/from Hugging Face Hub.
    
    This dataset enables seamless integration between Kedro pipelines and Hugging Face Hub
    for data distribution. It supports both single DataFrames and multiple DataFrames
    organized as a dictionary (for datasets with multiple splits/tables).
    
    Example usage in catalog.yml:
    ```yaml
    my_hf_dataset:
      type: matrix.datasets.huggingface.HuggingFaceDataset
      repo_id: "everycure/matrix-kg-v0.10.0"
      token: ${oc.env:HF_TOKEN}
      private: false
      save_args:
        commit_message: "Upload KG data v0.10.0"
        commit_description: "Knowledge graph nodes and edges for drug repurposing"
      load_args:
        split: null  # Load all splits as DatasetDict
    ```
    
    For single DataFrame:
    ```python
    from kedro_datasets.spark import SparkDataset
    from matrix.datasets.huggingface import HuggingFaceDataset
    
    # Load from GCS
    sds = SparkDataset("gs://bucket/path/to/data.parquet")
    df = sds.load().toPandas()
    
    # Save to HuggingFace
    hfds = HuggingFaceDataset(
        repo_id="everycure/my-dataset",
        token="hf_...",
        private=False
    )
    hfds.save(df)
    ```
    
    For multiple DataFrames (e.g., nodes and edges):
    ```python
    # Save multiple tables
    data_dict = {
        "nodes": nodes_df,
        "edges": edges_df
    }
    hfds.save(data_dict)
    
    # Load back
    loaded_dict = hfds.load()  # Returns {"nodes": DataFrame, "edges": DataFrame}
    ```
    """

    def __init__(
        self,
        *,
        repo_id: str,
        token: Optional[str] = None,
        private: bool = False,
        save_args: Optional[Dict[str, Any]] = None,
        load_args: Optional[Dict[str, Any]] = None,
        version: Optional[Version] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize the HuggingFaceDataset.
        
        Args:
            repo_id: Repository ID on Hugging Face Hub (e.g., "username/dataset-name")
            token: Hugging Face API token. If None, will try to use cached token
            private: Whether the repository should be private
            save_args: Additional arguments for saving:
                - commit_message: Commit message for the upload
                - commit_description: Longer description for the commit
                - create_pr: Whether to create a pull request instead of direct commit
                - revision: Branch/tag to commit to (default: "main")
            load_args: Additional arguments for loading:
                - split: Specific split to load (e.g., "train"). If None, loads all splits
                - revision: Branch/tag to load from (default: "main")
                - streaming: Whether to use streaming mode
            version: Kedro version (not used, as HF handles versioning via git)
            metadata: Additional metadata
        """
        self._repo_id = repo_id
        self._token = token
        self._private = private
        self._save_args = save_args or {}
        self._load_args = load_args or {}
        self._version = version
        self._metadata = metadata or {}
        
        # Initialize HF API
        self._api = HfApi(token=self._token)
        
    def _describe(self) -> Dict[str, Any]:
        """Return dataset description for Kedro."""
        return {
            "repo_id": self._repo_id,
            "private": self._private,
            "save_args": self._save_args,
            "load_args": self._load_args,
        }
        
    def _ensure_repo_exists(self) -> None:
        """Ensure the repository exists on Hugging Face Hub."""
        try:
            self._api.repo_info(self._repo_id, repo_type="dataset")
            logger.info(f"Repository {self._repo_id} already exists")
        except Exception:
            logger.info(f"Creating new dataset repository: {self._repo_id}")
            create_repo(
                repo_id=self._repo_id,
                repo_type="dataset",
                private=self._private,
                token=self._token,
            )
            
    def _save_single_dataframe(self, data: pd.DataFrame, temp_dir: Path) -> None:
        """Save a single DataFrame as parquet file."""
        file_path = temp_dir / "data.parquet"
        
        # Convert to pyarrow table for efficient parquet writing
        table = pa.Table.from_pandas(data)
        pq.write_table(table, file_path)
        
        logger.info(f"Saved DataFrame with {len(data)} rows to {file_path}")
        
    def _save_multiple_dataframes(self, data: Dict[str, pd.DataFrame], temp_dir: Path) -> None:
        """Save multiple DataFrames as separate parquet files."""
        for split_name, df in data.items():
            file_path = temp_dir / f"{split_name}.parquet"
            
            # Convert to pyarrow table for efficient parquet writing
            table = pa.Table.from_pandas(df)
            pq.write_table(table, file_path)
            
            logger.info(f"Saved {split_name} DataFrame with {len(df)} rows to {file_path}")
            
    def _create_dataset_card(self, temp_dir: Path, data: Union[pd.DataFrame, Dict[str, pd.DataFrame]]) -> None:
        """Create a README.md file with dataset information."""
        readme_path = temp_dir / "README.md"
        
        if isinstance(data, pd.DataFrame):
            # Single DataFrame
            rows = len(data)
            cols = len(data.columns)
            splits_info = f"- **data.parquet**: {rows:,} rows, {cols} columns"
            
        else:
            # Multiple DataFrames
            splits_info = []
            for split_name, df in data.items():
                rows = len(df)
                cols = len(df.columns)
                splits_info.append(f"- **{split_name}.parquet**: {rows:,} rows, {cols} columns")
            splits_info = "\n".join(splits_info)
        
        readme_content = f"""---
license: mit
task_categories:
- other
tags:
- knowledge-graph
- drug-repurposing
- matrix
- everycure
pretty_name: MATRIX Knowledge Graph Data
size_categories:
- 1M<n<10M
---

# MATRIX Knowledge Graph Dataset

This dataset contains knowledge graph data from the MATRIX drug repurposing platform by Every Cure.

## Dataset Structure

{splits_info}

## Usage

```python
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("{self._repo_id}")

# Access data
data = dataset["train"]  # or specific split name
df = data.to_pandas()
```

## Citation

If you use this dataset in your research, please cite:

```bibtex
@misc{{matrix_kg_dataset,
  title={{MATRIX Knowledge Graph Dataset}},
  author={{Every Cure}},
  year={{2024}},
  url={{https://huggingface.co/datasets/{self._repo_id}}}
}}
```

## License

MIT License - see LICENSE file for details.

## More Information

- [MATRIX Documentation](http://docs.dev.everycure.org)
- [Every Cure](https://everycure.org)
- [GitHub Repository](https://github.com/everycure-org/matrix)
"""
        
        with open(readme_path, "w") as f:
            f.write(readme_content)
            
        logger.info(f"Created dataset card at {readme_path}")

    def save(self, data: Union[pd.DataFrame, Dict[str, pd.DataFrame]]) -> None:
        """Save data to Hugging Face Hub.
        
        Args:
            data: Either a single pandas DataFrame or a dictionary of DataFrames
                 where keys are split names (e.g., {"nodes": df1, "edges": df2})
        """
        if not isinstance(data, (pd.DataFrame, dict)):
            raise DatasetError(
                f"HuggingFaceDataset can only save pandas DataFrames or "
                f"dictionaries of DataFrames, got {type(data)}"
            )
            
        if isinstance(data, dict):
            for key, value in data.items():
                if not isinstance(value, pd.DataFrame):
                    raise DatasetError(
                        f"All values in data dictionary must be pandas DataFrames, "
                        f"got {type(value)} for key '{key}'"
                    )
        
        # Ensure repository exists
        self._ensure_repo_exists()
        
        # Create temporary directory for files
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Save data files
            if isinstance(data, pd.DataFrame):
                self._save_single_dataframe(data, temp_path)
            else:
                self._save_multiple_dataframes(data, temp_path)
                
            # Create dataset card
            self._create_dataset_card(temp_path, data)
            
            # Upload to Hugging Face Hub
            commit_message = self._save_args.get("commit_message", "Upload dataset")
            commit_description = self._save_args.get("commit_description", None)
            create_pr = self._save_args.get("create_pr", False)
            revision = self._save_args.get("revision", "main")
            
            logger.info(f"Uploading dataset to {self._repo_id}...")
            
            commit_info = upload_folder(
                folder_path=temp_dir,
                repo_id=self._repo_id,
                repo_type="dataset",
                commit_message=commit_message,
                commit_description=commit_description,
                create_pr=create_pr,
                revision=revision,
                token=self._token,
            )
            
            logger.info(f"Successfully uploaded dataset. Commit: {commit_info.oid}")
            
    def load(self) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """Load data from Hugging Face Hub.
        
        Returns:
            Either a single pandas DataFrame (if dataset has one split/file) or
            a dictionary of DataFrames (if dataset has multiple splits)
        """
        try:
            split = self._load_args.get("split", None)
            revision = self._load_args.get("revision", "main")
            streaming = self._load_args.get("streaming", False)
            
            logger.info(f"Loading dataset from {self._repo_id} (revision: {revision})")
            
            # Load dataset from Hugging Face Hub
            dataset = load_dataset(
                self._repo_id,
                split=split,
                revision=revision,
                streaming=streaming,
                token=self._token,
            )
            
            if isinstance(dataset, Dataset):
                # Single split - return as DataFrame
                df = dataset.to_pandas()
                logger.info(f"Loaded single DataFrame with {len(df)} rows")
                return df
                
            elif isinstance(dataset, DatasetDict):
                # Multiple splits - return as dictionary of DataFrames
                result = {}
                for split_name, split_dataset in dataset.items():
                    df = split_dataset.to_pandas()
                    result[split_name] = df
                    logger.info(f"Loaded {split_name} DataFrame with {len(df)} rows")
                return result
                
            else:
                raise DatasetError(f"Unexpected dataset type: {type(dataset)}")
                
        except Exception as e:
            raise DatasetError(f"Failed to load dataset from {self._repo_id}: {str(e)}") from e

    def exists(self) -> bool:
        """Check if the dataset exists on Hugging Face Hub."""
        try:
            self._api.repo_info(self._repo_id, repo_type="dataset")
            return True
        except Exception:
            return False