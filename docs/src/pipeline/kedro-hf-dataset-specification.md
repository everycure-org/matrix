# Kedro HuggingFace Dataset Specification

## Overview

This specification defines a custom Kedro dataset implementation that integrates with HuggingFace Hub for data distribution and consumption, with support for the Xet file system format. This enables Every Cure to publish and distribute knowledge graph releases through HuggingFace Hub while maintaining compatibility with the existing Kedro pipeline infrastructure.

## Background

As part of the open sourcing initiative (XDATA-248), Every Cure needs to establish a data distribution mechanism for public consumption of knowledge graph releases. HuggingFace Hub provides an ideal platform for this purpose, offering:

- Version-controlled dataset hosting
- Easy discovery and access for researchers
- Integration with the broader ML ecosystem
- Support for large files through Git LFS and Xet
- Community features and documentation

## Requirements

### Functional Requirements

1. **Data Loading**: Load datasets from HuggingFace Hub repositories into Kedro pipelines
2. **Data Saving**: Publish datasets to HuggingFace Hub repositories from Kedro pipelines
3. **Xet Support**: Handle large files using Xet file system format for efficient storage and transfer
4. **Authentication**: Support HuggingFace Hub authentication tokens
5. **Versioning**: Support dataset versioning through HuggingFace Hub's revision system
6. **Metadata**: Handle dataset metadata, descriptions, and tags
7. **Multiple Formats**: Support common data formats (Parquet, CSV, JSON, etc.)

### Non-Functional Requirements

1. **Performance**: Efficient handling of large knowledge graph datasets
2. **Reliability**: Robust error handling and retry mechanisms
3. **Security**: Secure handling of authentication credentials
4. **Compatibility**: Full integration with existing Kedro infrastructure
5. **Documentation**: Comprehensive documentation and examples

## Architecture

### Core Components

```
HuggingFaceDataset
├── HuggingFaceParquetDataset
├── HuggingFaceCSVDataset
├── HuggingFaceJSONDataset
└── HuggingFaceXetDataset
```

### Class Hierarchy

```python
# Base class for all HuggingFace datasets
class HuggingFaceBaseDataset(AbstractDataset):
    """Base dataset for HuggingFace Hub integration"""

# Specific format implementations
class HuggingFaceParquetDataset(HuggingFaceBaseDataset):
    """Parquet format dataset for HuggingFace Hub"""

class HuggingFaceXetDataset(HuggingFaceBaseDataset):
    """Xet format dataset for large files on HuggingFace Hub"""
```

## Implementation Details

### Base Dataset Configuration

```yaml
# Example catalog.yml configuration
knowledge_graph_nodes:
  type: matrix.datasets.huggingface.HuggingFaceParquetDataset
  repo_id: "everycure/matrix-kg-v1.0"
  filename: "nodes.parquet"
  revision: "main"  # or specific commit/tag
  credentials: huggingface_token
  load_args:
    columns: ["id", "name", "category"]
  save_args:
    commit_message: "Update knowledge graph nodes"
    commit_description: "Updated nodes with latest data processing"
    private: false
  metadata:
    description: "Knowledge graph nodes for drug repurposing"
    tags: ["knowledge-graph", "drug-repurposing", "biomedical"]
```

### Xet Integration

```yaml
# Large file handling with Xet
large_embeddings:
  type: matrix.datasets.huggingface.HuggingFaceXetDataset
  repo_id: "everycure/matrix-embeddings-v1.0"
  filename: "node_embeddings.xet"
  revision: "main"
  credentials: huggingface_token
  xet_args:
    chunk_size: "64MB"
    compression: "lz4"
  load_args:
    lazy: true  # Enable lazy loading for large files
```

### Authentication

```yaml
# credentials.yml
huggingface_token: "${oc.env:HF_TOKEN}"

# Alternative: file-based token
huggingface_token:
  token_file: "~/.cache/huggingface/token"
```

## API Design

### Core Methods

```python
class HuggingFaceBaseDataset:
    def __init__(
        self,
        repo_id: str,
        filename: str,
        revision: str = "main",
        credentials: Optional[Dict[str, Any]] = None,
        load_args: Optional[Dict[str, Any]] = None,
        save_args: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Initialize HuggingFace dataset.
        
        Args:
            repo_id: HuggingFace repository ID (e.g., "everycure/matrix-kg")
            filename: File name within the repository
            revision: Git revision (branch, tag, or commit hash)
            credentials: Authentication credentials
            load_args: Arguments for loading data
            save_args: Arguments for saving data
            metadata: Dataset metadata for repository
        """

    def load(self) -> Any:
        """Load data from HuggingFace Hub."""

    def save(self, data: Any) -> None:
        """Save data to HuggingFace Hub."""

    def exists(self) -> bool:
        """Check if dataset exists in HuggingFace Hub."""

    def _get_file_path(self) -> str:
        """Get local cached file path."""

    def _authenticate(self) -> str:
        """Handle HuggingFace authentication."""

    def _upload_file(self, local_path: str) -> None:
        """Upload file to HuggingFace Hub."""

    def _download_file(self) -> str:
        """Download file from HuggingFace Hub."""
```

### Xet-Specific Methods

```python
class HuggingFaceXetDataset(HuggingFaceBaseDataset):
    def _configure_xet(self) -> None:
        """Configure Xet file system settings."""

    def _upload_xet_file(self, local_path: str) -> None:
        """Upload large file using Xet format."""

    def _download_xet_file(self) -> str:
        """Download large file from Xet storage."""
```

## Usage Examples

### Loading Data

```python
# In a Kedro node
def process_knowledge_graph(kg_nodes: pd.DataFrame) -> pd.DataFrame:
    """Process knowledge graph nodes loaded from HuggingFace Hub."""
    # Data is automatically loaded from HuggingFace Hub
    return kg_nodes.drop_duplicates()
```

### Saving Data

```python
# In a Kedro pipeline
def create_processed_nodes() -> pd.DataFrame:
    """Create processed nodes to be saved to HuggingFace Hub."""
    # Process data
    processed_nodes = ...
    return processed_nodes  # Automatically saved to HuggingFace Hub
```

### Programmatic Access

```python
from matrix.datasets.huggingface import HuggingFaceParquetDataset

# Direct usage
dataset = HuggingFaceParquetDataset(
    repo_id="everycure/matrix-kg-v1.0",
    filename="nodes.parquet",
    credentials={"token": "hf_token"}
)

data = dataset.load()
dataset.save(processed_data)
```

## Integration Points

### Kedro Catalog Integration

The datasets will be registered in the Kedro catalog system and can be used like any other Kedro dataset:

```python
# In pipeline definition
from kedro import pipeline, node

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=process_nodes,
            inputs="hf_knowledge_graph_nodes",  # From HuggingFace
            outputs="processed_nodes",
            name="process_kg_nodes"
        ),
        node(
            func=generate_embeddings,
            inputs="processed_nodes",
            outputs="hf_node_embeddings",  # To HuggingFace
            name="generate_embeddings"
        )
    ])
```

### Data Release Pipeline Integration

```python
# Integration with existing data release hooks
def publish_to_huggingface(context: KedroContext, pipeline_name: str):
    """Hook to automatically publish datasets to HuggingFace after pipeline completion."""
    catalog = context.catalog
    
    # Publish specific datasets
    datasets_to_publish = [
        "knowledge_graph_nodes",
        "knowledge_graph_edges", 
        "node_embeddings"
    ]
    
    for dataset_name in datasets_to_publish:
        if dataset_name in catalog._datasets:
            dataset = catalog._datasets[dataset_name]
            if isinstance(dataset, HuggingFaceBaseDataset):
                # Dataset will be automatically uploaded on save
                pass
```

## Error Handling

### Common Error Scenarios

1. **Authentication Failures**: Invalid or expired HuggingFace tokens
2. **Network Issues**: Connection timeouts, rate limiting
3. **Repository Access**: Private repositories, insufficient permissions
4. **File Not Found**: Missing files in repository
5. **Large File Handling**: Xet configuration issues

### Error Handling Strategy

```python
class HuggingFaceDatasetError(Exception):
    """Base exception for HuggingFace dataset operations."""

class AuthenticationError(HuggingFaceDatasetError):
    """Raised when authentication fails."""

class RepositoryNotFoundError(HuggingFaceDatasetError):
    """Raised when repository is not accessible."""

class FileNotFoundError(HuggingFaceDatasetError):
    """Raised when file is not found in repository."""

# Retry mechanism with exponential backoff
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((ConnectionError, TimeoutError))
)
def _download_with_retry(self, url: str) -> bytes:
    """Download with retry logic."""
```

## Testing Strategy

### Unit Tests

```python
class TestHuggingFaceDataset:
    def test_load_existing_dataset(self):
        """Test loading an existing dataset from HuggingFace Hub."""
    
    def test_save_new_dataset(self):
        """Test saving a new dataset to HuggingFace Hub."""
    
    def test_authentication_handling(self):
        """Test various authentication scenarios."""
    
    def test_xet_large_file_handling(self):
        """Test Xet integration for large files."""

    def test_error_scenarios(self):
        """Test error handling and recovery."""
```

### Integration Tests

```python
class TestHuggingFaceIntegration:
    def test_kedro_pipeline_integration(self):
        """Test full integration with Kedro pipeline."""
    
    def test_data_release_workflow(self):
        """Test complete data release workflow."""
```

## Performance Considerations

### Optimization Strategies

1. **Caching**: Leverage HuggingFace Hub's built-in caching mechanisms
2. **Lazy Loading**: Support lazy loading for large datasets
3. **Parallel Downloads**: Use concurrent downloads for multiple files
4. **Compression**: Utilize Xet compression for large files
5. **Incremental Updates**: Support incremental dataset updates

### Benchmarking

- Measure download/upload speeds for various file sizes
- Compare performance with/without Xet for large files
- Monitor memory usage during large file operations

## Security Considerations

### Token Management

1. **Environment Variables**: Store tokens in environment variables
2. **File-based Storage**: Support secure token file storage
3. **Credential Rotation**: Handle token expiration and rotation
4. **Scope Limitations**: Use minimal required permissions

### Data Privacy

1. **Private Repositories**: Support for private dataset repositories
2. **Access Control**: Respect HuggingFace Hub access controls
3. **Audit Logging**: Log data access and modifications

## Migration Path

### From Existing Systems

1. **GCS Integration**: Maintain compatibility with existing GCS datasets
2. **Local Files**: Support migration from local file storage
3. **Gradual Adoption**: Enable incremental migration of datasets

### Backward Compatibility

- Maintain existing dataset interfaces
- Provide migration utilities
- Support hybrid configurations

## Future Enhancements

### Planned Features

1. **Dataset Streaming**: Support for streaming large datasets
2. **Delta Updates**: Incremental dataset updates
3. **Multi-format Support**: Additional file format support
4. **Advanced Metadata**: Rich dataset metadata and lineage
5. **Community Features**: Integration with HuggingFace community features

### Research Areas

1. **Performance Optimization**: Advanced caching and prefetching
2. **Data Lineage**: Integration with Kedro's data lineage tracking
3. **Automated Publishing**: CI/CD integration for automated dataset publishing

## Dependencies

### Required Packages

```python
# Core dependencies
huggingface_hub >= 0.20.0
datasets >= 2.16.0
kedro >= 0.19.0
kedro-datasets >= 6.0.0

# Xet integration
xet-core >= 0.16.0  # Hypothetical Xet Python package

# Optional dependencies
pandas >= 2.1.0  # For DataFrame support
pyarrow >= 14.0.0  # For Parquet support
```

### System Requirements

- Python 3.11+
- Git (for HuggingFace Hub operations)
- Sufficient disk space for dataset caching
- Network connectivity to HuggingFace Hub

## Documentation Requirements

### User Documentation

1. **Getting Started Guide**: Quick setup and basic usage
2. **Configuration Reference**: Complete configuration options
3. **Best Practices**: Recommendations for optimal usage
4. **Troubleshooting Guide**: Common issues and solutions

### Developer Documentation

1. **API Reference**: Complete API documentation
2. **Architecture Overview**: System design and components
3. **Contributing Guide**: Guidelines for contributors
4. **Testing Guide**: How to run and write tests

## Conclusion

This specification provides a comprehensive foundation for implementing a Kedro-HuggingFace dataset integration with Xet support. The proposed solution will enable Every Cure to efficiently distribute knowledge graph data through HuggingFace Hub while maintaining seamless integration with existing Kedro pipelines.

The modular design allows for incremental implementation and future enhancements, while the robust error handling and security considerations ensure production-ready reliability.

## Next Steps

1. **Prototype Development**: Create initial implementation of core functionality
2. **Testing**: Develop comprehensive test suite
3. **Documentation**: Create user and developer documentation
4. **Integration**: Integrate with existing MATRIX pipeline infrastructure
5. **Deployment**: Deploy to production environment with monitoring

---

*This specification document was partially generated using AI assistance.*