# PCA Uploader

**Staff Engineer Implementation** - Clean, robust PCA coordinates uploader for Neo4j with checkpoint
support.

## Overview

A modern, production-ready application that efficiently uploads PCA coordinates from your Matrix
pipeline to Neo4j nodes with automatic checkpoint/resume functionality.

## Key Features

🚀 **Kedro-First Design**

- Native catalog integration (`catalog.load()`)
- Automatic Spark DataFrame handling
- Zero file path management

  ⚡ **Robust & Efficient**

  - Checkpoint/resume on interruption
  - Configurable batching with parallel processing
  - Connection pooling and error recovery
  - Joblib caching for expensive data loading operations

🎯 **Clean Architecture**

- Separation of concerns with dataclasses
- Modern Click CLI interface
- Comprehensive logging and progress tracking

🔒 **Production Ready**

- Environment-based configuration
- Graceful error handling
- `.env` file support

## Installation

```bash
# Install dependencies
uv sync

# Or install in development mode
uv sync --dev
```

## Configuration

### Environment Variables

Create a `.env` file in your working directory:

```bash
# Copy the example file
cp env.example .env

# Edit .env with your actual values
NEO4J_HOST=bolt://your-neo4j-host:7687
NEO4J_USER=your-username
NEO4J_PASSWORD=your-password
NEO4J_DATABASE=analytics
```

### Default Values

- Host: `bolt://127.0.0.1:7687`
- User: `neo4j`
- Password: `admin`
- Database: `analytics`

## Usage

### CLI Interface (Recommended)

**Important**: Run from the matrix project root directory (where `pipelines/matrix/` exists)

```bash
# Basic upload with automatic Kedro integration (uses cloud environment by default)
python apps/pca-uploader/upload_pca_to_neo4j.py upload

# Specify different environment
python apps/pca-uploader/upload_pca_to_neo4j.py upload --env local

# Custom configuration
python apps/pca-uploader/upload_pca_to_neo4j.py upload \
  --dataset-name embeddings.reporting.topological_pca \
  --batch-size 5000 \
  --max-workers 16 \
  --project-path pipelines/matrix \
  --env cloud

# Check upload status and progress
python apps/pca-uploader/upload_pca_to_neo4j.py status

# Clean checkpoint files
python apps/pca-uploader/upload_pca_to_neo4j.py clean

# Force fresh start (ignore checkpoints)
python apps/pca-uploader/upload_pca_to_neo4j.py upload --no-resume
```

### Programmatic Usage (from \_workbench.py)

```python
from upload_pca_to_neo4j import upload_pca_coordinates_from_catalog

# Simple upload with defaults
upload_pca_coordinates_from_catalog(catalog)

# Upload with custom settings
upload_pca_coordinates_from_catalog(
    catalog,
    dataset_name='embeddings.reporting.topological_pca',
    batch_size=5000,
    max_workers=16
)
```

## Checkpoint/Resume System

The uploader automatically saves progress and can resume from interruptions:

- **Automatic checkpoints**: Saved every 10 batches
- **Resume on restart**: Automatically detects and resumes interrupted uploads
- **Progress tracking**: Shows exact completion status
- **Error recovery**: Handles network failures and Neo4j connection issues

```bash
# If upload is interrupted, simply restart
python upload_pca_to_neo4j.py upload  # Automatically resumes

# Check current progress
python upload_pca_to_neo4j.py status
```

## Architecture

### Clean Separation of Concerns

```python
@dataclass
class UploadConfig:
    """Configuration management"""

@dataclass
class Checkpoint:
    """Checkpoint persistence with save/load/delete"""

class PCAPipeline:
    """Main pipeline with context management"""
    - load_pca_data()       # Kedro catalog integration
    - create_batches()      # Efficient batching
    - update_batch()        # Neo4j operations
    - upload_with_checkpoints() # Progress tracking
```

### Error Handling Strategy

- **Connection resilience**: Automatic retry and connection pooling
- **Batch-level errors**: Individual batch failures don't stop the process
- **Checkpoint integrity**: Atomic checkpoint updates
- **Graceful degradation**: Detailed error logging with context

## Performance Optimization

### Caching

The application uses joblib to cache expensive data loading operations. The cache is stored in the
`.cache/` directory and will automatically reuse previously loaded PCA data, significantly speeding
up subsequent runs.

### Recommended Settings

```bash
# For large datasets (>1M records)
python upload_pca_to_neo4j.py upload --batch-size 5000 --max-workers 16

# For smaller datasets or limited resources
python upload_pca_to_neo4j.py upload --batch-size 1000 --max-workers 4
```

### Tuning Guidelines

- **Batch size**: 2000-5000 for optimal Neo4j performance
- **Workers**: Scale with CPU cores and memory (4-16 typical)
- **Memory**: ~2GB for 1M records with batch size 2000

## File Structure

- `upload_pca_to_neo4j.py` - Main application with clean architecture
- `example_kedro_usage.py` - Usage examples and patterns
- `pyproject.toml` - Project configuration with Click dependency
- `env.example` - Environment variables template
- `README.md` - This documentation

## Development

```bash
# Install with development tools
uv sync --dev

# Run linting
uv run ruff check .

# Format code
uv run ruff format .

# Run tests
uv run pytest
```

## Migration from Old Version

The new implementation provides backward compatibility:

```python
# Old API still works
from upload_pca_to_neo4j import upload_pca_coordinates_from_catalog
upload_pca_coordinates_from_catalog(catalog)

# New CLI is recommended for new usage
# python upload_pca_to_neo4j.py upload
```

## Troubleshooting

### Common Issues

**Connection failures:**

```bash
python upload_pca_to_neo4j.py upload  # Will test connection first
```

**Interrupted uploads:**

```bash
python upload_pca_to_neo4j.py status  # Check progress
python upload_pca_to_neo4j.py upload  # Resume automatically
```

**Memory issues:**

```bash
python upload_pca_to_neo4j.py upload --batch-size 1000 --max-workers 2
```

**Fresh start needed:**

```bash
python upload_pca_to_neo4j.py clean   # Clear checkpoints
python upload_pca_to_neo4j.py upload --no-resume
```
