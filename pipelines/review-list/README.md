# Review List Pipeline

A Kedro-based pipeline for combining dataframes of drug-disease pairs from multiple ranked datasets.

## Overview

The Review List Pipeline takes multiple ranked datasets (e.g., drug-disease pairs) and combines them into a single, weighted interleaved list suitable for human review. It ensures that:

- Each input dataset contributes according to its specified weight
- The final output respects the original ranking within each dataset
- Duplicate source-target pairs are avoided
- The output is limited to a specified number of rows

## Architecture

The pipeline consists of two main stages:

### 1. Prefetch Top Quota (`prefetch_top_quota`)
- Calculates the quota for each input dataset based on weights and the total limit
- Applies a 20% buffer to ensure sufficient data in case there are duplicates
- Returns trimmed DataFrames containing only the top-ranked rows from each input

### 2. Weighted Interleaving (`weighted_interleave_dataframes`)
- Performs weighted random selection of datasets for each output position
- Maintains ranking order within each dataset
- Tracks seen pairs to avoid duplicates
- Produces a final DataFrame with sequential ranks

## Input Data Requirements

Each input dataset must contain:
- `source`: The source entity (e.g., drug name)
- `target`: The target entity (e.g., disease name)  
- `rank`: Numeric ranking

## Configuration

### Parameters (`conf/base/parameters.yml`)

```yaml
inputs_to_review_list:
  input_dataframe_1:
    weight: 0.7  # 70% of the final output
  input_dataframe_2:
    weight: 0.3  # 30% of the final output

review_list_config:
  limit: 10  # Maximum number of rows in final combined result
```

**Important**: Weights must sum to 1.0

### Data Catalog (`conf/base/catalog.yml`)

Input datasets should be configured as Spark Parquet datasets.

```
input_dataframe_1:
  <<: *_spark_parquet
  filepath:  data/review_list/input_dataframe_1

input_dataframe_2:
  <<: *_spark_parquet
  filepath:  data/review_list/input_dataframe_2
```

The names should correspond to the same entries in the params file.

## Usage

### Running the Pipeline

```bash
# From the pipeline directory
kedro run

# Run specific nodes
kedro run --node=prefetch_top_quota_node
kedro run --node=weighted_interleave_dataframes_node
```


## Output

The pipeline produces:
- **Trimmed datasets**: `trimmed_{dataset_name}` - Top-ranked rows from each input with buffer
- **Combined result**: `combined_ranked_pairs_dataframe` - Final interleaved list with sequential ranks


## Development

### Setup

```bash
# Install dependencies
uv sync

# Activate virtual environment
source .venv/bin/activate  # Unix/macOS
```

### Testing

```bash
# Run specific test file
pytest tests/pipelines/test_review_list.py
```


## Troubleshooting

### Common Issues

1. **Weights don't sum to 1**: Ensure all weights in `inputs_to_review_list` sum exactly to 1.0
2. **Missing rank column**: Verify all input datasets contain a `rank` column
3. **Insufficient data**: The 20% buffer should handle most cases, but very small datasets may need adjustment
