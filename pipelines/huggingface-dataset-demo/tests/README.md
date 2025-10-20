# Test Suite for Hugging Face Dataset Demo

This directory contains tests for the custom HFIterableDataset implementation that enables reading
and writing to Hugging Face Hub across Spark, Pandas, and Polars formats.

## Test Structure

### Unit Tests (`tests/unit/`)

- **`test_hf_dataset.py`**: Tests the custom HFIterableDataset implementation

## Test Design Philosophy

### Given-When-Then Structure

All tests follow the Given-When-Then pattern for clarity:

```python
def test_example_function():
    """
    GIVEN a specific input condition
    WHEN the function is called
    THEN it should produce the expected output
    """
    # TODO: Test implementation goes here
    pass
```

### Test Coverage Areas

1. **Configuration**: HFIterableDatasetConfig validation and parameter handling
2. **Token Resolution**: Environment variables, credentials, and explicit tokens
3. **Data Loading**: Reading from Hugging Face Hub in Spark/Pandas/Polars formats
4. **Data Saving**: Writing to Hugging Face Hub in Spark/Pandas/Polars formats
5. **Error Handling**: Graceful failure modes and missing dependencies
6. **Data Consistency**: Round-trip integrity across all formats
7. **Fallback Mechanisms**: Arrow fallback for older Spark versions

## Running Tests

### All Tests

```bash
pytest
```

### With Coverage

```bash
pytest --cov=huggingface_dataset_demo.hf_kedro.datasets.hf_iterable_dataset --cov-report=html
```

### Specific Test Categories

```bash
# Unit tests only
pytest tests/unit/

# Tests marked as slow
pytest -m "not slow"
```

## Test Data

Tests use synthetic knowledge graph edge data with:

- **Subjects**: `EC-DRUG-{id}` (drug entities)
- **Predicates**: `biolink:*` (relationship types)
- **Objects**: `EC-DISEASE-{id}` (disease entities)
- **Descriptions**: `Edge {id}` (human-readable descriptions)

## Mocking Strategy

- **External APIs**: Hugging Face Hub operations are mocked for CI
- **Spark Sessions**: Real Spark sessions used in integration tests
- **File I/O**: Temporary directories for test data isolation

## Continuous Integration

Tests are designed to run in CI environments:

- Unit tests require no external dependencies
- All external services are mocked
- All tests are deterministic and isolated
