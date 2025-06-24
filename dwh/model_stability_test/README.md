# Model Stability Test Experiment

## Overview
This experiment demonstrates how to measure the stability of model outputs across multiple runs using BigQuery. The goal is to provide a robust, scalable, and transparent way to quantify how much model predictions change from run to run, and to encourage teams to expose their model outputs in a standardized, analytics-friendly format.

## Objective
We want to understand and quantify the stability of model scores (e.g., drug-disease associations) across repeated runs of a model. This is crucial for:
- Detecting unintended changes in model behavior
- Ensuring reproducibility
- Building trust in model outputs

By storing all model runs in a partitioned and clustered table, we can efficiently compute stability metrics (such as correlation) between any pair of runs.

## Experimental Design
- **Runs 1–3:** Completely random scores. Stability between these runs should be very low (near zero correlation).
- **Runs 4–6:** Identical, deterministic scores for each drug-disease pair. Stability between these runs should be perfect (correlation = 1).
- **Runs 7–10:** Increasingly highly correlated scores, with small run-dependent perturbations. Stability between these runs should be very high (correlation close to 1, but not exactly 1).

This setup allows us to validate that our stability metric behaves as expected in different scenarios.

## Results

```bash
python dwh/model_stability_test/nb.py run-all
Running query 1
Running query 2
Running query 3
Running query 4
    run_id_1 run_id_2  stability
0   run_01   run_02    -0.038318
1   run_02   run_03    -0.006610
2   run_03   run_04     0.002053
3   run_04   run_05     1.000000
4   run_05   run_06     1.000000
5   run_06   run_07     0.828487
6   run_07   run_08     0.762112
7   run_08   run_09     0.892925
8   run_09   run_10     0.964798
```

As expected, the stability between runs 1 and 2 is very low, and the stability between runs 4 and 5 is perfect. Run 6-10 are increasingly correlated, with a stability closing in to 1.

## Why This Matters
If all model runs are stored in a single, partitioned, and clustered table, stability metrics become trivial to compute. This enables:
- Fast, scalable analytics on model behavior
- Easy integration with monitoring and alerting systems
- Consistent, organization-wide standards for model evaluation

## Table Schema
The core table for this experiment is `model_stability_test.scores`, which is both partitioned and clustered for performance and cost efficiency ([BigQuery docs](https://cloud.google.com/bigquery/docs/clustered-tables)).

### SQL Schema
```sql
CREATE OR REPLACE TABLE model_stability_test.scores (
  source STRING OPTIONS(description="Drug ID (maps to model_stability_test.drugs.id)"),
  target STRING OPTIONS(description="Disease ID (maps to model_stability_test.diseases.id)"),
  score FLOAT64 OPTIONS(description="Model score (e.g., association strength)"),
  run_id STRING OPTIONS(description="Identifier for the data generation run (e.g., 'run_01')"),
  model_id STRING OPTIONS(description="Model name, e.g., 'matrix'"),
  model_version STRING OPTIONS(description="Model version in semver, e.g., '0.1.3'"),
  rank INT64 OPTIONS(description="Rank of the score across all runs (1 = highest score)"),
  run_number INT64 OPTIONS(description="Run number (1-10)")
)
PARTITION BY RANGE_BUCKET(rank, GENERATE_ARRAY(1, 250000001, 50000))
CLUSTER BY run_id, target
OPTIONS(
  description="Scores table joining drugs and diseases, partitioned by rank and clustered by target (disease_id)"
);
```

### Pydantic Model
```python
from pydantic import BaseModel
from typing import Literal

class ScoreRecord(BaseModel):
    source: str  # Drug ID
    target: str  # Disease ID
    score: float  # Model score
    run_id: str  # e.g., 'run_01'
    model_id: Literal['matrix']  # Model name
    model_version: str  # e.g., '0.1.3'
    rank: int  # Rank of the score across all runs
    run_number: int  # Run number (1-10)
```

## Partitioning and Clustering in BigQuery
- **Partitioning** (by `rank`): Organizes data into buckets, making queries that filter on rank much faster and cheaper.
- **Clustering** (by `run_id`, `target`): Sorts data within partitions, further improving query performance, especially for queries filtering by run or disease. See [BigQuery clustering docs](https://cloud.google.com/bigquery/docs/clustered-tables) and [partitioning and clustering recommendations](https://cloud.google.com/blog/products/data-analytics/new-bigquery-partitioning-and-clustering-recommendations).

## How to Use This Metric
With this schema, you can compute stability metrics (e.g., Pearson correlation) between any two runs with a simple SQL query:

```sql
SELECT CORR(s1.score, s2.score) AS stability
FROM model_stability_test.scores s1
JOIN model_stability_test.scores s2
  ON s1.source = s2.source AND s1.target = s2.target
WHERE s1.run_id = 'run_01' AND s2.run_id = 'run_02';
```

This can be extended to a full stability matrix across all runs.

## Call to Action
**Modeling teams:**
- Expose your model outputs in a partitioned and clustered table with the schema above.
- This will make stability analysis, monitoring, and reproducibility checks trivial and scalable for all downstream consumers.

## Files in This Directory
- `01_create_schema_and_tables.sql`: Sets up the schema and tables
- `02_insert_synthetic_scores.sql`: Populates the table with synthetic data for the experiment
- `03_create_stability_metric_function.sql`: (Optional) Example of a reusable stability metric function
- `04_run_stability_metrics.sql`: Computes stability between all consecutive runs
- `nb.py`: Python notebook/script to orchestrate the experiment
