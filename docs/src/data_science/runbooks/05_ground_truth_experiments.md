# Ground Truth Experiments: How to Use Ground Truth Data in the Pipeline

This runbook explains how to leverage ground truth datasets for training of our models in our data science pipeline.

## Step-by-Step Guide

### 1. Ingest the Ground Truth Data
- Ensure the ground truth dataset you wish to use is ingested into the pipeline.
- By default, `ec_indication_list` and `kgml_xdtd_ground_truths` are ingested.
- If you need to add a new ground truth source, ensure its schema is compatible with the logic in the transformers for ground truth, such as [`ec_ground_truth.py`](../../../pipelines/matrix/src/matrix/pipelines/integration/transformers/ec_ground_truth.py) and [`kgml_xdtd_ground_truth.py`](../../../pipelines/matrix/src/matrix/pipelines/integration/transformers/kgml_xdtd_ground_truth.py). You need to make sure that your transformer is formatting the schema of the ingested GT dataset into a schema compatible with our edge schema (defined in schema.py)

#### Example: ec_indication_list Ground Truth Transformer
```python
from pyspark.sql import DataFrame
import pyspark.sql.functions as f
import pyspark.sql.types as T

class GroundTruthTransformer(Transformer):
    """Transformer for ground truth data"""

    def __init__(self, upstream_source: str = "matrix_indication_list", **kwargs):
        super().__init__(**kwargs)
        self.upstream_source = upstream_source

    def transform(self, positive_edges: DataFrame, negative_edges: DataFrame, **kwargs) -> dict[str, DataFrame]:
        pos_edges = (
            self._extract_edges(positive_edges)
            .withColumn("predicate", f.lit("indication").cast(T.StringType()))
            .withColumn("y", f.lit(1))
        )
        neg_edges = (
            self._extract_edges(negative_edges)
            .withColumn("predicate", f.lit("contraindication").cast(T.StringType()))
            .withColumn("y", f.lit(0))
        )
        edges = pos_edges.union(neg_edges).withColumn("upstream_source", f.lit(self.upstream_source))
        id_list = edges.select("subject").union(edges.select("object")).distinct().withColumnRenamed("subject", "id")
        return {"nodes": id_list, "edges": edges}

    def _extract_edges(self, edges_df: DataFrame) -> DataFrame:
        return (
            edges_df.withColumnRenamed("final normalized drug id", "subject")
            .withColumnRenamed("final normalized disease id", "object")
            .withColumnRenamed("final normalized drug label", "subject_label")
            .withColumnRenamed("final normalized disease label", "object_label")
            .withColumnRenamed("downfilled from mondo", "flag")
            .withColumnRenamed("drug|disease", "id")
            .select("subject", "object", "subject_label", "object_label", "id", "flag")
        )
```
See the full implementation in [`ec_ground_truth.py`](../../../pipelines/matrix/src/matrix/pipelines/integration/transformers/ec_ground_truth.py).

#### Example: kgml_xdtd_ground_truth Transformer
```python
from pyspark.sql import DataFrame
import pyspark.sql.functions as f
import pyspark.sql.types as T

class GroundTruthTransformer(Transformer):
    """Transformer for ground truth data"""

    def transform(self, positive_edges: DataFrame, negative_edges: DataFrame, **kwargs) -> dict[str, DataFrame]:
        pos_edges = (
            self._extract_edges(positive_edges)
            .withColumn("predicate", f.lit("indication").cast(T.StringType()))
            .withColumn("y", f.lit(1))
        )
        neg_edges = (
            self._extract_edges(negative_edges)
            .withColumn("predicate", f.lit("contraindication").cast(T.StringType()))
            .withColumn("y", f.lit(0))
        )
        edges = pos_edges.union(neg_edges).withColumn("upstream_source", f.lit("kgml_xdtd"))
        id_list = edges.select("subject").union(edges.select("object")).withColumnRenamed("subject", "id").distinct()
        return {"nodes": id_list, "edges": edges}

    def _extract_edges(self, edges_df: DataFrame) -> DataFrame:
        return (
            edges_df.withColumnRenamed("source", "subject")
            .withColumnRenamed("target", "object")
            .withColumn("id", f.concat_ws("|", f.col("subject"), f.col("object")))
            .withColumn("subject_label", f.lit(None).cast(T.StringType()))
            .withColumn("object_label", f.lit(None).cast(T.StringType()))
            .withColumn("flag", f.lit(None).cast(T.StringType()))
            .select("subject", "object", "subject_label", "object_label", "id", "flag")
        )
```
See the full implementation in [`kgml_xdtd_ground_truth.py`](../../../pipelines/matrix/src/matrix/pipelines/integration/transformers/kgml_xdtd_ground_truth.py).

### 2. Enable the Ground Truth in Settings
- Open `settings.py` and enable the new ground truth dataset by adding it to the relevant configuration section.
- Use Pydantic models to ensure settings are validated and consistent.

### 3. Run the Data Engineering Pipeline
- For official releases, run the `data_engineering` pipeline or `kg_release` pipeline as appropriate.
- This step ensures the ground truth data is processed and available for downstream tasks.
- Leverage Kedro for pipeline orchestration and reproducibility.
- Use joblib caching for any expensive preprocessing steps involving ground truth data to improve performance.

### 4. Configure the Ground Truth Set in defaults.yml
- Once the pipeline has finished, specify which ground truth set(s) to use in `defaults.yml`.
- You can specify more than one ground truth set for multi-source evaluation.
- Example:
  ```yaml
  ground_truth_sets:
    - ec_indication_list
    - kgml_xdtd_ground_truths
  ```

### 5. Experiment Tracking and Logging
- The specified ground truth set(s) will be automatically logged in MLflow for each experiment run, ensuring full traceability and reproducibility of results.
- Clearly document which ground truth sets were used in each experiment for future reference.

## Troubleshooting
- If your ground truth data is not available in the pipeline, check the ingestion step and schema compatibility.
- Ensure the ground truth set is enabled in both `settings.py` and `defaults.yml`.
- Review MLflow logs to confirm the correct ground truth set is being tracked. 