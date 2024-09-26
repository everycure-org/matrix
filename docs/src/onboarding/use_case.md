# Kedro integration workflow: Data Scientist Daniel

This is an investigation tracing code, data and logic flow from ingestion of a new data source to generation of new embeddings.

# Task

Daniel has just integrated a new knowledge graph into Matrix; he has also implemented a new ML model for predicting drug x disease efficacy scores. We will reconstruct what Daniel needs to do to go from data, to drug x disease predictions.

# Integration

## Integrating the new graph into the codebase

### Raw Data

Daniel has integrated a new biomedical toxicity graph, **RTX-KG2**. RTX-KG2 is a knowledge graph, passed in as node and edge data.

This data can be viewed at `gs://mtrx-us-central1-hub-dev-storage/data/01_RAW/KGs/rtx_kg2/v2.10.0/` and consists of two .tsv files, `edges_c.tsv` and `nodes_c.tsv` and database `node_synonymizer_v1.0_KG2.10.0.sqlite` 

Daniel has added this data to GCS manually - there is no automated ingestion system. However, we maintain multiple versions of those graphs.

### Programmatic access to data

Let’s see how this data is picked up by Kedro.

`pipelines/matrix/src/matrix/pipelines/ingestion/pipeline.py`

```python
import pyspark.sql.functions as F

def create_pipeline(**kwargs) -> Pipeline:
    """Create ingestion pipeline."""
    return pipeline(
        [
            node(
                func=lambda x: x.withColumn("kg_source", F.lit("rtx_kg2")),
                inputs=["ingestion.raw.rtx_kg2.nodes@spark"],
                outputs="ingestion.int.rtx_kg2.nodes",
                name="write_rtx_kg2_nodes",
                tags=["rtx_kg2"],
            ),
            node(
                func=lambda x: x.withColumn("kg_source", F.lit("rtx_kg2")),
                inputs=["ingestion.raw.rtx_kg2.edges@spark"],
                outputs="ingestion.int.rtx_kg2.edges",
                name="write_rtx_kg2_edges",
                tags=["rtx_kg2"],
            ),
        ]
    )
```

`node` object in Kedro is an arbitrary Python function performing computation

- F.lit(): Stands for “literal” and is used to wrap a constant or fixed value in a way that it can be used as a column in PySpark DataFrame transformations.
- inputs: `ingestion.raw.rtx_kg2.nodes@spark` refers to an object defined in Kedro’s **Data Catalog.**
    - `...@spark` implies that format of the input data is a PySpark DataFrame.
- outputs: `ingestion.int.rtx_kg2.nodes` refers to an object defined in Kedro’s **Data Catalog.**
    - Absence is the `...@spark` tag means that this dataset is not explicitly cast as PySpark DataFrame. However, if Data Catalog defines it as such, it will still be written as Spark dataframe.

`create_pipeline()` is then being called in `pipelines/matrix/src/matrix/pipeline_registry.py`

```python
from matrix.pipelines.ingestion.pipeline import (
    create_pipeline as create_ingestion_pipeline,
)

def register_pipelines() -> Dict[str, Pipeline]:
    pipelines = {}
    pipelines["ingestion"] = create_ingestion_pipeline()
    
    # more code ...
    
    return pipelines

```

⚠️ `register_pipeline()` is special function in Kedro, which defines pipelines for the entire deployment.

**Data Catalog**

Official documentation: https://docs.kedro.org/en/stable/data/data_catalog.html

In a Kedro project, the Data Catalog is a registry of all data sources available for use by the project. It is specified with a YAML catalog file that maps the names of node inputs and outputs as keys in the `DataCatalog` class.

Below is an in-depth investigation into Kedro Data Catalog.

[Data Catalog: in-depth](Kedro%20integration%20workflow%20Data%20Scientist%20Daniel%2010cb57e013738085a3d2eb1f8ca3ac71/Data%20Catalog%20in-depth%2010db57e0137380bc8ccbc69cbfd34591.md)

**Example**

Node `write_rtx_kg2_nodes` will read raw `ingestion.raw.rtx_kg2.nodes` dataset from Spark, and will write them to `ingestion.int.rtx_kg2.nodes` , casting columns to literals. Variables are defined in `globals.yml`.

- Read: `gs://mtrx-us-central1-hub-dev-storage/kedro/data/01_raw/rtx_kg2/v2.7.3/nodes_c.tsv`
- Write: `gs:/mtrx-us-central1-hub-dev-storage/runs/run-sept-first-node2vec-e5962a18/02_intermediate/rtx_kg2/nodes/nodes_c.tsv`

Importantly, **version** is defined statically in `.yml` code. On the other hand, `RUN_NAME` is pulled from enviornment variables via `run_name: ${oc.env:RUN_NAME}`

It is still not clear to me where the value of RUN_NAME variable comes from.

```yaml

... globals.yml

data_sources:
  rtx-kg2:
    version: v2.7.3

run_name: ${oc.env:RUN_NAME}

gcs_bucket: gs://${oc.env:GCP_BUCKET}
gcp_project: ${oc.env:GCP_PROJECT_ID}

# NOTE: MLflow does not like "new options" in the mlflow.yml
# due to schema validation. Will make PR.
mlflow_artifact_root: ${gcs_bucket}/runs/${run_name}/mlflow

paths:
  # hard coded against our central data location for raw

  # TODO Need to re-work our approach between what is a "run" and what is a "release"
  # TODO(mateusz.wasilewski): Change this to use a variable
  raw: gs://mtrx-us-central1-hub-dev-storage/kedro/data/01_raw
  int: ${gcs_bucket}/runs/${run_name}/02_intermediate
  prm: ${gcs_bucket}/runs/${run_name}/03_primary
  feat: ${gcs_bucket}/runs/${run_name}/04_feature
  model_input: ${gcs_bucket}/runs/${run_name}/05_model_input
  models: ${gcs_bucket}/runs/${run_name}/06_models
  model_output: ${gcs_bucket}/runs/${run_name}/07_model_output
  reporting: ${gcs_bucket}/runs/${run_name}/08_reporting
  
...

node(
    func=lambda x: x.withColumn("kg_source", F.lit("rtx_kg2")),
    inputs=["ingestion.raw.rtx_kg2.nodes@spark"],
    outputs="ingestion.int.rtx_kg2.nodes",
    name="write_rtx_kg2_nodes",
    tags=["rtx_kg2"],
)

...

ingestion.raw.rtx_kg2.nodes@spark:
  <<: *_layer_raw
  type: matrix.datasets.gcp.SparkWithSchemaDataset
  filepath: ${globals:paths.raw}/rtx_kg2/${globals:data_sources.rtx-kg2.version}/nodes_c.tsv
  file_format: csv
  load_args:
    sep: "\t"
    header: false
    index: false
    schema:
      object: pyspark.sql.types.StructType
      fields:
        - object: pyspark.sql.types.StructField
          name: id
          dataType: 
            object: pyspark.sql.types.StringType
          nullable: False

	      ... more schema fields follow
	      
	...
	
	ingestion.int.rtx_kg2.nodes:
  <<: [*_spark_parquet, *_layer_int]
  filepath: ${globals:paths.int}/rtx_kg2/nodes

```

### Visibility in Kedro

![Screenshot 2024-09-26 at 12.02.52.png](Kedro%20integration%20workflow%20Data%20Scientist%20Daniel%2010cb57e013738085a3d2eb1f8ca3ac71/Screenshot_2024-09-26_at_12.02.52.png)

### Kedro Run

How is this pipeline executed?

Pipeline object is passed to Kedro’s `register_pipelines` ****, where is becomes exposed to commands such as `kedro run -e local -p ingestion` , which would trigger execution of the pipeline locally.

This would trigger execution of this pipeline, in `local` environment, and on local machine. But this is not the primary way we run our workloads in production.

## ArgoCD

Questions:

- How do we set run name for particular run?
- What is the mechanism via which a run is submitted to our K8s cluster?

Use cases that pipeline split refactor must cover:

- User wants to materialize embeddings only.
- User wants to materialize embeddings + modelling, but the latter fails. In response to this, we want to re-run modelling only, with old embeddings as input.
-