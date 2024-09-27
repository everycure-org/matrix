# Walkthrough: Data Scientist Daniel

This use case follows the journey of a dataset through the Kedro workflow, from raw data to its use in model training. The dataset was added by the hypothetical Data Scientist, Daniel.

# Task

Daniel has integrated a new knowledge graph into Matrix and implemented a machine learning model for predicting drug-disease efficacy scores. We will reconstruct the steps Daniel needs to take to go from raw data to generating drug-disease predictions.

# Integration

## Integrating the New Graph into the codebase

### Raw Data

Daniel has integrated a new biomedical graph, **RTX-KG2**, which is a knowledge graph consisting of node and edge data.

The data can be found at `gs://mtrx-us-central1-hub-dev-storage/data/01_RAW/KGs/rtx_kg2/v2.10.0/.` It includes two `.tsv` files (`edges_c.tsv` and `nodes_c.tsv`) and a database file (`node_synonymizer_v1.0_KG2.10.0.sqlite`) used for node synonymization process.

Daniel manually added this data to Google Cloud Storage (GCS), as there is currently no automated ingestion system. However, multiple versions of these graphs are maintained.

### Programmatic access to data

Next, let’s look at how Kedro picks up this data.

The ingestion process can be found in the following file:

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

`node` object in Kedro is an arbitrary Python function performing computation.

- F.lit(): Stands for “literal” and is used to wrap a constant or fixed value in a way that it can be used as a column in PySpark DataFrame transformations.
- inputs: `ingestion.raw.rtx_kg2.nodes@spark` refers to an object defined in Kedro’s **Data Catalog.**
    - `...@spark` implies that format of the input data is a PySpark DataFrame.
- outputs: `ingestion.int.rtx_kg2.nodes` refers to an object defined in Kedro’s **Data Catalog.**
    - Absence of the `...@spark` suffix means that this dataset is not explicitly cast as PySpark DataFrame. However, if Data Catalog defines it as such, it will still be written as Spark dataframe.

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

⚠️ `register_pipeline()` is a special function in Kedro that defines pipelines for the entire deployment.

**Data Catalog**

Official documentation: https://docs.kedro.org/en/stable/data/data_catalog.html

In a Kedro project, the Data Catalog is a registry of all data sources available for use by the project. It is specified in a YAML catalog file, which maps the names of node inputs and outputs as keys in the `DataCatalog` class.

Below is an in-depth investigation into the Kedro Data Catalog:

[Data Catalog: In-depth](walkthrough_data_catalog.md)

**Example**

The node `write_rtx_kg2_nodes` reads the raw `ingestion.raw.rtx_kg2.nodes` dataset from Spark and writes it to `ingestion.int.rtx_kg2.nodes`, casting columns to literals. Variables are defined in `globals.yml`.

- **Read**: `gs://mtrx-us-central1-hub-dev-storage/kedro/data/01_raw/rtx_kg2/v2.7.3/nodes_c.tsv`
- **Write**: `gs://mtrx-us-central1-hub-dev-storage/runs/run-sept-first-node2vec-e5962a18/02_intermediate/rtx_kg2/nodes/nodes_c.tsv`

Importantly, the **version** is defined statically in the `.yml` file. On the other hand, `RUN_NAME` is pulled from enviornment variables via `run_name: ${oc.env:RUN_NAME}`. 

These environment variables are defined within `.env` file for local runs.

When ran in the cloud, the name can be specified as a part of argo workflow which gets submitted to the cluster.

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

### As visible in Kedro-Viz

![Screenshot 2024-09-26 at 12.02.52.png](../assets/img/kedro_daniel_pipeline.png)


### Kedro Run

**How is this pipeline executed?**

The pipeline object is passed to Kedro’s `register_pipelines()` function, where it becomes exposed to commands like `kedro run -e local -p ingestion`. This command triggers the execution of the pipeline locally, in the `local` environment, on the local machine.

However, this is not the primary method for running workloads in production.

### Summary 

Daniel has successfully integrated a new knowledge graph into the codebase, and added appropriate nodes to data catalog. The new pipeline can be now viewed in Kedro-Viz, and executed to produce node embeddings.

Now, we will look at how this pipeline can be executed.






## ArgoCD

### Questions:

- How do we set the run name for a particular run?
- What is the mechanism by which a run is submitted to our Kubernetes (K8s) cluster?

### Use Cases the Pipeline Split Refactor Must Address:

- A user wants to materialize embeddings only.
- A user wants to materialize embeddings and modeling, but the latter fails. In response, we want to re-run the modeling only, using the previous embeddings as input.
