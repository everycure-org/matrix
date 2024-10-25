---
title: Pipeline
---


# Pipeline

## Overview

Roughly speaking, our pipeline consists of five logical stages, i.e., ingestion, integration, embedding, modelling, and evaluation.

![](../assets/img/status.drawio.svg)

### Preprocessing

The current pre-processing pipeline is highly preliminary and is used to ingest experimental nodes and edges proposed by our medical team. The pipeline is integrated with a Google
sheet for rapid hypothesis testing.

### Ingestion

The ingestion pipeline aims to ingest all the downstream data in BigQuery, our data warehouse of choice. Data from different sources is assigned metadata for lineage tracking.

We've established a lightweight data versioning system to ensure we can easily revert to an older version of the input data if required. All of our data should be stored in Google Cloud Storage (GCS) under the following path:

```
gs://<bucket>/kedro/data/01_raw/<source>/<version>/<source_specific_files>
```

Next, our pipeline globals provide an explicit listing of the versions that should be used during pipeline run, for instance:

```yaml
# globals.yml
versions:
  sources:
    rtx-kg2: v2.7.3
    another-kg: v.1.3.5
    ... # Other data source versions here
```

Finally, catalog entries should be defined to ensure the correct linkage of the catalog entry to the version.

```yaml
# catalog.yml
integration.raw.rtx_kg2.edges:
  filepath: ${globals:paths.raw}/rtx_kg2/${globals:data_sources.rtx-kg2.version}/edges.tsv
  ... # Remaining configuration here
```

Note specifically the use of `globals:data_sources.rtx-kg2` in the definition of the catalog entry. Whenever new data becomes available, code changes are limited to bumping the `versions.sources.<source>` entry in the globals.

!!! info
    To date our pipeline only ingests data from the RTX-KG2 source.

### Integration

The integration stage aims to produce our internal knowledge-graph, in [biolink](https://biolink.github.io/biolink-model/) format. As we ingest data from different sources, entity resolution becomes a prevalent topic. The integration step consolidates entities across sources to avoid data duplication in the knowledge graph.

There are 3 main steps in the integration pipeline:

![](../assets/img/kg_integration_approach.excalidraw.svg)

### Embeddings

Embeddings are vectorized representations of the entities in our knowledge graph. These are currently computed in two stages:

1. Node Attribute Embedding Computation - in this step we use GenAI model (e.g. OpenAI's `text-embedding-3-small`) to compute individual node embeddings. 
2. Topological Embedding Computation - in this step we use GraphSAGE embedding algorithm on the previously calculated node embeddings. Alternatively, you can also use Node2Vec for topological embeddings computation - the model is not as well supported in Neo4J however it does not rely on Node Attribute Embedding Computation.

!!! info
    Our graph database, i.e., [Neo4J](https://neo4j.com/docs/graph-data-science/current/algorithms/) comes with out-of-the-box functionality to compute both node and topological embeddings in-situ. The Kedro pipeline orchestrates the computation of these.


### Modelling 

The modelling pipeline trains drug repurposing prediction models using knowledge graph embeddings generated in the embeddings pipeline. 

The main steps are as follows:
1. *Prepare ground truth dataset*. Load ground truth positive and negative drug-disease pairs. Perform test-train split. 
2. *Synthesise additional training data*. Synthesise additional drug-disease pairs for training using an appropriate sampling strategy.  
3. *Perform hyperparameter tuning*. Optimise model hyperparameters to maximise performance according to a chosen objective function.  
4. *Train final model*.  Train a drug repurposing prediction model with hyperparameters found in the previous step. 
5. *Check model performance*. Computes classification metrics using the test portion of the ground truth data. 

As well as single models, the pipeline has the capability to deal with *ensembles* of models trained with resampled synthesised training data.  

!!! info
    The step *check model performance* only gives a partial indication of model performance intended as a quick sanity check. This is because, in general, the ground truth data alone is not a good reflection of the data distribution that the model will see while performing its task. The evaluation pipeline must be run before making conclusions about model performance. 

### Matrix Generation 

The matrix generation step pipeline uses the models trained in the modelling pipeline to generate score for all pairs in the matrix, that is pairs of drugs and diseases from the official EC lists. 

Flags for known positives and negatives are also generated in this pipeline. In addition, we remove any training data from the matrix as they may have artificially high scores. 


### Evaluation

The evaluation pipeline computes various metrics in order to assess the performance of the models trained in the previous stages. 

The input to the evaluation pipeline is a dataset of pairs, complete with scores, from the matrix generation pipeline. By using the matrix generation pipeline as an intermediary, we can avoid repeating the computationally expensive steps of model inference and instead focus on computing metrics.  

There are several types of metrics computed in this pipeline, organised into three categories:

1. **Full-matrix ranking metrics**: These metrics focus on how well the model ranks the set of pairs comprising the matrix.
2. **Disease-specific ranking metrics**: These metrics focus on how well the model ranks drugs for a specific disease.
3. **Ground truth classification metrics**: These metrics focus on the model's ability to distinguish between known positive and known negative drug-disease pairs.

More details on the metrics computed in each category can be found in the [evaluation deep-dive](../data_science/evaluation_deep_dive.md)



### Inference (requests)

Our inference pipeline can be used for running ad-hoc requests coming from medical team/stakeholders to generate drug-disease predictions for a specifid drug, disease or a pair of both. The drugs/diseases to predict against are coming from our official drug and disease lists (which are versioned in the .env file). The pipeline is running inference using a single/several trained models stored as artifacts in MLFlow and utilizes the same version of data that was used for training. 

![](../assets/img/inference.drawio.png)

You can find the sheet [here](https://docs.google.com/spreadsheets/d/1CioSCCQxUdACn1NfWU9XRyC-9j_ERc2hmZzaDd8XgcQ/edit?gid=0#gid=0). At the moment we don't execute this as a part of the default pipeline. Also note that in order to use the trained models which are stored in the MLFlow (i.e. models trained using e2e pipeline) you will need to execute the inference pipeline from `cloud` environment.

### Release

Our release pipeline currently builds the final integrated Neo4J data product for consumption. We do not execute this as part of the default pipeline run but with a separate `-p release` execution as we do not want to release every pipeline data output.

!!! info
    If you wish to populate your local Neo4J instance with the output data for a release, populate the `RUN_NAME` in your `.env` file and run `kedro run -p release -e cloud`.

## Environments

We have 4 environments declared in the kedro project for `MATRIX`:

- `base`: Contains the base environment which reads the real data from GCS and operates in your local compute environment
- `cloud`: Contains the cloud environment with real data. All data is read and written to a GCP project as configured (see below). Assumes fully stateless local machine operations (e.g. in docker containers)
- `test`: Fully local and contains parameters that "break" the meaning of algorithms in the pipeline (e.g. 2 dimensions PCA). This is useful for running an integration test with mock data to validate the programming of the pipeline is correct to a large degree. 
- `local`: A default environment which you can use for local adjustments and tweaks. Changes to this repo are not usually committed to git as they are unique for every developer. 

!!! info
    Our `cloud` environment is equipped with environment variables that allows for configuring the GCP project to use. This is especially relevant to switch between the `hub` and `wg` projects as desired.

    The source code contains a `.env.tmpl` configuration template file. To configure the `cloud` environment, create your own `.env` file from the template and uncomment variables relevant to your configuration. 

You can run any of the environments using the `--env` flag. For example, to run the pipeline in the `cloud` environment, you can use the following command:

```bash
kedro run --env cloud
```

!!! info
    Environments are abstracted away by Kedro's data catalog which is, in turn, defined as configuration in YAML. The catalog is dynamic, in the sense that it can combine the `base` environment with another environment during execution. This allows for overriding some of the configuration in `base` such that data can flow into different systems according to the selected _environment_. 

    The image below represents a pipeline configuration across three environments, `base`, `cloud` and `test`. By default the pipeline reads from Google Cloud Storage (GCS) and writes to the local filesystem. The `cloud` environment redefines the output dataset to write to `BigQuery` (as opposed to local). The `test` environment redefines the input dataset to read the output from the fabricator pipeline, thereby having the effect that the pipeline runs on synthetic data.

![](../assets/img/environments.drawio.svg)

### Run with fake data locally

To run the full pipeline locally with fake data, you can use the following command:

```bash
kedro run --env test -p test 
```

This runs the full pipeline with fake data.

### Run with real data locally

To run the full pipeline with real data by copying the RAW data from the central GCS bucket and then run everything locally you can simply run from the default environment. We've setup an intermediate pipeline that copies data to avoid constant copying of the data from cloud.

```bash
# Copy data from cloud to local
kedro run -p ingestion
```

Hereafter, you can run the default pipeline.

```bash
# Default pipeline in default environment
kedro run
```

## Pipeline with Jupyter notebooks

Kedro may be used with [Jupyter notebooks](https://docs.kedro.org/en/stable/notebooks_and_ipython/kedro_and_notebooks.html) for interactive experiments. This allows us to utilise data and models generated by pipeline runs in Jupyter notebooks, as well as to take advantage of the functions and classes in the Kedro project codebase. 

Jupyter notebooks should be created in the directory `pipelines/matrix/notebooks/scratch`. This will be ignored by the matrix git repository. 

!!! tip
    A separate git repository for notebook version control may be created inside the `scratch` directory. It can also be nice to create a symbolic link to `scratch` from a directory of your choice on your machine. 

    An example notebook is also added to our documentation [here](./walkthroughs/kedro_notebook_example.ipynb which you can copy into the scratch directory for a quickstart

Within a notebook, first run a cell with the following magic command:

```python
%load_ext kedro.ipython
```

By default, this loads the `base` Kedro environment which is used only with fabricated data. 
To load the `cloud` Kedro environment with real data, run another cell with the following command:
```python
%reload_kedro --env=cloud
```

These commands define several useful global variables on your behalf: `context`, `session`, `catalog` and `pipelines`.

In particular, the `catalog` variable provides an interface to the Kedro data catalog, which includes all data, models and model outputs produced during the latest `cloud` run of the Kedro pipeline. The following command lists the available items in the data catalog:
```python
catalog.list()
```
Items may be loaded into memory using the `catalog.load` method. For example, if we have a catalog item `modelling.model_input.splits`, it may be loaded in as follows: 
```python
splits = catalog.load('modelling.model_input.splits')
```

Functions and classes in the Kedro project source code may be imported as required. For example, a function `train_model` defined in the file `pipelines/matrix/src/matrix/pipelines/modelling/nodes.py` may be imported as follows:
```
from matrix.pipelines.modelling.nodes import train_model
```

Further information may be found [here](https://docs.kedro.org/en/stable/notebooks_and_ipython/kedro_and_notebooks.html). 
