# Running the Matrix Pipeline

Now that you understand the different environments, have a good understanding of kedro, and have successfully run the pipeline end-to-end with test data, let's explore how to run specific parts of the pipeline in different environments.

## Pipeline Structure

As we explained at the beginning of this section, the Matrix pipeline is composed of several sub-pipelines that can be run independently or combined. The main components are:

1. **Data Engineering Pipeline** (`data_engineering`): Handles data ingestion and integration
2. **Feature Pipeline** (`feature`): Performs filtering and generates embeddings
3. **Modelling Pipeline** (`modelling_run`): Includes model training, matrix generation, evaluation and transformations

In this section we will break them down in test environment to teach you how to run the specific part of the pipeline.


## Running Individual Pipeline Components

As explained earlier in the section, data engineering pipeline is responsible for ingesting and integrating data from various biomedical knowledge graphs and datasets. 
The data engineering pipeline consists of two main stages:

**Ingestion Stage** (`ingestion`):
```bash
kedro run --pipeline=ingestion # also can do kedro run -p ingestion
```
This stage:

1. Loads raw data from each source
1. Performs initial data validation
1. Converts data into a standardized format
1. Stores intermediate results in Parquet format for efficient processing

**Integration Stage** (`integration`):
```bash
kedro run --pipeline=integration
```
This stage:

1. Normalizes data from different sources to a common format (Biolink)
1. Resolves entity synonyms to ensure consistent node IDs
1. Unions and deduplicates nodes and edges
1. Produces a unified knowledge graph

The pipeline is highly configurable, allowing you to enable or disable different data sources based on your needs. These sources are configured in `settings.py`:

```python 
DYNAMIC_PIPELINES_MAPPING = lambda: disable_private_datasets(
    generate_dynamic_pipeline_mapping(
        {
            "cross_validation": {
                "n_cross_val_folds": 3,
            },
            "integration": [
                {"name": "rtx_kg2", "integrate_in_kg": True, "is_private": False},
                {"name": "robokop", "integrate_in_kg": True, "is_private": False},
                # ... other sources
                {"name": "drug_list", "integrate_in_kg": False, "has_edges": False},
                {"name": "disease_list", "integrate_in_kg": False, "has_edges": False},
            ],
```

In the Matrix project, we use [dynamic pipelines](https://getindata.com/blog/kedro-dynamic-pipelines/) to handle multiple data sources, model variants, and evaluation types. The `DYNAMIC_PIPELINES_MAPPING` in `settings.py` defines which data sources to include, which models to train, and which evaluations to run. This allows us to easily enable/disable different components without modifying the core pipeline code. Note that the dynamic pipelines are not natively supported by kedro and it is our own custom implementation.

Therefore, the settings above reflect the Knowledge Graphs (eg RTX-KG2, SPOKE, ROBOKOP, and Embiology) which get ingested as well as other core datasets which are essential for Matrix (Drug List, Disease List, Ground Truth, Clinical Trials and Medical Team data). Thanks to `settings.py` and dynamic pipelines, they are not statically defined/hard-coded but configured through settings

Now let's get back to running the pipeline. Try running the following command first:
```bash
kedro run -p data_engineering
```

You should see all different datasets present in the `integration` part of settings being processed. It should run to completion fairly quickly:
```
Total pipeline duration: 0:01:20.391892
```


Now uncomment all of them except for RTX-KG2, like this:
```python 
DYNAMIC_PIPELINES_MAPPING = lambda: disable_private_datasets(
    generate_dynamic_pipeline_mapping(
        {
            "cross_validation": {
                "n_cross_val_folds": 3,
            },
            "integration": [
                {"name": "rtx_kg2", "integrate_in_kg": True, "is_private": False},
                #{"name": "robokop", "integrate_in_kg": True, "is_private": False},
                # ... other sources
                #{"name": "drug_list", "integrate_in_kg": False, "has_edges": False},
                #{"name": "disease_list", "integrate_in_kg": False, "has_edges": False},
            ],
```
Notice that the pipeline completes much faster, with only one KG being processed.

```
Total pipeline duration: 0:00:24.715172
```

### Running Specific Nodes

By controlling `settings.py` we can decide which datasets should be processed by only executing specific nodes. As we found out in the kedro introductory video, nodes are the building blocks of the pipeline which can also be executed on their own for debugging or to test individual components. 

For example, to run just the ingestion of drug list:

```bash
kedro run --pipeline=ingestion --nodes=write_drug_list
```

Or to run the node deduplication step in the integration pipeline:

```bash
kedro run --pipeline=integration --nodes=create_prm_unified_nodes
```
You can see how these nodes fit together with other parts of the pipeline by checking `integration/pipeline.py` and `integration/nodes.py`.

## Modifying parameters in the feature pipeline

The exact details on integration and ingestion pipeline can be found in [the pipeline section](../../pipeline/index.md). In short however, running the following command:
```
kedro run -p data_engineering -e test
```
Will create a unified knowledge graph. Such KG can then be used by the feature pipeline for generating embeddings and the modeling pipeline for training ML models. The `feature `pipeline consists of:

- `filtering`: Applies custom filters to the knowledge graph
- `embeddings`: Generates topological embeddings for nodes

Both components have configurable parameters that can significantly impact the pipeline's behavior and runtime - we will explore them in more detail to give you a better understanding of `config` directory role.

#### Filtering Parameters

The filtering pipeline allows you to control which data sources and relationships are included in the knowledge graph. You can modify these parameters in `conf/base/filtering/parameters.yml`:

```yaml
filtering:
  node_filters:
    filter_sources:
      _object: matrix.pipelines.filtering.filters.KeepRowsContainingFilter
      column: upstream_data_source
      keep_list:
        - rtxkg2
        - ec_medical
        # - robokop  # Uncomment to include ROBOKOP data
  # ...
  edge_filters:
    filter_sources:
      _object: matrix.pipelines.filtering.filters.KeepRowsContainingFilter
      column: upstream_data_source
      keep_list:
        - rtxkg2
        - ec_medical
        # - robokop

```

!!! warning "Important: Dependency Injection"
    You might have noticed that `_object` parameter occurs quite frequently across different environment and config files. This is a critical concept in our codebase - we leverage the [dependency injection](https://www.geeksforgeeks.org/dependency-injectiondi-design-pattern/) pattern to ensure clean configuration and re-usability. This design pattern is fundamental to how we structure our code and configuration. We will dig into the details of this custom kedro extension in the [kedro extension section](../deep_dive/kedro_extensions.md).


For example, to include ROBOKOP data in your pipeline:

1. Uncomment the `robokop` line in the `keep_list`
2. Run the filtering pipeline: 
```bash
kedro run --pipeline=filtering -e test
```

This will ensure that nodes and edges derived from ROBOKOP only are kept for the feature generation. You can try running the pipeline with and without robokop parameter and see how it affects the runtime of the pipeline

#### Embeddings Parameters

The embeddings pipeline generates topological embeddings for nodes in the knowledge graph. The key parameters are defined in `conf/base/embeddings/parameters.yml`:

```yaml
embeddings.topological_estimator:
  _object: matrix.pipelines.embeddings.graph_algorithms.GDSNode2Vec 
  concurrency: 4
  embedding_dim: 512
  random_seed: 42
  iterations: 10
  walk_length: 30
  walks_per_node: 10
  window_size: 10
```

Do note however that those are the parameters for the `base` environment. Topological Embedding generation is a time-consuming process therefore we have an alternative set of parameters in the `test` environment to allow for quick iterations & testing. You will see that parameters in `conf/test/embeddings/parameters.yaml` are much simpler:

```yaml
embeddings.topological_estimator:
  iterations: 1
  embedding_dim: 3
  walk_length: 2
```

This is a good example showing the difference between test and base environment. If you try increasing the number of iterations in the `conf/base/embeddings/parameters.yaml` and run:
```bash
kedro run -p embeddings -e test
```
The number of iterations will not change, as we are running the pipeline in test environment. However if you do the same in the `conf/test/embeddings/parameters.yaml` - the runtime will extend accordingly. The same rule applies to all parameters across our codebase.

## Finding Data Products while running the modelling pipeline

Once we generated embeddings, we have everything we might need for running the modelling pipeline and generating the MATRIX. This is as simple as running the following command:

```bash
kedro run --pipeline=modelling_run
```

This pipeline includes:

- `modelling`: Trains the ML models
- `matrix_generation`: Produces the drug-disease prediction matrix
- `evaluation`: Evaluates model performance
- `matrix_transformations`: Applies post-processing transformations

Just like we learnt earlier, we can run individual components (e.g. `kedro run -p modelling -e test`) of the pipeline or modify parameters (e.g. `conf/base/modelling/defaults.yml` to modify train-test splits). In the modelling pipeline we also have a choice of selecting different algorithms for modelling - these can be selected in settings.py, just like we learnt in the first section: 

```python
"modelling": {
                "model_name": "xg_ensemble",  # model_name suggestions: xg_baseline, xg_ensemble, rf, xg_synth
                "model_config": {"num_shards": 3},
            },
```

The pipeline runs multiple types of evaluations which are also configured in `settings.py`:

```python
"evaluation": [
    {"evaluation_name": "simple_classification"},      # Basic classification metrics
    {"evaluation_name": "disease_specific"},          # Disease-specific ranking
    {"evaluation_name": "full_matrix"},               # Full matrix ranking
    {"evaluation_name": "full_matrix_negatives"},     # Negative pairs evaluation
    {"evaluation_name": "simple_classification_trials"}, # Clinical trials evaluation
    {"evaluation_name": "disease_specific_trials"},   # Disease-specific trials
    {"evaluation_name": "full_matrix_trials"},        # Full matrix trials
    {"evaluation_name": "disease_specific_off_label"}, # Off-label evaluation
    {"evaluation_name": "full_matrix_off_label"}      # Full matrix off-label
]
```

Each evaluation type produces different metrics, you can learn about them in detail [here](../../pipeline/data_science/evaluation_deep_dive.md).

So once we run the pipeline:
```
kedro run -p modelling_run -e test
```
How do we find the results? There are many data products being generated in the matrix pipeline however you can find appropriate product based on pipeline data catalog. For instance, if you want to find the generated matrix, you can go to `conf/base/matrix_generation/catalog.yml` where once you follow the data layer convention, you can see the output:

```bash
"matrix_generation.fold_{fold}.model_output.sorted_matrix_predictions@spark":
  <<: *_spark_parquet
  filepath: ${globals:paths.matrix_generation}/model_output/fold_{fold}/matrix_predictions
```
The output contains several variables, such as `{folds}` for cross-validation folds or `globals:path.matrix_generation` from `globals.yml` which specify the global variable for matrix generation directory. Similar variables can be found within `conf/base/evaluation/catalog.yml` where many pathways have `source` variables - these correspond to the evaluation_metrics variables from `settings.py` - you can comment some of them out to see how your pipeline matrix run changes

## Running the pipeline with a (subset of) real data

Now that you understand how to run different parts of the pipeline, let's try to run the pipeline with _real_ data. The real KG is large and requires plenty of compute/time however we can run the pipeline with a subset of real data if you follow next steps.

```
TODO add instructions on runnign the sampled pipeline e2e with public data thats in GCS:
```

After approximately 20-30 mins, the pipeline should have finished all stages. If that's the case - well done! You can now repeat the entire process with real data if you would like however note that it will take a very long time - without parallelization, you can expect it to run for +24hrs for KGs such as RTX-KG2. Smaller Graphs might be easier.

!!! info
    Remember that the pipeline is modular by design, allowing you to run and test components independently. It's very rare that we run the pipeline with real data e2e; we usually first run data_engineering pipeline to examine the generated KG, then we extract features and only after that's complete, we would start modelling.


[Go to Deep Dive Section :material-skip-next:](../deep_dive/index.md){ .md-button .md-button--primary }
