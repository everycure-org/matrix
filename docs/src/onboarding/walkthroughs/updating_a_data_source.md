# Walkthrough: Updating a Data Source

## Task

Sometimes we need to update the version of a data source. This walkthrough shows updating RTX 2.7.3 to 2.10.0.

## Steps

### 1. Checking the raw data

Identify the new version of the data source in BigQuery.
It should follow this pattern:

```
mtrx-us-central1-hub-dev-storage/data/01_RAW/KGs/rtx_kg2/{version}
```

For example:

```
mtrx-us-central1-hub-dev-storage/data/01_RAW/KGs/rtx_kg2/v2.10.0
```


Check the nodes and edges tables are there in the correct format

e.g.

```
nodes_c.tsv
edges_c.tsv
curie_to_pmids.sqlite
```

You can check what the catalog expects here:

[conf/base/ingestion/catalog.yml](https://github.com/everycure-org/matrix/blob/main/pipelines/matrix/conf/base/ingestion/catalog.yml)

RTX nodes:
```
ingestion.raw.rtx_kg2.nodes@spark:
  <<: *_layer_raw
  type: matrix.datasets.gcp.LazySparkDataset
  filepath: ${globals:paths.kg_raw}/KGs/rtx_kg2/${globals:data_sources.rtx_kg2.version}/nodes_c.tsv
  file_format: csv
  load_args:
    sep: "\t"
    header: true
```

This would be the corresponding catalog entry for ROBOKOP:
```
ingestion.raw.robokop.nodes@spark:
  <<: *_layer_raw
  type: matrix.datasets.gcp.LazySparkDataset
  filepath: ${globals:paths.kg_raw}/KGs/robokop-kg/${globals:data_sources.robokop.version}/nodes.orig.tsv
  file_format: csv
  load_args:
    sep: "\t"
    header: true
    index: false
```

Update the files (preferably) or catalog entry as appropriate. Make sure changes are backwards compatible.

### 2. Update the parameters

Update the parameters in the `globals.yml` file.

[conf/base/globals.yml](https://github.com/everycure-org/matrix/blob/main/pipelines/matrix/conf/base/globals.yml)


```
data_sources:
  rtx_kg2:
    version: &_rtx_kg_version v2.7.3
  robokop:
    version: c5ec1f282158182f
```

Update to the desired version.


### 3. Run the pipeline

Save and commit the changes. Push your branch to the remote repository.

Run the pipeline to update the data source.

```
 kedro experiment run -p kg_release -e cloud --release-version=af_test_release_2.10.0 --username=amy --experiment-name rtx-2-10-0
```

kedro will prompt you to:

* confirm your MLFlow experiment name and create an experiment if it doesn't exist
* add a run name
* confirm to submit the run



### 4. Following the run

Open the Argo CD link to follow the run.



