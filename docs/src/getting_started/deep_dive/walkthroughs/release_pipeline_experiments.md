# Testing a change in the release pipeline

All releases are run in an automated way. **Manual releases should never be run.**

## Task

This walkthrough shows an experiment updating RTX from version 2.7.3 to 2.10.0.

## Steps

### 1. Checking the raw data

Identify the new version of the data source in GCS. 
For public KG data sources like RTX-KG2, it should follow this pattern:

```
data.dev.everycure.org/data/01_RAW/KGs/rtx_kg2/{version}
```

For example, [RTX v2.10.0](https://console.cloud.google.com/storage/browser/data.dev.everycure.org/data/01_RAW/KGs/rtx_kg2/v2.10.0)

```
data.dev.everycure.org/data/01_RAW/KGs/rtx_kg2/v2.10.0
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
  type: matrix_gcp_datasets.gcpLazySparkDataset
  filepath: ${globals:paths.raw_public}/KGs/rtx_kg2/${globals:data_sources.rtx_kg2.version}/nodes_c.tsv
  file_format: csv
  load_args:
    sep: "\t"
    header: true
```

This would be the corresponding catalog entry for ROBOKOP:
```
ingestion.raw.robokop.nodes@spark:
  <<: *_layer_raw
  type: matrix_gcp_datasets.gcpLazySparkDataset
  filepath: ${globals:paths.raw_public}/KGs/robokop-kg/${globals:data_sources.robokop.version}/nodes.orig.tsv
  file_format: csv
  load_args:
    sep: "\t"
    header: true
    index: false
```

Make sure the files in the file path match with the catalog entry.

### 2. Make required code changes

Checkout a new branch. The branch name should reflect the name of the experiment. 

!!! info
    Do NOT checkout a branch with the desired release version (e.g. `release/v0.4.8`) as previously advised. The actual release logic will be handled by automation.

```
git checkout main
git pull origin main
git checkout -b <your-branch-name>
```

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
 kedro experiment run -p data_engineering -e cloud --release-version=experiment_rtx_2.10.0 --is-test --username=${USER} --experiment-name rtx-2-10-0
```

**Make sure to use the `--is-test` flag to run the pipeline in test mode** until we want to create an official release.

!!!info
    **Make sure that the `--release-version` is set to a test name, NOT an official release version such as v0.6.1**

kedro will prompt you to:

* confirm your MLFlow experiment name and create an experiment if it doesn't exist
* add a run name
* confirm to submit the run



### 4. Following the run

Open the Argo CD link to follow the run.


### 5. Merging the changes for the next release

* Open a PR with your desired changes and get approval from an Every Cure team member
* Merge to main
* Our automation will run according to the release cadence and your changes will be applied in the next release. Releases are scheduled for:
  ** Every Tuesday - patch version bump
  ** Every month - minor version bump

