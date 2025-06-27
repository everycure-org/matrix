
### Ingestion

The ingestion pipeline ingests all input data into the workspace of the pipeline. Data from different sources is assigned metadata for lineage tracking.

We've established a lightweight data versioning system to ensure we can easily revert to
an older version of the input data if required. All of our data should be stored in
Google Cloud Storage (GCS) under the following path:

```
gs://<bucket>/kedro/data/releases/<release_version>/datasets/...
```

And each run (anything downstream of the KG) that is based on a release stores its data in the following path:

```
gs://<bucket>/kedro/data/releases/<release_version>/runs/<run_name>/datasets/...
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
  filepath: ${globals:paths.raw}/rtx_kg2/${globals:data_sources.rtx_kg2.version}/edges.tsv
  ... # Remaining configuration here
```

Note specifically the use of `globals:data_sources.rtx-kg2` in the definition of the catalog entry. Whenever new data becomes available, code changes are limited to bumping the `versions.sources.<source>` entry in the globals.

!!! info
    To date our pipeline only ingests 3rd party data from the RTX-KG2 and ROBOKOP sources.
