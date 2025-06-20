<!-- TODO PBR complet -->
## Environments

We have 4 environments declared in the kedro project for `MATRIX`:

- `base`: Contains the base environment which reads the real data from GCS and operates in your local compute environment
- `cloud`: Contains the cloud environment with real data. All data is read and written to a GCP project as configured (see below). Assumes fully stateless local machine operations (e.g. in docker containers)
- `test`: Fully local and contains parameters that "break" the meaning of algorithms in the pipeline (parameters we use don't make sense; e.g. we use 2 dimensions PCA instead of 100). This is useful for running an integration test with mock data to validate the programming of the pipeline is correct to a large degree. 
- `local`: A default environment which you can use for local adjustments and tweaks. Changes to this repo are not usually committed to git as they are unique for every developer. 
- `sample`: Contains a sample of the data and is useful for fast iterations on the pipeline from the embeddings pipeline and on.

!!! info
    Remember the `.env.default` and `.env` mentioned in the [repository structure](./repo_structure.md)? Our `cloud` environment is equipped with environment variables that allow for controlling your credentials (e.g. github token) or configuring the GCP project to use ([more about this in deep dive](../deep_dive/gcp_setup.md)) 

You can run any of the environments using the `--env` flag. For example, to run the pipeline in the `cloud` environment, you will  use the following command:

```bash
kedro run --env cloud # NOTE: this is just an example; do not run it
```
!!! Google Cloud Platform
    Note that our cloud environment both reads and writes all intermediate data products to our Google Cloud Storage. In general, it should be only used for pipeline runs which are being executed on our kubernetes cluster. 
 
### Run with fake data locally

To run the full pipeline locally with fake data, you can use the following command:

```bash
kedro run --env test -p test 
```

This runs the full pipeline with fake data. This is exactly what we did as a part of `make integration_test` in the previous section but now we are not using make wrapper.

### Run with real data locally

To run the full pipeline with real data by copying the RAW data from the central GCS bucket and then run everything locally you can simply run from the default environment. We've setup an intermediate pipeline that copies data to avoid constant copying of the data from cloud.

```bash
# Copy data from cloud to local
kedro run -p ingestion
```

Hereafter, you can run the default pipeline.

```bash
# Default pipeline in default environment
kedro run -p data_engineering
```
Note that running the pipeline with a real data locally is very time- and compute-intensive so for development purposes we recommend only using sampled data or fabricated data.

### Run with sample data locally

To run the the pipeline from the embeddings step onwards with a smaller dataset for testing or development purposes, use the sample environment:

```bash
# Run pipeline with sample data
kedro run -e sample -p test_sample
```
For more details on sampling environment, see the [Sample Environment Guide](../deep_dive/sample_environment.md).

!!! info
    Environments are abstracted away by Kedro's data catalog which is, in turn, defined as configuration in YAML. The catalog is dynamic, in the sense that it can combine the `base` environment with another environment during execution. This allows for overriding some of the configuration in `base` such that data can flow into different systems according to the selected _environment_. 

    The image below represents a pipeline configuration across three environments, `base`, `cloud` and `test`. By default the pipeline reads from Google Cloud Storage (GCS) and writes to the local filesystem. The `cloud` environment redefines the output dataset to write to `BigQuery` (as opposed to local). The `test` environment redefines the input dataset to read the output from the fabricator pipeline, thereby having the effect that the pipeline runs on synthetic data.

![](../../assets/img/environments.drawio.svg)

Now that you have a good understanding of different environments, we can run the pipeline with a sample of real data.

!!! info
    If you want to learn more about each environment, there is a dedicated session for [test](../deep_dive/test_environment.md), [sample](../deep_dive/sample_environment.md), [base](../deep_dive/sample_environment.md) and [cloud](../deep_dive/cloud_environment.md) environments in the deep dive section.


[Running the pipeline :material-skip-next:](./run_pipeline.md){ .md-button .md-button--primary }