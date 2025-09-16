# Running an experiment using previous data outputs

This walkthrough explains how to “branch” a run: reuse outputs from a previous run as inputs for a new run. This is useful when you want to experiment starting from an existing set of datasets produced by another run, without recomputing everything from scratch.

![](../../../assets/img/branched_runs.drawio.svg)

### kedro `--from-run` flag

- This flag tells kedro to read specified inputs from a different run’s datasets while writing results to the current `run-name` default outputs.
- When used alone (no datasets explicitly listed), all pipeline **input datasets** are read from the specified run’s catalog.
- When combined with `--from-run-datasets` (see below), only the listed datasets are read from the specified run’s catalog; all other inputs follow the normal catalog.

Example (use all inputs from a previous run):

```bash
kedro run --env test -p matrix_transformations --from-run my-previous-run
```

### How it works

- This follows the same concept as the `--from-env` flag, which allows users to run the pipeline locally while pulling data from a different environment (e.g. `cloud`).
- The `--from-env` flag introduced the concept of `from_catalog`, which constructs a secondary `DataCatalog` representing the source. Applicable catalog entries for pipeline inputs are overridden in the current run’s catalog to read from that source catalog.

![](../../../assets/img/from_catalog.drawio.svg)

<!-- TODO: Add some specific from_catalog examples! -->

## Running locally (`kedro run`)

- **All inputs from another run**: Just supply `--from-run`. The session will discover the current pipeline’s input datasets and read them from the `from_catalog`.
- **Selected inputs from another run**: Add `--from-run-datasets` with a comma-separated list.

## Running on the cluster (`kedro experiment run`)

Each Argo task executes an individual Kedro node. It doesn’t have full context of what ran previously in the same workflow or the source run.
Therefore we need an additional step to pre-compute the exact datasets that should be overwritten in the catalog before setting off the workflow.
Otherwise, a downstream node would try to take all inputs from the `--from-run`, even if some of its inputs had been generated in the current run.

### Calculating the correct inputs

- **Pipeline inputs discovery**: During workflow template generation, the system precomputes the pipeline’s external input datasets (excluding params). These are passed into the workflow as `from_run_datasets`.
- **Parameter propagation**: The Argo template includes parameters `from_run` and `from_run_datasets`, and each node invocation runs Kedro with:
  - `--from-run "{{workflow.parameters.from_run}}"`
  - `--from-run-datasets "{{workflow.parameters.from_run_datasets}}"`

### `--from-run-datasets`

- If `--from-run` is set but `from_run_datasets` is empty, all inputs are overridden to read from the from-catalog (mirrors local default behavior).
- If both `--from-run` and non-empty `from_run_datasets` are set, only those datasets are overridden. This is necessary on cluster runs where each node needs explicit, precomputed input lists.

### Running on `kedro experiment run`

```bash
kedro experiment run \
    --pipeline=matrix_transformations \
    --release-version v0.10.0 \
    --environment=cloud \
    --experiment-name=cool-experiment \
    --run-name=my-new-run \
    --from-run=previous-run
```

Note that you do not have to define `--from-run-datasets` yourself. `kedro experiment run` logic will figure this out for you.

You can test out locally and define this in `kedro run` though. It is possible to run this locally too: Example (use only selected inputs from a previous run):

```bash
kedro run -p my_pipeline -e dev \
  --from-run previous-run-2025-09-01 \
  --from-run-datasets "feature_table_raw,model_inputs_v1"
```
