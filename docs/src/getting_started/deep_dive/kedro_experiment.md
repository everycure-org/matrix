---
title: Kedro Cloud run on the Cluster
---

# How to run an experiment using kedro

In the first onboarding section you were shown how to run the pipeline with a fabricated & sampled data. In this section we will show you how to run matrix pipeline on the kubernetes cluster using `kedro experiment run` command. To learn about the submissions to the cluster and argo please go to [infrastructure section](../../infrastructure/runbooks/03_run_pipeline_from_branch.md). 

Running an experiment in the Matrix Project is done by running the `kedro experiment run` command. 

!!! info
    This command replaces `kedro submit`. All the existing paramaters are the same, but there is now a new `--experiment-name` parameter. Note that `Kedro Experiment` is our custom extension of kedro

Example:
```bash
kedro experiment run --pipeline=embeddings --release-version v0.2.7 --environment=cloud --experiment-name=af-test-gcs-5 --run-name=run-1
```

When you run this:

* If an `--experiment-name` is provided, it will use this as an experiment name for MLFlow experiment. Otherwise, it will attempt to use the current branch name, without the `experiment/` prefix. 
* Next, it will look up this experiment name in MLFlow. If it doesn't exist, it will create a new experiment with this name.
* If you have not defined a `--run-name`, you will be prompted to enter one.
* Then, docker containerizes your codebase and pushes it to GCP Artifact Registry 
* Next step is to create an Argo Workflow Template which is filled with kedro pipeline specifications and computational resources (this allows many nodes to run in parallel in argo).
* Argo Workflow is then submitted to the kubernetes cluster and you will receive a link where you can follow your experiment.
* All runs are logged in MLFlow under the experiment name provided. This is where you can check your parameters and Outpus

### How to create an experiment

It is possible to create an experiment from `kedro experiment run`, as shown above. Alternatively, you can create an experiment without submitting anything using the `kedro experiment create` command.

```bash
kedro experiment create --experiment-name={EXPERIMENT_NAME}
```

* If an `experiment-name` is provided, it will use this name.
* If no `experiment-name` is provided:  
    * If the branch name starts with `experiment/` it will use the branch name exlcuding the `experiment/` prefix.
    * Otherwise it will prompt you to enter an experiment name.
* It will then attempt to create an experiment in MLFlow with this name. 
    * If no experiment with this name exists, it will create a new experiment and return the experiment id.
    * If an active experiment with this name already exists, it will error.
    * If a _deleted_ experiment with this name exists, it will prompt you to confirm that you want to rename the deleted experiment and proceed with this name. (See more below)


!!! info "Re-use or delete experiments"
    MLFlow does not allow you to re-use experiment names, even if the previous experiment has been deleted.
    [This deletion](https://mlflow.org/docs/latest/python_api/mlflow.client.html#mlflow.client.MlflowClient.delete_experiment) is a soft-delete, not a permanent deletion. Experiment names can not be reused, unless the deleted experiment is permanently deleted by a database admin.
    As a workaround, if users want to re-use an experiment name after it has been deleted, we append a `deleted-` prefix and a random suffix to the deleted experiment name.


!!! Error
    If you encounter
    ```
    ERROR: No such file or directory: 'conf/local//oauth_client_secret.txt'  
    ```
    Make sure to run `make fetch_secrets` first.

Note that running the pipeline with real data requires a lot of time. If you want to check if your pipeline will even work (e.g. let's assume you developed a new feature which works with fabricated data but you are unsure about real data) you can test it using `sample` environment as described in [first steps section](./../first_steps/environments_overview.md)

[Jupyter Setup :material-skip-next:](./kedro_jupyter.md){ .md-button .md-button--primary }