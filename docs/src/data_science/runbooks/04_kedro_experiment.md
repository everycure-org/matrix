---
title: Running using kedro experiment
---

# How to run an experiment using kedro

We use kedro as our data pipeline framework. For a kedro intro please see [kedro onboarding](https://docs.dev.everycure.org/onboarding/kedro/#pipeline-framework-kedro).



### How to run an experiment

Running an experiment is done by running the `kedro experiment run` command.


!!! info
    This command replaces `kedro submit`. All the existing paramaters are the same, but there is now a new `--experiment-name` parameter.

Example:
```bash
kedro experiment run --pipeline=embeddings --release-version v0.2.7 --environment=cloud --experiment-name=af-test-gcs-5 --run-name=run-1
```

When you run this:

* If an `--experiment-name` is provided, it will use this. Otherwise, it will attempt to use the current branch name, without the `experiment/` prefix. 
* Next, it will look up this experiment name in MLFlow. If it doesn't exist, it will create a new experiment with this name.
* If you have not defined a `--run-name`, you will be prompted to enter one.
* Then, it will invoke `kedro submit` and submit the pipeline.
* All runs are logged in MLFlow under the experiment name provided.



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


### Common errors

```
ERROR: No such file or directory: 'conf/local//oauth_client_secret.txt'  
```

Make sure to run `make fetch_secrets` first.