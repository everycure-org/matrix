---
title: Plugging into Cloud Environment Locally
--- 

## Plugging into cloud outputs

As you might have noticed, local setup is mainly focusing on the local execution of the pipeline with the fabricated data. However, we also run our full pipeline on production data in the `cloud` environment. Our pipeline is orchestrated using Argo Workflows, and may take several hours to complete. Given our current test process, that solely executes end-to-end tests on synthetic data, it is possible that the pipeline runs into an error due to a node handling data edge cases unsuccesfully.

Troubleshooting such issues is tedious, and validating a fix requires the entire pipeline to be re-executed. This is where the `--from-env` flag comes in.

The figure below visualizes the same pipeline, with the environment distinguishing source/destination systems.

![](../../assets/img/from-env-pipeline.drawio.svg)

Now imagine that a node in the pipeline fails. Debugging is hard, due to the remote execution environment. The `--from-env` flag allows for executing a step in the pipeline, in a given environment, while consuming the _input_ datasets from another environment.

![](../../assets/img/from-env-run.drawio.svg)

In order to run against the `cloud` environment it's important to set the `RUN_NAME` variable in the `.env` file, as this determines the run for which datasets are pulled. You can find the `RUN_NAME` on the labels of the Argo Workflow.

The following command can be used to re-run the node locally, while consuming data from `cloud`:

```bash
# NOTE: You can specify multiple nodes to rerun, using a comma-separated list
kedro run --from-env cloud --nodes preprocessing_node_name
```

!!! note 

    !!! warning
        Make sure to disable port forwarding once you're done and remove the environment variables from the .env file, otherwise this might result in you pushing local runs to our cloud MLFlow instance.
    
    If you wish to pull data from MLFlow it's currently required to setup [port-forwarding](https://emmer.dev/blog/port-forwarding-to-kubernetes/) into the MLFlow tracking container on the cluster. You can do this as follows:

    ```bash
    kubectl port-forward -n mlflow svc/mlflow-tracking 5002:80
    ```

    Next, add the following entry to your `.env` file.

    ```dotenv
    MLFLOW_ENDPOINT=http://127.0.0.1:5002
    ```

[Running jupyter within kedro :material-skip-next:](./kedro_jupyter.md){ .md-button .md-button--primary }