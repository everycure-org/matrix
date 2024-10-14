---
title: Local setup
---

The fastest way to check if everything works locally is to execute the following command in `pipelines/matrix`

```
make
```

!!! help "Encountering issues?"
    If you're experiencing any problems running the `MakeFile`, please refer to our [Common Errors FAQ](../FAQ/common_errors.md) for troubleshooting guidance. This resource contains solutions to frequently encountered issues and may help resolve your problem quickly.


!!! tip

    If you are running on an ARM machine (e.g., MacBooks with Apple Silicon), you might not get the best performance. In this case call `make TARGET_PLATFORM=linux/arm64` instead which will build the image for your specific architecture.

This command executes a number of make targets, namely:
- set up a virtual environment
- install the python dependencies
- lint the codebase
- test the codebase
- build the docker image
- run the docker image

Generally, the `Makefile` is a good place to start to get a sense of how our codebase is structured. 

<div style="position: relative; width: 100%; height: 0; padding-bottom: 56.25%;"><iframe src="https://us06web.zoom.us/clips/embed/ghxELqxMExaMh96j_58dMq8UnXaXEcEtSTRVxXf7zXNd5l4OSgdvC5xwANUE1ydp7afd-M42UkQNe_eUJQkCrXIZ.SAPUAWjHFb-sh7Pj" frameborder="0" allowfullscreen="allowfullscreen" style="position: absolute; width: 100%; height: 100%; top: 0; left: 0; "></iframe></div>

??? note "Understanding and troubleshooting your `make` run"
   
    As mentioned in the video, our pipeline is evolving quickly. Running `make` is a simple and quick way to check if everything works locally as it is composed of several different stages which get executed one after another. When `make` gets executed, what is happening under the hood is that the following seven 'make' "subcommands" get executed:
    ```
    Make prerequisites # checks if all prerequisites are set up correctly
    Make venv # sets up a virtual environment using uv
    Make install # installs all dependencies within the virtual environment
    Make precommit # installs and runs pre-commit hooks
    Make fast_test # tests the codebase 
    Make compose_down # ensures that there is no docker-compose down running 
    Make docker_test # executes an integration test, which sets up a docker-compose and kicks off a local CI pipeline
    ```
    The Makefile is composed of many other useful commands which might also help you use the codebase with ease. One of such commands is `Make clean` which will clean various cache locations.
    
    If you are getting any errors during your default `make` run, first step could be to run `Make clean` and re-run the default Makefile to see whether this could be a cache-related error. If the error persists, a quick way to find the cause of the error is to execute the mentioned `make` subcommands one after another to locate where the error is happening. For instance: if you can run all make commands except docker_test, then it means that there is only an issue with the integration test, and remaining stages of setup work well.

    Note that the most 'complex' step is to run `Make docker_test` as this command executes the entire pipeline in a test environment, with its MLFlow and Neo4j dependencies, within a separate docker container. Therefore, if you get errors when running `Make docker_test`, a good sanity check is to try running the following command to see whether you can execute the pipeline locally (i.e. not in the container:
    ```
    kedro run -e test -p test --runner ThreadRunner 
    ```
    Note that for this command to work, you need to have your docker-daemon running (can be done by running `Make compose_up` or by turning it on via Docker Desktop). If you can successfully execute the pipeline through that command, then the issue is likely due to docker setup/docker container and might be due to specific OS environments - you can check for potential solutions in `common_errors.md`.


### Docker compose for local execution

Our codebase features code that allows for fully localized execution of the pipeline and
its' auxiliary services using `docker compose`. The deployment consists of two files that
can be [merged](https://docs.docker.com/compose/multiple-compose-files/merge/) depending
on the intended use, i.e.,

1. The base `docker compose` file defines the runtime services, i.e.,
    - Neo4J graph database
    - [MLFlow](https://www.mlflow.org/docs/latest/index.html) instance
    - [Mockserver](https://www.mock-server.com/) implementing a OpenAI compatible GenAI API
        - This allows for running the full pipeline e2e without a provider token
2. The `docker-compose.ci` file adds in the pipeline container for integration testing
    - This file is used by our CI/CD setup and can be ignored for local development.

![](../assets/img/docker-compose.drawio.svg)

Run the following command in the `workflows/matrix` directory to bring up the services
which are required by the pipeline. This way, you can develop the pipeline locally while
keeping the services running in the background.

```bash
make compose_up
```

To validate whether the setup is running, navigate to [localhost](http://localhost:7474/) in your browser, this will open the Neo4J dashboard. Use `neo4j` and `admin` as the username and password combination sign in. Please note that the Neo4J database would be empty at this stage.

### .env file for local credentials

If you want to execute the pipeline directly on your local machine (but without docker), you need to create a
.env file in the root of the `matrix` pipeline. Use the `.env.tmpl` file to get started
by renaming it to `.env`.

The key here is that the pipeline will not run fully without credentials
for the dependent services (at the moment only OpenAI). Reach out to the team through
Slack if you need a credential. 

### Plugging into cloud outputs

We run our full pipeline on production data in the `cloud` environment. Our pipeline is orchestrated using Argo Workflows, and may take several hours to complete. Given our current test process, that solely executes end-to-end tests on synthetic data, it is possible that the pipeline runs into an error due to a node handling data edge cases succesfully.

Troubleshooting such issues is tedious, and validating a fix requires the entire pipeline to be re-executed. This is where the `--from-env` flag comes in.

The figure below visualizes the same pipeline, with the environment distinguishing source/destination systems.

![](../assets/img/from-env-pipeline.drawio.svg)

Now imagine that a node in the pipeline fails. Debugging is hard, due to the remote execution environment. The `--from-env` flag allows for executing a step in the pipeline, in a given environment, while consuming the _input_ datasets from another environment.

![](../assets/img/from-env-run.drawio.svg)

In order to run against the `cloud` environment it's important to set the `RUN_NAME` variable in the `.env` file, as this determines the run for which datasets are pulled. You can find the `RUN_NAME` on the labels of the Argo Workflow.

The following command can be used to re-run the node locally, while consuming data from `cloud`:

```bash
# NOTE: You can specify multiple nodes to rerun, using a comma-seperated list
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
