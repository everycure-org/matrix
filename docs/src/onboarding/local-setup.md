---
title: Local setup
---

Our codebase is structured around the `Makefile`. This allows for a quick and easy setup of the local environment using one command only. At the end of this section, we will show you how you can set up your entire local environment by using a single `make` command however prior to that, we will explain the set-up on a step-by-step basis.

### .env file for local credentials

To execute the pipeline directly on your local machine, you'll first need to create a `.env` file in the root of the matrix pipeline. Get started by renaming the `.env.tmpl` file to `.env`.

Note that the pipeline will not run fully without credentials for the dependent services (at the moment only OpenAI). Reach out to the team through Slack if you need a credential. 

### Virtual environment for python dependencies

To execute the codebase, you need to set up a virtual environment for the python dependencies. This can be done by running the following command in the `pipelines/matrix` directory:

```bash
make install

# This command wraps the following commands:

# Checking the pre-requisites installed
@command -v docker >/dev/null 2>&1 || { echo "Error: docker is not installed." >&2; exit 1; }
@command -v gcloud >/dev/null 2>&1 || { echo "Error: gcloud is not installed." >&2; exit 1; }
@command -v python3 >/dev/null 2>&1 || { echo "Error: python3 is not installed." >&2; exit 1; }
@command -v java >/dev/null 2>&1 || { echo "Error: java is not installed." >&2; exit 1; }
@command -v uv >/dev/null 2>&1 || { echo "Error: uv is not installed." >&2; exit 1; }
@echo "All prerequisites are installed."

# Setting up a virtual environment and installing the dependencies
if [ ! -d .venv ]; then uv venv -p 3.11; fi
.venv/bin/python -m ensurepip --upgrade || true
deactivate || true
# activate freshly created venv
(source .venv/bin/activate; \
uv pip compile requirements.in --output-file requirements.txt; \
uv pip install -r requirements.txt)
```

### Pre-commit hooks
We have pre-commit hooks installed to ensure code quality and consistency. To run the pre-commit hooks, you can run the following command:

```bash
make precommit

# This command wraps the following commands (provided venv is active):
uv pip install pre-commit
.venv/bin/pre-commit install --install-hooks

# Fetch updates from remote and run pre-commit hooks against diff between main and current branch
git fetch origin
.venv/bin/pre-commit run -s origin/main -o HEAD
```

### Fast tests
To ensure that the codebase is working as expected, you can run the following command to execute the fast tests:

```bash
make fast_test

# This command wraps the following commands (provided venv is active):
TESTMON_DATAFILE=/tmp/.testmondata .venv/bin/pytest --testmon -v tests/
```
Note that the first time you run this command, it might take a while to complete as it needs to download the testmon data file. However any other fast_test command will be faster as it will use the cached data file.

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

Run the following `Makefile` command in the `workflows/matrix` directory to bring up the services
which are required by the pipeline. This way, you can develop the pipeline locally while
keeping the services running in the background.

```bash
make compose_up

# Alternatively you can also run the following command from matrix/pipelines/compose directory to bring up the services
docker compose -f compose/docker-compose.yml up -d --wait --remove-orphans
```

To validate whether the setup is running, navigate to [localhost](http://localhost:7474/) in your browser, this will open the Neo4J dashboard. Use `neo4j` and `admin` as the username and password combination sign in. Please note that the Neo4J database would be empty at this stage.

### Kedro test run

To run the pipeline end-to-end locally using a fabricated dataset, you can run the following command:

```bash
make integration_test

# This command wraps the following commands (provided venv is active and that you have your docker containers up and running):
.venv/bin/kedro run --env test -p test --runner ThreadRunner --without-tags xgc,not-shared
```
This command will kick off our kedro pipeline in a test environment using a fabricated dataset. This is useful to ensure that the pipeline works as expected locally after you are finished with the local setup.

### Makefile setup

Except for the `.env` setup, all the sections can be set-up automatically using `Makefile`. Thus, the fastest way to check if everything works locally is to execute the following command in `pipelines/matrix`

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
- set up containers for local execution
- run integration tests

Generally, the `Makefile` is a good place to refer to when you need to re-set your environment. Once the command runs successfully, you should be able to run the pipeline end-to-end locally!

<div style="position: relative; width: 100%; height: 0; padding-bottom: 56.25%;"><iframe src="https://us06web.zoom.us/clips/embed/ghxELqxMExaMh96j_58dMq8UnXaXEcEtSTRVxXf7zXNd5l4OSgdvC5xwANUE1ydp7afd-M42UkQNe_eUJQkCrXIZ.SAPUAWjHFb-sh7Pj" frameborder="0" allowfullscreen="allowfullscreen" style="position: absolute; width: 100%; height: 100%; top: 0; left: 0; "></iframe></div>

??? note "Understanding and troubleshooting your `make` run"
   
   ### Understanding the `make` command
    As mentioned in the video, our pipeline is evolving quickly. Running `make` is a simple and quick way to check if everything works locally as it is composed of several different stages which get executed one after another. When `make` gets executed, what is happening under the hood is that the following seven 'make' "subcommands" get executed:
    ```
    Make prerequisites # checks if all prerequisites are set up correctly
    Make venv # sets up a virtual environment using uv
    Make install # installs all dependencies within the virtual environment
    Make precommit # installs and runs pre-commit hooks
    Make fast_test # tests the codebase 
    Make compose_down # ensures that there is no docker-compose down running 
    Make integration_test # executes an local kedro pipeline run in a test environment
    ```
    We encourage you to examine the `Makefile` to get a sense of how the codebase is structured - note that not all commands are part of the default make command. Some useful command which are not part of the default make command are:
    - `Make full_test` which executes all unit tests within the codebase
    - `Make clean` which cleans various cache locations - useful when you're running into cache related issues or when you want to start fresh
    - `Make wipe_neo` which wipes the neo4j database - useful for development purposes
    - `Make docker_test` which executes e2e integration tests by running the pipeline in a docker container - useful for debugging CI issues 
    - `Make format` which formats the codebase using `ruff`
    - `Make fabricate` which runs the fabricator pipeline

### Plugging into cloud outputs

As you might have noticed, local setup is mainly focusing on the local execution of the pipeline with the fabricated data. However, we also run our full pipeline on production data in the `cloud` environment. Our pipeline is orchestrated using Argo Workflows, and may take several hours to complete. Given our current test process, that solely executes end-to-end tests on synthetic data, it is possible that the pipeline runs into an error due to a node handling data edge cases succesfully.

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
