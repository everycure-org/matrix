---
title: Local setup
---

Our codebase is structured around the `Makefile`. This allows for a quick and easy setup of the local environment using one command only - the video below explains the Makefile structure and how it relates to our codebase in more detail:

<div style="position: relative; width: 100%; height: 0; padding-bottom: 56.25%;"><iframe src="https://us06web.zoom.us/clips/embed/ghxELqxMExaMh96j_58dMq8UnXaXEcEtSTRVxXf7zXNd5l4OSgdvC5xwANUE1ydp7afd-M42UkQNe_eUJQkCrXIZ.SAPUAWjHFb-sh7Pj" frameborder="0" allowfullscreen="allowfullscreen" style="position: absolute; width: 100%; height: 100%; top: 0; left: 0; "></iframe></div>

At the end of this section, we will show you how you can set up your entire local environment by using a single `make` command however prior to that, we will explain the set-up on a step-by-step basis.

<!--
TODO wrong place
### Environment Variables Setup

To execute the pipeline directly on your local machine, you'll need to set up your environment variables. We use a two-file system:

1. `.env.defaults` - Contains shared default values and is version controlled. This file includes all available configuration options with default values and documentation.
2. `.env` - Contains your local overrides and credentials (gitignored)

!!! tip
    Start by reviewing `.env.defaults` to understand available configuration options. In most cases, for local executions no overrides are needed. If you do want to override a variable, create the `.env` file and override the necessary variables.

-->

### Virtual environment for python dependencies

To execute the codebase, you need to set up a virtual environment for the python dependencies. This can be done by running the following command in the `pipelines/matrix` directory:

```bash
make install
```

!!! question "What does this command do?"

    This command wraps the following commands:

    1. **Environment Setup**: Creates and configures your Python development environment with all necessary dependencies
    2. **Code Validation**: Runs various unit tests to ensure your local setup is working correctly
    3. **Spins up docker services**: Spins up several dependent services we need for the execution of the pipeline
    4. **Integration Test**: Runs an end-to-end test of the pipeline using fabricated data to verify the pipeline is working as expected

    This single command gets you from a fresh checkout to a fully functional local development environment. After it completes successfully, you'll be ready to start developing!
    
    We do encourage you to read through the `Makefile` to get a sense of how the codebase is structured.


### Pre-commit hooks
We have pre-commit hooks installed to ensure code quality and consistency. To run the pre-commit hooks, you can run the following command:

```bash
make precommit
```

These hooks were also installed at the time you called `make` so whenever you try to push something to the repository, the hooks will run automatically. We ensure a minimum level of code quality this way.

### Fast tests
To ensure that the codebase is working as expected, you can run the following command to execute the fast tests:

```bash
make fast_test
```

Note that the first time you run this command, it might take a while to complete as it
needs to run all tests. However any other fast_test command will be faster as it will use
the cached data and only execute tests where the underlying code has changed.

### Docker compose for local execution

Our codebase features code that allows for fully local execution of the pipeline and
its auxiliary services using `docker compose`. The deployment consists of two files that
can be [merged](https://docs.docker.com/compose/multiple-compose-files/merge/) depending
on the intended use, i.e.,

1. The base `docker compose` file defines the runtime services, i.e.,
    - Neo4J graph database
    - [MLFlow](https://www.mlflow.org/docs/latest/index.html) instance
    - [Mockserver](https://www.mock-server.com/) implementing an OpenAI compatible GenAI API
        - This allows for running the full pipeline e2e without a provider token
2. The `docker-compose.ci` file adds in the pipeline container for integration testing
    - This file is used by our CI/CD setup and can be ignored for local development.

![](../assets/img/docker-compose.drawio.svg)

Run the following `Makefile` command in the `pipelines/matrix` directory to bring up the services
which are required by the pipeline. This way, you can develop the pipeline locally while
keeping the services running in the background.

```bash
make compose_up
```

To validate whether the setup is running, navigate to [localhost](http://localhost:7474/) in your browser, this will open the Neo4J dashboard. Use `neo4j` and `admin` as the username and password combination sign in. Please note that the Neo4J database would be empty at this stage.

### Kedro test run

To run the pipeline end-to-end locally using a fabricated dataset, you can run the following command:

```bash
make integration_test
```
This command wraps the following commands (provided venv is active and that you have your docker containers up and running):
```bash
.venv/bin/kedro run --env test -p test --runner ThreadRunner --without-tags xgc,not-shared
```
This command will kick off our kedro pipeline in a test environment using a fabricated dataset. This is useful to ensure that the pipeline works as expected locally after you are finished with the local setup.

### Makefile setup

Generally, the `Makefile` is a good place to refer to when you need to re-set your environment. Once the command runs successfully, you should be able to run the pipeline end-to-end locally!

!!! help "Encountering issues?"
    If you're experiencing any problems running the `MakeFile`, please refer to our [Common Errors FAQ](../FAQ/common_errors.md) for troubleshooting guidance. This resource contains solutions to frequently encountered issues and may help resolve your problem quickly.


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
    We encourage you to examine the `Makefile` to get a sense of how the codebase is structured - note that not all commands are part of the default make command. Some useful commands which are not part of the default make command are:

    - `Make full_test` which executes all unit tests within the codebase
    - `Make clean` which cleans various cache locations - useful when you're running into cache related issues or when you want to start fresh
    - `Make wipe_neo` which wipes the neo4j database - useful for development purposes
    - `Make docker_test` which executes e2e integration tests by running the pipeline in a docker container - useful for debugging CI issues 
    - `Make format` which formats the codebase using `ruff`
    - `Make fabricate` which runs the fabricator pipeline

### Plugging into cloud outputs

As you might have noticed, local setup is mainly focusing on the local execution of the pipeline with the fabricated data. However, we also run our full pipeline on production data in the `cloud` environment. Our pipeline is orchestrated using Argo Workflows, and may take several hours to complete. Given our current test process, that solely executes end-to-end tests on synthetic data, it is possible that the pipeline runs into an error due to a node handling data edge cases unsuccesfully.

Troubleshooting such issues is tedious, and validating a fix requires the entire pipeline to be re-executed. This is where the `--from-env` flag comes in.

The figure below visualizes the same pipeline, with the environment distinguishing source/destination systems.

![](../assets/img/from-env-pipeline.drawio.svg)

Now imagine that a node in the pipeline fails. Debugging is hard, due to the remote execution environment. The `--from-env` flag allows for executing a step in the pipeline, in a given environment, while consuming the _input_ datasets from another environment.

![](../assets/img/from-env-run.drawio.svg)

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

[Next up, our different environments in kedro :material-skip-next:](./environments_overview.md){ .md-button .md-button--primary }
