---
title: Local setup
---

As [mentioned earlier](./repo_structure.md#core-project-files), our codebase is structured around the `Makefile`. This allows for a quick and easy setup of the local environment using one command only - the video below explains the Makefile structure and how it relates to our codebase in more detail:

<div style="position: relative; width: 100%; height: 0; padding-bottom: 56.25%;"><iframe src="https://us06web.zoom.us/clips/embed/ghxELqxMExaMh96j_58dMq8UnXaXEcEtSTRVxXf7zXNd5l4OSgdvC5xwANUE1ydp7afd-M42UkQNe_eUJQkCrXIZ.SAPUAWjHFb-sh7Pj" frameborder="0" allowfullscreen="allowfullscreen" style="position: absolute; width: 100%; height: 100%; top: 0; left: 0; "></iframe></div>


!!! tip 
    For the impatient ones, just run `make`. No errors? Great you're all set! It is probably still worth reading this page
    to understand everything that's happening though. 

## Set up your local Environment with Make

### Virtual environment for python dependencies

To execute the codebase, you need to set up a virtual environment for the python dependencies. This can be done by running the following command in the `pipelines/matrix` directory:

```bash
make install
```

!!! question "What does this command do?"

    This command wraps the following commands:

    1. **Virtual Environment Creation**: Creates a virtual environment using `uv`
    2. **Pre-commit setup**: Installs pre-commit hooks and executes pre-commits check on the repo
    3. **Installs Dependencies**: Installs all dependencies from requirements.txt using uv

    We do encourage you to read through the `Makefile` to get a sense of how the codebase is structured. There are many commands (some are for diagnostics, others for GCP setup) there therefore we recommend focusing on the ones you are interested in running them!


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
    - [MLFlow](https://www.mlflow.org/docs/latest/index.html) instance (for cloud environment)
    - [Mockserver](https://www.mock-server.com/) implementing an OpenAI compatible GenAI API
        - This allows for running the full pipeline e2e without a provider token
2. The `docker-compose.ci` file adds in the pipeline container for integration testing
    - This file is used by our CI/CD setup and can be ignored for local development.

![](../../assets/img/docker-compose.drawio.svg)

Run the following `Makefile` command in the `pipelines/matrix` directory to bring up the services
which are required by the pipeline. This way, you can develop the pipeline locally while
keeping the services running in the background.

```bash
make compose_up
```

To validate whether the setup is running, navigate to [localhost](http://localhost:7474/) in your browser, this will open the Neo4J dashboard. Use `neo4j` and `admin` as the username and password combination sign in. Please note that the Neo4J database would be empty at this stage.

### Kedro test run

Now you should be ready to run the pipeline end-to-end locally using a fabricated dataset! You can run the following command:

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
    If you're experiencing any problems running the `MakeFile`, please refer to our [Common Errors FAQ](../../references/common_errors.md) for troubleshooting guidance. This resource contains solutions to frequently encountered issues and may help resolve your problem quickly.

Congrats on successfully running the MATRIX pipeline with fabricated data! In the deep-dive we will explain exactly how the fabricator works and what happened in detail but now, we will explain to you how to run the pipeline in different environments before running it with a real (but sampled) data.

[Check our environment overview :material-skip-next:](./environments_overview.md){ .md-button .md-button--primary }
