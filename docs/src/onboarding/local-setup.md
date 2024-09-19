---
title: Local setup
---

The fastest way to check if everything works locally is to execute the following command in `pipelines/matrix`

```
make
```

This command executes a number of make targets, namely:
- set up a virtual environment
- install the python dependencies
- lint the codebase
- test the codebase
- build the docker image
- run the docker image

Generally, the `Makefile` is a good place to start to get a sense of how our codebase is structured. 

<div style="position: relative; width: 100%; height: 0; padding-bottom: 56.25%;"><iframe src="https://us06web.zoom.us/clips/embed/ghxELqxMExaMh96j_58dMq8UnXaXEcEtSTRVxXf7zXNd5l4OSgdvC5xwANUE1ydp7afd-M42UkQNe_eUJQkCrXIZ.SAPUAWjHFb-sh7Pj" frameborder="0" allowfullscreen="allowfullscreen" style="position: absolute; width: 100%; height: 100%; top: 0; left: 0; "></iframe></div>

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

To validate whether the setup is running, navigate to [localhost](http://localhost:7474/) in your browser, this will open the Neo4J dashboard. Use `neo4j` and `admin` as the username and password combination sign in.

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

![](../assets/img/from-env-pipeline.png)

Now imagine that a node in the pipeline fails. Debugging is hard, due to the remote execution environment. The `--from-env` flag allows for executing a step in the pipeline, in a given environment, while consuming the _input_ datasets from another environment.

![](../assets/img/from-env-run.png)

In order to run against the `cloud` environment it's important to set the `RUN_NAME` variable in the `.env` file, as this determines the run for which datasets are pulled.

!!! note 
    If you wish to pull data from MLFlow it's currently required to setup [port-forwarding](https://emmer.dev/blog/port-forwarding-to-kubernetes/) into the MLFlow tracking container on the cluster. You can do this as follows:

    ```bash
    kubectl port-forward -n mlflow pod/mlflow-tracking-54f5bcd66c-7k9mc 5002:5000
    ```

    Next, add the following entry to your `.env` file.

    ```dotenv
    MLFLOW_ENDPOINT=http://127.0.0.1:5002
    ```
