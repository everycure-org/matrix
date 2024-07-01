# Local development and execution

Our codebase features code that allows for fully localized execution of the pipeline and its' auxiliary services using `docker-compose`. The deployment consists of two files that can be [merged](https://docs.docker.com/compose/multiple-compose-files/merge/) depending on the intended use, i.e.,

1. The `base` file defines the runtime services, i.e.,
    - Neo4J graph database
    - [Mockserver](https://www.mock-server.com/) implementing a OpenAI compatible GenAI API
2. The `test` file adds in the pipeline container for integration testing

> NOTE: The `test` file is used in our CI to orchestrate the tests.

![](../assets/img/docker-compose.drawio.svg)

## Pre-requisites

### Docker

Make sure you have docker and docker-compose installed. Docker can be downloaded directly from the from the [following page](https://docs.docker.com/desktop/install/mac-install/). Once installed, proceed with the following command:

```bash
brew install docker-compose
```

> NOTE: The default settings of Docker have rather low resources configured, you might want to increase those in Docker desktop.

### Java

Our pipeline uses Spark for distributed computations, which requires Java under the hood.

```
brew install openjdk@11
```

> NOTE: Don't forget to link your Java installation using the instructions prompted after the downloaded.

### GCP Service account

To correctly leverage the GCP services, you will need a service-account key. You can create a key through the [Google CLI](https://cloud.google.com/storage/docs/gsutil_install) as follows:

> NOTE: This will be provisoned using Terraform and git-crypt in the future.

!!! note

    The below token is time bound and thus this command needs to be re-run regularly, it's only meant for temporary local testing, not for a long-running workload. 

```bash
gcloud config set project mtrx-hub-dev-3of
gcloud iam service-accounts keys create --iam-account=test-gcp@mtrx-hub-dev-3of.iam.gserviceaccount.com  conf/local/service-account.json
```

### Vertex token

To succesfully run embeddings in the `prod` environment, export your GCP access token using the following command:

```bash
export VERTEX_AI_ACCESS_TOKEN=$(gcloud auth print-access-token)
```

## Launching the deployment

After completing the installation, run the following command from the `deployments/compose` directory to bring up the services.

```
docker-compose up
```

> Alternatively, you can add the `-d` flag at the end of the command to run in the background.

To validate whether the setup is running, navigate to [localhost](http://localhost:7474/) in your browser, this will open the Neo4J dashboard. Use `neo4j` and `admin` as the username and password combination sign in.

The pipeline is configured to pick-up these services automatically, so you can execute the pipeline as usual.