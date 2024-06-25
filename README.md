# Local development and execution

Our codebase features code that allows for fully localized execution of the pipeline and its' auxiliary services using `docker-compose`. This guide assists you in setting it up.

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

## Launching the deployment

After completing the installation, run the following command to bring up the services.

```
docker-compose up
```

> Alternatively, you can add the `-d` flag at the end of the command to run in the background.

To validate whether the setup is running, navigate to [localhost](http://localhost:7474/) in your browser, this will open the Neo4J dashboard. Use `neo4j` and `admin` as the username and password combination sign in.

The pipeline is configured to pick-up these services automatically, so you can execute the pipeline as usual.