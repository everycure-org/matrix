# Common Errors

!!! info "Collection of errors we see during development"
    This page is a collection of errors we've seen during dev and should help those
    that come after us debug issues we solved before. We need this because some errors appear when trying something else and that is not codified because we codify _what works_ not what we tried to get to this working state. However, reoccuring errors often occur in software engineering and experienced project members regularly help by "giving the solution" to the error that "they have seen before". This page seeks to collect those errors.


## Compute Engine Metadata server unavailable on attempt x out of 5

This error occurs when the authentication libraries from Google try to fetch
authentication credentials from the Compute Engine Metadata server. This occurs because
no credentials were found. When running on a personal machine, a `make fetch_secrets`
should fix this issue. This usually gets executed automatically when running `make` upon
setup of the local environment.


## Attempting to build local instance of matrix pipeline with Python 3.12

If you attempted to build the matrix pipeline locally with Python 3.12, it will fail due to the removal of distutils from Python after version 3.11. you may get a message that looks somewhat like the following:

```
error: Failed to prepare distributions
  Caused by: Failed to fetch wheel: numpy==1.23.5
  Caused by: Build backend failed to determine extra requires with `build_wheel()` with exit status: 1

...

ModuleNotFoundError: No module named 'distutils'
```


To fix this, remove the directory ".venv" from `pipelines/matrix` and set the python version to 3.11:


then `make` again.



## Hanging up during make process
While running `make`, you may encounter an issue where the process hangs up around the following step:

```
[build 12/12] ADD . .
```

If this step takes more than a few minutes, it is likely that the memory limit in docker is insufficient to build the docker image required to run the pipeline. To fix this issue, open docker and in **settings** adjust the memory limit.  16 GB is the recommended minimum. 

If you do not have access to lots of system memory, you can also increase the maximum Swap file size. 


## Unexpected keyword argument 'delimiter'

You may encounter the following error during install:

`TypeError: generate_random_arrays() got an unexpected keyword argument 'delimiter'`

This issue was caused when we updated a function in the `packages/data_fabricator` package. Since we cannot set a python version of a local dependency, `uv` caches this dependency and does not pull the latest version into the `venv.

**Solution**:
Run this inside of `pipelines/matrix`
```bash
rm -rf .venv/
rm -rf $(uv cache dir)
make venv
make install
```

This wipes the uv cache, which leads to a functioning installation of the library. This error does not occur in CI because our docker container is always built cleanly. 

Finally, try running the kedro pipeline test while docker is up and running:

```
kedro run -p test -e test
```

## Module not found in python
```
ModuleNotFoundError: No module named <some_module>
```

Someone added a new dependency to the project.
Run `make install`



## Sending empty batches to OpenAI / Neo4J

```
2024-07-26 08:47:47.672+0000 WARN  Error during iterate.commit:
2024-07-26 08:47:47.673+0000 WARN  1 times: org.neo4j.graphdb.QueryExecutionException: Failed to invoke procedure `apoc.ml.openai.embedding`: Caused by: java.lang.NullPointerException: Cannot invoke "java.util.List.get(int)" because "nonNullTexts" is null
2024-07-26 08:47:47.673+0000 WARN  Error during iterate.execute:
2024-07-26 08:47:47.673+0000 WARN  1 times: Cannot invoke "java.util.List.get(int)" because "nonNullTexts" is null
```


This is an obscure error, what actually happens is that we're trying to create an embedding call to OpenAI but we don't have anything to embed because all nodes already were embedded. Thus somehow Neo4J returns a null and passes this to the below function as a first argument (`nonNullTexts`)

```
cypher.CALL.openai_embedding(f"[item in $_batch | {'+'.join(f'coalesce(item.p.{item}, {empty})' for item in features)}]", "$apiKey", "{endpoint: $endpoint, model: $model}").YIELD("index", "text", "embedding")
```

## Spark throwing null pointers without actually having issues

```
24/07/26 10:52:32 ERROR Inbox: Ignoring error
java.lang.NullPointerException
	at org.apache.spark.storage.BlockManagerMasterEndpoint.org$apache$spark$storage$BlockManagerMasterEndpoint$$register(BlockManagerMasterEndpoint.scala:677)
	at org.apache.spark.storage.BlockManagerMasterEndpoint$$anonfun$receiveAndReply$1.applyOrElse(BlockManagerMasterEndpoint.scala:133)
	at org.apache.spark.rpc.netty.Inbox.$anonfun$process$1(Inbox.scala:103)
	at org.apache.spark.rpc.netty.Inbox.safelyCall(Inbox.scala:213)
	at org.apache.spark.rpc.netty.Inbox.process(Inbox.scala:100)
	at org.apache.spark.rpc.netty.MessageLoop.org$apache$spark$rpc$netty$MessageLoop$$receiveLoop(MessageLoop.scala:75)
	at org.apache.spark.rpc.netty.MessageLoop$$anon$1.run(MessageLoop.scala:41)
	at java.base/java.util.concurrent.ThreadPoolExecutor.runWorker(ThreadPoolExecutor.java:1128)
	at java.base/java.util.concurrent.ThreadPoolExecutor$Worker.run(ThreadPoolExecutor.java:628)
	at java.base/java.lang.Thread.run(Thread.java:829)
24/07/26 10:52:32 WARN Executor: Issue communicating with driver in heartbeater
org.apache.spark.SparkException: Exception thrown in awaitResult:
```

## MLFlow error about changing params when executing locally

```
    raise MlflowException(msg, INVALID_PARAMETER_VALUE)
mlflow.exceptions.MlflowException: INVALID_PARAMETER_VALUE: Changing param values is not allowed. Param with key='evaluation.simple_ground_truth_classification.evaluation_options.generator.object' was already logged with value='' for run ID='6814719483f34d5ca6e6f1b6f596715c'. Attempted logging new value 'matrix.datasets.pair_generator.GroundTruthTestPairs'.

The cause of this error is typically due to repeated calls
to an individual run_id event logging.

Incorrect Example:
---------------------------------------
with mlflow.start_run():
    mlflow.log_param("depth", 3)
    mlflow.log_param("depth", 5)
---------------------------------------

Which will throw an MlflowException for overwriting a
logged parameter.

Correct Example:
---------------------------------------
with mlflow.start_run():
    with mlflow.start_run(nested=True):
        mlflow.log_param("depth", 3)
    with mlflow.start_run(nested=True):
        mlflow.log_param("depth", 5)
---------------------------------------

Which will create a new nested run for each individual
model and prevent parameter key collisions within the
tracking store.
make: *** [integration_test] Error 1
```

Do 
```
docker stop mlflow
docker rm mlflow
docker-compose up -d
``` 

to wipe the local mlflow instance. This error occurs if one has previously run a node against a different environment. 


## gcloud command fails while venv is activated

```
E0815 10:47:32.491253   41064 memcache.go:265] couldn't get current server API group list: Get "https://34.123.77.254/api?timeout=32s": getting credentials: exec: executable /opt/homebrew/share/google-cloud-sdk/bin/gke-gcloud-auth-plugin failed with exit code 1
F0815 10:47:32.722549   41114 cred.go:145] print credential failed with error: Failed to retrieve access token:: failure while executing gcloud, with args [config config-helper --format=json]: exit status 1 (err: ERROR: gcloud crashed (AttributeError): module 'google._upb._message' has no attribute 'MessageMapContainer'

If you would like to report this issue, please run the following command:
  gcloud feedback

To check gcloud for common problems, please run the following command:
  gcloud info --run-diagnostics
)
E0815 10:47:32.723410   41064 memcache.go:265] couldn't get current server API group list: Get "https://34.123.77.254/api?timeout=32s": getting credentials: exec: executable /opt/homebrew/share/google-cloud-sdk/bin/gke-gcloud-auth-plugin failed with exit code 1
F0815 10:47:32.968713   41131 cred.go:145] print credential failed with error: Failed to retrieve access token:: failure while executing gcloud, with args [config config-helper --format=json]: exit status 1 (err: ERROR: gcloud crashed (AttributeError): module 'google._upb._message' has no attribute 'MessageMapContainer'

If you would like to report this issue, please run the following command:
  gcloud feedback

To check gcloud for common problems, please run the following command:
  gcloud info --run-diagnostics
)
E0815 10:47:32.969215   41064 memcache.go:265] couldn't get current server API group list: Get "https://34.123.77.254/api?timeout=32s": getting credentials: exec: executable /opt/homebrew/share/google-cloud-sdk/bin/gke-gcloud-auth-plugin failed with exit code 1
Unable to connect to the server: getting credentials: exec: executable /opt/homebrew/share/google-cloud-sdk/bin/gke-gcloud-auth-plugin failed with exit code 1
```

This is caused by a proto-plus dependency in your current python environment that is incompatible with `gcloud`. Because gcloud is also python based it uses the PYTHON_PATH dependencies of the venv. Not ideal, but there's a quick fix:

**Solution: Install `proto-plus` dev instance**
```bash
uv pip install proto-plus==1.24.0.dev1
```

## Docker running out of disk space (usually on MacOS/Windows)

```
 => ERROR [build  8/12] RUN uv pip install --system -r requirements.txt                                                                                                            23.4s
------
 > [build  8/12] RUN uv pip install --system -r requirements.txt:
8.842 Resolved 259 packages in 8.54s
23.31 error: Failed to prepare distributions
23.31   Caused by: Failed to fetch wheel: nvidia-nccl-cu12==2.22.3
23.31   Caused by: Failed to extract archive
23.31   Caused by: failed to write to file `/root/.cache/uv/.tmppT25gF/nvidia/nccl/lib/libnccl.so.2`
23.31   Caused by: No space left on device (os error 28)
------
Dockerfile:19
--------------------
  17 |     # Install Python dependencies
  18 |     ADD packages/ ./packages/
  19 | >>> RUN uv pip install --system -r requirements.txt
  20 |
  21 |     # Cache drivers as part of image
--------------------
ERROR: failed to solve: process "/bin/sh -c uv pip install --system -r requirements.txt" did not complete successfully: exit code: 2
make: *** [docker_push_dev] Error 1
```

Docker runs in a VM on MacOS. This can cause the disk to run full when building a lot of images. [This Stackoverflow post](https://stackoverflow.com/a/48561621) contains 2 options:
1. `docker system prune -a` to remove all files. This will lead you to re-build layers.
2. Expand the disk size of the docker VM


## Too many open files

```
WARNING  Retrying (Retry(total=2, connect=2, read=5, redirect=5, status=5)) after connection broken by 'NewConnectionError('<urllib3.connection.HTTPConnection object at   connectionpool.py:870
         0x30a712ad0>: Failed to establish a new connection: [Errno 24] Too many open files')':
         /api/2.0/mlflow-artifacts/artifacts?path=423706734757444990%2F3c4581c10f50444bb63fca093cd00e73%2Fartifacts%2Fxg_baseline
WARNING  Retrying (Retry(total=2, connect=2, read=5, redirect=5, status=5)) after connection broken by 'NewConnectionError('<urllib3.connection.HTTPConnection object at   connectionpool.py:870
         0x30a6efc10>: Failed to establish a new connection: [Errno 24] Too many open files')':
         /api/2.0/mlflow-artifacts/artifacts/423706734757444990/3c4581c10f50444bb63fca093cd00e73/artifacts/xg_baseline/MLmodel
WARNING  Retrying (Retry(total=2, connect=2, read=5, redirect=5, status=5)) after connection broken by 'NewConnectionError('<urllib3.connection.HTTPConnection object at   connectionpool.py:870
         0x30a73fa10>: Failed to establish a new connection: [Errno 24] Too many open files')':
```

Setting `ulimit -n 1000000` in the shell before running the pipeline will fix this issue.

## `make` stuck or takes forever on an ARM device (MacOS, some modern Windows or Linux)

1) on MacOS It may be necessary to update your rosetta license agreement.

```
softwareupdate --install-rosetta
```
Then agree to the license agreement with `A`

2) Generally, you can also run make with `TARGET_PLATFORM=linux/arm64` which ensures you build a docker image that is actually efficient for your device.

!!! note 
    We currently do not push the arm64 image to the registry because our cluster runs on intel CPUs. So this is only meant for local development.

## Permission denied error when trying to connect to Docker daemon on WSL

We noticed that some WSL users encountered the following error when onboarding, specifically when running `Make` for the first time:
```
permission denied while trying to connect to the Docker daemon socket at unix:///var/run/docker.sock: Post "http://%2Fvar%2Frun%2Fdocker.sock/v1.24/images/us-central1-docker.pkg.dev/mtrx-hub-dev-3of/matrix-images/matrix/tag?repo=us-central1-docker.pkg.dev%2Fmtrx-hub-dev-3of%2Fmatrix-images%2Fmatrix&tag=robynmac": dial unix /var/run/docker.sock: connect: permission denied
make: *** [Makefile:57: docker_build] Error 1
```

This error indicates that you have not added your user to the docker group. One solution to fix this is to run the following:
```
# Create a 'docker' group where members are allowed to interact with the Docker daemon w/o sudo 
sudo groupadd docker

# Add your user to the group
sudo usermod -aG docker ${USER}

# Refresh group membership (alternatively you can quit WSL (`wsl --shutdown`) and and re-open WSL terminal)
su -s ${USER}

# To test if it worked
docker run hello-world
```
If the hello-world container works, then you have successfully added your user to the group, and now you should have correct permissions to interact with docker. However if the error persists after that, you might want to modify the permissions of the docker socket directly. The following command gives the owner of the docker group members read and write access, ensuring that only user who owns the file and members of the docker group can communicate with the docker daemon.
```
sudo chmod 660 /var/run/docker.sock
```
Note that these permissions might reset when Docker or WSL is restarted, and you may need to reapply them.

## Runtime error `dictionary changed size during iteration`
```
matrix      | RuntimeError: dictionary changed size during iteration
matrix exited with code 1
```
You might encounter this when running integration tests within the docker container. This is a [kedro-related ThreadRunner error](https://github.com/kedro-org/kedro/issues/4191) and should be now fixed within the MATRIX pipeline by pinning a specific kedro version)[]. In case you stumble upon this error during your development, you solve this issue by specifying the following in the requirements.in
```
kedro==0.19.13
```

## MLFlow exception error when running the pipeline
```
MlflowException: API request to
http://127.0.0.1:5001/api/2.0/mlflow/experiments/create failed with exception
HTTPConnectionPool(host='127.0.0.1', port=5001): Max retries exceeded with url:
/api/2.0/mlflow/experiments/create (Caused by
NewConnectionError('<urllib3.connection.HTTPConnection object at 0x130404d90>:
Failed to establish a new connection: [Errno 61] Connection refused'))
```
This error is due to kedro trying to send API requests to your MLFlow container which hasn't been set up. You can set the MLFlow container from your Docker Desktop application or by running `make compose_up` from your terminal. This should set up a healthy docker container to which kedro can send API requests. 



## The RAW data files appear much smaller than expected


The issue arises as people have write permission to the RAW folder
and may accidentally run the fabricator pipeline in the cloud environment. This leads to
the pipeline writing the fabricator output to the raw datasets and because the cloud
environment is selected, the data is not stored in the cloud bucket. However, the full
raw data resides here, so the pipeline overwrites it. 

**Solution**: 
- Guardrails have been put in place to avoid this from happening again.
- If it still occurs, please notify the team. Our guardrails should prevent this from happening again.

!!! info
    This error should be resolved but in case it still occurs, please create a new issue
    and tag Every Cure devs. 

### Orphan container error when running docker
```
Error response from daemon: Conflict. The container name “/mockserver” is already in use by container “a2381853d58b482a3c4b82e17dbb25173e5af75903e98e7cb3481318f6abc7f1". You have to remove (or rename) that container to be able to reuse that name.
```
This error occurs when you have a container (mockserver in this instance) that is not properly removed and you try creating a new container with the same name. One solution to fix it is to run 
```
docker-compose up --remove-orphans
```
Alternatively you can remove the container by running `docker rm <container_id>` (in this case `docker rm mockserver`). **Note that this will remove the container and all data associated with it** so if it's your custom container, care should be taken - in such case you can rename the container instead:
```
docker rename mockserver mockserver_old
```

### MLFlow invalid parameter value error when doing kedro run 
```
MlflowException: INVALID_PARAMETER_VALUE: Invalid parameter name:
'fabricator.robokop.nodes.columns.equivalent_identifiers:string[].type'. Names may only
contain alphanumerics, underscores (_), dashes (-), periods (.), spaces ( ) and slashes (/).

The cause of this error is typically due to repeated calls
to an individual run_id event logging.
```
Parameter names in MLflow can only include alphanumeric characters, underscores (_), dashes (-), periods (.), spaces ( ), and slashes (/). The issue likely stems from repeated logging attempts to the same run_id, where one or more invalid characters (e.g., colons : or square brackets []) were used in the parameter name. 

To solve this issue, first ensure that you have only alphanumeric characters in the run name in your .env. Also make sure that you have disabled tracking for the `fabricator` pipeline in your mlflow.yml file:
```yaml
tracking:
  # You can specify a list of pipeline names for which tracking will be disabled
  # Running "kedro run --pipeline=<pipeline_name>" will not log parameters
  # in a new mlflow run

  disable_tracking:
    pipelines: ["fabricator"]
```

Alternatively, you can try to remove the run from the MLFlow UI and re-run the pipeline. You can do it with the following command:
```bash
docker stop mlflow
docker rm -f mlflow
```
Then follow Make instructions to setup the container again. 

### API request error when running the pipeline
```
MlflowException: API request to
http://127.0.0.1:5001/api/2.0/mlflow/experiments/create failed with exception
HTTPConnectionPool(host='127.0.0.1', port=5001): Max retries exceeded with url:
/api/2.0/mlflow/experiments/create (Caused by
NewConnectionError('<urllib3.connection.HTTPConnection object at 0x130404d90>:
Failed to establish a new connection: [Errno 61] Connection refused'))
```
This error is due to the MLFlow container not being set up (can also encounter it for neo4j or mockserver). You can set the MLFlow container from your Docker Desktop application or by running `make compose_up` from your terminal. This should set up a healthy docker container to which kedro can send API requests. 

### No Partition defined for this dataset
```
matrix      | [09/21/24 04:47:03] INFO     Saving data to                  data_catalog.py:581
matrix      |                              embeddings.feat.graph.node_embe
matrix      |                              ddings (LazySparkDataset)...
24/09/21 04:47:03 WARN WindowExec: No Partition Defined for Window operation! Moving all data to a single partition, this can cause serious performance degradation.
matrix      | 24/09/21 04:47:03 WARN WindowExec: No Partition Defined for Window operation! Moving all data to a single partition, this can cause serious performance degradation.
24/09/21 04:47:04 WARN WindowExec: No Partition Defined for Window operation! Moving all data to a single partition, this can cause serious performance degradation.
matrix      | 24/09/21 04:47:04 WARN WindowExec: No Partition Defined for Window operation! Moving all data to a single partition, this can cause serious performance degradation.
```
You dont need to worry about this warning as this is an expected behavior for our implementation of `LazySparkDataset` in Kedro, allowing us for parallelized computation of embeddings. In the future, we might refactor the code so that the error is not appearing.
### "Environment variable 'OPENAI_API_KEY' not found or default value None is None"
This error is likely due to having no OPENAI_API_KEY environment variable set up. First, ensure that you have copied the contents of `.env.tmpl `file into `.env` file. This should allow kedro to read environmental variables set up in .env  Note that if you need the actual OPENAI_API_KEY, you will need to reach out to our team
### Quota project error
```
WARNING: Your active project does not match the quota project in your local Application Default Credentials file. This might result in unexpected quota issues.

To update your Application Default Credentials quota project, use the `gcloud auth application-default set-quota-project` command.
Updated property [core/project].
```
As the error suggests, your active project likely does not match the quota project in your local Application Default Credentials file. You can solve this by running the following command:
```bash
gcloud auth application-default set-quota-project mtrx-hub-dev-3of
```

### Fabricator error `IndexError: tuple index out of range` 
```
ERROR    Node fabricate_kg2_datasets: fabricate_datasets() ->  failed with error:                                                                                node.py:386
                             tuple index out of range  
IndexError: tuple index out of range
```
This error is most likely due to data fabricator version being outdated after the most recent updates to main. A simple solution is to clean your current venv, re-create and re-install the venv with needed dependencies:
```
uv cache clean # if you use pip or other library management tool, you should clean cache of that
deactivate # ensure no venv is active
make clean
make
```

However we noted that error persists if you have miniconda3 or conda installed on your system. Note that conda and uv (which is a preferred package management system) are very incompatible and using both might lead to errors. Therefore, if you run the command above and still get the IndexError, please make sure you have no miniconda installed. If you do have miniconda on your system, you might need to remove it or ensure it's completely separated. Once it's removed, you should re-do re-create the matrix repo and re-install venv as mentioned above


### Error reading data from cloud
```
DatasetError: 
Project was not passed and could not be determined from the environment..
```

After setting up installing the gcloud SDK, make sure that a default project is set by running:
```
gcloud config list
```
If no project is listed, then it can be set by running:

```
gcloud config set project mtrx-hub-dev-3of
```

### Issue with kedro run -e test -p test after updating git pull.

```
filter_by_category() missing 1 required positional argument: 'categories
```

This is due to our local libraries (e.g. `data_fabricator`) being cached in the uv cache
and thus not being installed to the latest version when running `make install`. Cleaning
the uv cache solves this issue which you can do via `make clean` and then run a fresh
`make install`.


### Issues with Neo4j authentication

```
ServiceUnavailable: Couldn't connect to 127.0.0.1:7687 (resolved to ()):

OR various other authentication errors related to Neo4j or issues with Spark failing when it attempts to write to Neo4j
```

This may be due to another neo4j instance running on your device. Common sources for these services may be either brew or neo4j desktop. 

To check brew for running neo4j instances, run:

```bash
brew services list
```

if you see neo4j running, try:

```bash
brew services stop neo4j
```


### libomp for LLMs

The [libomp](https://openmp.llvm.org/index.html) library might be required as a local runtime for LLMs. If not installed it will trigger an error containing the following:

```
* OpenMP runtime is not installed
  - vcomp140.dll or libgomp-1.dll for Windows
  - libomp.dylib for Mac OSX
  - libgomp.so for Linux and other UNIX-like OSes
  Mac OSX users: Run `brew install libomp` to install OpenMP runtime.
```

To install it on MacOS, run:

```bash
brew install libomp
```

## Error: Permission Denied When Deleting RAW Data Files

If you attempt to delete files directly from the RAW data bucket, you may encounter a permission denied error or find that you do not have the necessary rights to perform deletions. This is intentional: to protect the integrity of our RAW data, direct delete permissions are not granted to individual users.

**Why:**
- RAW data is critical and must not be accidentally or maliciously deleted.
- We enforce a "four eyes" principle for deletions to ensure all removals are reviewed.

**Solution:**
- If you need to delete files from the RAW bucket, do **not** attempt to do so manually.
- Instead, add the paths you wish to delete to `scripts/cleanup_files.txt` and submit a PR.
- Once reviewed and merged, the automated cleanup process (see `.github/workflows/cleanup_raw_bucket.yml`) will safely delete the files using a controlled, auditable workflow.

This process ensures data safety and compliance with our data governance policies.
