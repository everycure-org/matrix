# Common Errors

!!! info "Collection of errors we see during development"
    This page is a collection of errors we've seen during dev and should help those
    that come after us debug issues we solved before. We need this because some errors appear when trying something else and that is not codified because we codify _what works_ not what we tried to get to this working state. However, reoccuring errors often occur in software engineering and experienced project members regularly help by "giving the solution" to the error that "they have seen before". This page seeks to collect those errors.


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

```
rm -r .venv
pyenv install 3.11
pyenv global 3.11
```

then `make` again.


## Module not found in python
```
ModuleNotFoundError: No module named <some_module>
```

Someone added a new dependency to the project.
Run `make install`



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

TODO

## Failed batches in the embedding step


```
│ main   File "/usr/local/lib/python3.11/site-packages/kedro/runner/runner.py", line 117, in run                                                                                                                   │
│ main     self._run(pipeline, catalog, hook_or_null_manager, session_id)  # type: ignore[arg-type]                                                                                                                │
│ main     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^                                                                                                                                          │
│ main   File "/usr/local/lib/python3.11/site-packages/kedro/runner/sequential_runner.py", line 75, in _run                                                                                                        │
│ main     run_node(node, catalog, hook_manager, self._is_async, session_id)                                                                                                                                       │
│ main   File "/usr/local/lib/python3.11/site-packages/kedro/runner/runner.py", line 413, in run_node                                                                                                              │
│ main     node = _run_node_sequential(node, catalog, hook_manager, session_id)                                                                                                                                    │
│ main            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^                                                                                                                                    │
│ main   File "/usr/local/lib/python3.11/site-packages/kedro/runner/runner.py", line 506, in _run_node_sequential                                                                                                  │
│ main     outputs = _call_node_run(                                                                                                                                                                               │
│ main               ^^^^^^^^^^^^^^^                                                                                                                                                                               │
│ main   File "/usr/local/lib/python3.11/site-packages/kedro/runner/runner.py", line 472, in _call_node_run                                                                                                        │
│ main     raise exc                                                                                                                                                                                               │
│ main   File "/usr/local/lib/python3.11/site-packages/kedro/runner/runner.py", line 462, in _call_node_run                                                                                                        │
│ main     outputs = node.run(inputs)                                                                                                                                                                              │
│ main               ^^^^^^^^^^^^^^^^                                                                                                                                                                              │
│ main   File "/usr/local/lib/python3.11/site-packages/kedro/pipeline/node.py", line 392, in run                                                                                                                   │
│ main     raise exc                                                                                                                                                                                               │
│ main   File "/usr/local/lib/python3.11/site-packages/kedro/pipeline/node.py", line 380, in run                                                                                                                   │
│ main     outputs = self._run_with_dict(inputs, self._inputs)                                                                                                                                                     │
│ main               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^                                                                                                                                                     │
│ main   File "/usr/local/lib/python3.11/site-packages/kedro/pipeline/node.py", line 437, in _run_with_dict                                                                                                        │
│ main     return self._func(**kwargs)                                                                                                                                                                             │
│ main            ^^^^^^^^^^^^^^^^^^^^                                                                                                                                                                             │
│ main   File "/usr/local/lib/python3.11/site-packages/refit/v1/core/unpack.py", line 62, in wrapper                                                                                                               │
│ main     result = func(*args, **kwargs)                                                                                                                                                                          │
│ main              ^^^^^^^^^^^^^^^^^^^^^                                                                                                                                                                          │
│ main   File "/usr/local/lib/python3.11/site-packages/refit/v1/core/inject.py", line 171, in wrapper                                                                                                              │
│ main     result_df = func(*args, **kwargs)                                                                                                                                                                       │
│ main                 ^^^^^^^^^^^^^^^^^^^^^                                                                                                                                                                       │
│ main   File "/app/src/matrix/pipelines/embeddings/nodes.py", line 166, in compute_embeddings                                                                                                                     │
│ main     raise RuntimeError("Failed batches in the embedding step")                                                                                                                                              │
│ main RuntimeError: Failed batches in the embedding step                                                                                                                                                          │
│ main time="2024-07-26T09:55:54.436Z" level=info msg="sub-process exited" argo=true error="<nil>"                                                                                                                 │
│ main Error: exit status 1
```

This often means there is a `400` error from OpenAI or another backend issue. We throw this error ourselves explicitly to catch API errors.
Unfortuantely one has to dig into the Debug Log of Neo4J to find out the exact issue

1. connect to neo4j instance
2. cd to `logs`
3. tail / grep on `debug.log` and check what was logged by the DB


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


