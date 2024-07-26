# Common Errors

!!! info "Collection of errors we see during development"
    This page is a collection of errors we've seen during dev and should help those
    that come after us debug issues we solved before. We need this because some errors appear when trying something else and that is not codified because we codify _what works_ not what we tried to get to this working state. However, reoccuring errors often occur in software engineering and experienced project members regularly help by "giving the solution" to the error that "they have seen before". This page seeks to collect those errors.



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
