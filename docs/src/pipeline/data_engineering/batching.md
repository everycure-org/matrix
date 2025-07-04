# Using Kedro to process datasets in batches asynchronously

In this section you will deep-dive into how we optimized our embedding generation whilst leveraging PySpark and Kedro functionalities. 

## Preliminaries

Our organisation has been focussing on large-scale data enrichment. One of the problems we've ran into at multiple occasions is using an external API to enrich elements in a huge dataset in batched manner, i.e., millions of rows. The problem becomes more prevalent whenever the upstream API has rate limiting enabled, e.g., OpenAI for computing embeddings.

## Initial approach

We had an initial spark operation that would perform batching and invoke a UDF that implements the OpenAI invocation on a batch.

```python
def compute_embeddings(
    input: DataFrame,
    features: List[str],
    fn: Callable # pyspark udf to call OpenAI
):

    window = Window.orderBy(F.lit(1))

    res = (
        input.withColumn("row_num", F.row_number().over(window))
        .withColumn("batch", F.floor((F.col("row_num") - 1) / batch_size))
        .withColumn("input", F.concat(*[F.coalesce(F.col(feature), F.lit("")) for feature in features]),
        )
        .groupBy("batch")
        .agg(
            F.collect_list("id").alias("id"),
            F.collect_list("input").alias("input"),
        )
        .withColumn(attribute, batch_udf(F.col("input")))
        .withColumn("_conc", F.arrays_zip(F.col("id"), F.col(attribute)))
        .withColumn("exploded", F.explode(F.col("_conc")))
        .select(
            F.col("exploded.id").alias("id"),
            F.col(f"exploded.{attribute}").alias(attribute),
        )
        .join(input, on="id")
    )

    return res
```

The operation above has huge hardware requirements, and the parallelism is correlated to the number of partitions in PySpark. In multiple occasions the function above would come to a halt, signaling further resource issues. We felt like using PySpark for this was not the right choice.

## Towards a better approach

> Hive partitioning: our approach relies on hive partitioning to produce a batched dataset. Hive partitioning is an idea where a column of the data is 
> "promoted" to directories on file storage. For instance, if we use hive partitioning for a table with columns [id, batch], the file on object storage
> will be stored as follows:
>
> .
> └── dataset/
>     ├── batch=1/
>     │   └── df.parquet
>     ├── batch=2/
>     │   └── df.parquet
>     └── ...


### Bucketizing the input dataframe

As a first step, we're making sure to add a `batch` column to our data.

```python
def bucketize_df(df: DataFrame, bucket_size: int, input_features: List[str], max_input_len: int):
    """Function to bucketize input dataframe.

    Function bucketizes the input dataframe in N buckets, each of size `bucket_size`
    elements. Moreover, it concatenates the `features` into a single column and limits the
    length to `max_input_len`.

    Args:
        df: Dataframe to bucketize
        attributes: to keep
        bucket_size: size of the buckets
    """

    # Retrieve number of elements
    num_elements = df.count()
    num_buckets = (num_elements + bucket_size - 1) // bucket_size

    # Construct df to bucketize
    spark_session: SparkSession = SparkSession.builder.getOrCreate()

    # Bucketize df
    # NOTE: Alternatively can use .repartition? 
    buckets = spark_session.createDataFrame(
        data=[(bucket, bucket * bucket_size, (bucket + 1) * bucket_size) for bucket in range(num_buckets)],
        schema=["bucket", "min_range", "max_range"],
    )

    # Order and bucketize elements
    return (
        df.withColumn("row_num", F.row_number().over(Window.orderBy("id")) - F.lit(1))
        .join(buckets, on=[(F.col("row_num") >= (F.col("min_range"))) & (F.col("row_num") < F.col("max_range"))])
        # Concat input
        .withColumn(
            "text_to_embed",
            F.concat(*[F.coalesce(F.col(feature), F.lit("")) for feature in input_features]),
        )
        # Clip max. length
        .withColumn("text_to_embed", F.substring(F.col("text_to_embed"), 1, max_input_len))
        .select("id", *input_features, "text_to_embed", "bucket")
    )
```

We're now using the native Spark paritioning column to ensure the dataset is written in a partitioned manner.

```yaml
embeddings.feat.bucketized_nodes@spark:
  <<: *_spark_parquet
  filepath: ${globals:paths.tmp}/feat/bucketized_nodes
  save_args:
    mode: overwrite
    partitionBy:
      - bucket
```

### Loading the input as a partitioned DF

Next, we wish to process the dataframe in shards, we will be using Kedro's `PartitionedDataset`. The partitioned dataset is interesting in the sense that it does not _load_ any data, but rather provides a dictionary as input to the node, mapping the paths of the dataset to it's shards' load function. It's important to remember that the data is _only_ loaded whenever the load function is invoked.

> Note: to allow reading/loading the same underlying dataset, in different format, in Kedro, we're using transcoding. This allows for re-defining
> the way the dataset to be loaded, while ensuring the Kedro dag does not become disconnected.


```yaml
embeddings.feat.bucketized_nodes@partitioned:
  type: matrix.datasets.gcp.PartitionedAsyncParallelDataset 
  path: ${globals:paths.tmp}/feat/bucketized_nodes
  dataset: 
    # NOTE: Switching between spark/pandas thanks to underlying parquet structure
    <<: *_pandas_parquet
  filename_suffix: ".parquet"
```

### Setting up the processing logic

Next, let's set up the processing logic, we're codifying this as function that process an individual dataframe.


```yaml
# Configuration for node embeddings
embeddings.node:
  # Following configuration ensures that OpenAI requests are
  # batches in batches of 500 input elements, where each input
  # element is clipped at 100 characters. Max. token limit for 
  # embeddings call is roughly 8500 tokens, approx (100 * 500) / 6.
  batch_size: 500
  max_input_len: 100
  input_features: ["category", "name"]

  model:
    object: langchain_openai.OpenAIEmbeddings
    model: text-embedding-3-small
    openai_api_key: ${oc.env:OPENAI_API_KEY}
    dimensions: 100
    timeout: 10
```

```python
# NOTE: Should we call out already that native dataset does not support async?
@retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(3))
async def compute_df_embeddings_async(df: pd.DataFrame, embedding_model) -> pd.DataFrame:
    try:
        # Embed entities in batch mode
        combined_texts = df["text_to_embed"].tolist()
        df["embedding"] = await embedding_model.aembed_documents(combined_texts)
    except Exception as e:
        print(f"Exception occurred: {e}")
        raise e

    # Drop added column
    df = df.drop(columns=["text_to_embed"])
    return df
```

### Connecting the dots

We've now setup the a Kedro dataset to load the hive partitioned dataset as a Kedro Partitioned dataset, and we wish to produce a new Kedro PartitionedDataset that includes the embedding.

```yaml
embeddings.feat.graph.node_embeddings@partitioned:
  type: matrix.datasets.gcp.PartitionedAsyncParallelDataset 
  overwrite: True # important otherwise not properly reset on rerun
  path: ${globals:paths.tmp}/feat/tmp_nodes_with_embeddings
  dataset: 
    <<: *_pandas_parquet
  filename_suffix: ".parquet"
```

Saving to a PartitionedDatset follows the same idea, where we output a dictionary of paths mapped to their `save()` function from the Kedro node. The clue is now to setup this dictionary in a a manner that `save()` function is responsible for invoking the `load()` and `compute_df_embeddings_async()` function.


```python
@inject_object()
def compute_embeddings(
    dfs: Dict[str, Any],
    model: Dict[str, Any],
):
    """Function to bucketize input data.

    Args:
        dfs: mapping of paths to df load functions
        model: model to run
    """
    def _func(dataframe: pd.DataFrame):
        # NOTE: Very important to bake in the df=dataframe to avoid reference issues
        return lambda df=dataframe: compute_df_embeddings_async(df(), model)

    shards = {}
    for path, df in dfs.items():
        # Little bit hacky, but extracting batch from hive partitioning for input path
        # As we know the input paths to this dataset are of the format /shard={num}
        bucket = path.split("/")[0].split("=")[1]

        # Invoke function to compute embeddings
        shard_path = f"bucket={bucket}/shard"
        shards[shard_path] = _func(df)

    return shards
```

### Parallelising the PartitionedDataset

Kedro's default dataset does not support parallelizing writing the individual shards, hence why I've extended it's `save()` behaviour to execute on the async event loop. A `Semaphore` was used to limit the maximum number of threads running at a given time, thereby allowing it to work with APIs that have rate limiting enabled.


```python
class PartitionedAsyncParallelDataset(PartitionedDataset):
    """
    Custom implementation of the ParallelDataset that allows concurrent processing.
    """
    def _save(self, data: dict[str, Any], max_workers: int = 10, timeout: int = 60) -> None:
        if self._overwrite and self._filesystem.exists(self._normalized_path):
            self._filesystem.rm(self._normalized_path, recursive=True)

        # Helper function to process a single partition
        async def process_partition(sem, partition_id, partition_data):
            async with sem:
                try:
                    # Set up arguments and path
                    kwargs = deepcopy(self._dataset_config)
                    partition = self._partition_to_path(partition_id)
                    kwargs[self._filepath_arg] = self._join_protocol(partition)
                    dataset = self._dataset_type(**kwargs)  # type: ignore

                    # Evaluate partition data if it's callable
                    if callable(partition_data):
                        partition_data = await partition_data()  # noqa: PLW2901
                    else:
                        raise RuntimeError("not callable")

                    # Save the partition data
                    dataset.save(partition_data)
                except Exception as e:
                    print(f"Error in process_partition with partition {partition_id}: {e}")
                    raise

        # Define function to run asyncio tasks within a synchronous function
        def run_async_tasks():
            # Create an event loop and a thread pool executor for async execution
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            sem = asyncio.Semaphore(max_workers)

            tasks = [
                loop.create_task(process_partition(sem, partition_id, partition_data))
                for partition_id, partition_data in sorted(data.items())
            ]

            # Track progress with tqdm as tasks complete
            with tqdm(total=len(tasks), desc="Saving partitions") as progress_bar:

                async def monitor_tasks():
                    for task in asyncio.as_completed(tasks):
                        try:
                            await asyncio.wait_for(task, timeout)
                        except asyncio.TimeoutError as e:
                            print(f"Timeout error: partition processing took longer than {timeout} seconds.")
                            raise e
                        except Exception as e:
                            print(f"Error processing partition in tqdm loop: {e}")
                            raise e
                        finally:
                            progress_bar.update(1)

                # Run the monitoring coroutine
                try:
                    loop.run_until_complete(monitor_tasks())
                finally:
                    loop.close()

        run_async_tasks()
        self._invalidate_caches()
```

### Wrapping up

We now have an output dataset with hive partitioning that contains the result, we can now use the `SparkDataset` to load it as a full table, i.e., promote the directory to a column for downstream processing.

```yaml
embeddings.feat.graph.node_embeddings@spark:
  <<: *_spark_parquet
  filepath: ${globals:paths.tmp}/feat/tmp_nodes_with_embeddings
```


The final Kedro pipeline looks as follows:


```python
def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=nodes.bucketize_df,
                inputs={
                    "df": "filtering.prm.filtered_nodes",
                    "input_features": "params:embeddings.node.input_features",
                    "bucket_size": "params:embeddings.node.batch_size",
                    "max_input_len": "params:embeddings.node.max_input_len",
                },
                outputs="embeddings.feat.bucketized_nodes@spark",
                name="bucketize_nodes",
                tags=["argowf.fuse", "argowf.fuse-group.node_embeddings"],
            ),
            # Compute embeddings
            node(
                func=nodes.compute_embeddings,
                inputs={
                    "dfs": "embeddings.feat.bucketized_nodes@partitioned",
                    "model": "params:embeddings.node.model",
                },
                outputs="embeddings.feat.graph.node_embeddings@partitioned",
                name="add_node_embeddings",
                tags=["argowf.fuse", "argowf.fuse-group.node_embeddings"],
            )
            # Next step inputs `embeddings.feat.graph.node_embeddings@spark` to ensure
            # table is loaded as a full dataset.
        ]
    )
```

[Caching :material-skip-next:](./caching.md){ .md-button .md-button--primary }