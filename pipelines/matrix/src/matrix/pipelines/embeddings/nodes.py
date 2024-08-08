"""Nodes for embeddings pipeline."""
import os
from typing import List, Any, Dict


from neo4j import Driver
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from pyspark.ml.functions import array_to_vector, vector_to_array
from graphdatascience import GraphDataScience, QueryRunner
from neo4j import GraphDatabase
from pypher import __ as cypher, Pypher

from pypher.builder import create_function
from . import pypher_utils

from refit.v1.core.inject import inject_object
from refit.v1.core.unpack import unpack_params

import logging

logger = logging.getLogger(__name__)


class GraphDB:
    """Adaptor class to allow injecting the GraphDB object.

    This is due to a drawback where refit cannot inject a tuple into
    the constructor of an object.
    """

    def __init__(
        self,
        *,
        endpoint: str | Driver | QueryRunner,
        auth: F.Tuple[str] | None = None,
        database: str | None = None,
    ):
        """Create `GraphDB` instance."""
        self._endpoint = endpoint
        self._auth = tuple(auth)
        self._database = database

    def driver(self):
        """Return the driver object."""
        return GraphDatabase.driver(self._endpoint, auth=self._auth)


class GraphDS(GraphDataScience):
    """Adaptor class to allow injecting the GDS object.

    This is due to a drawback where refit cannot inject a tuple into
    the constructor of an object.
    """

    def __init__(
        self,
        *,
        endpoint: str | Driver | QueryRunner,
        auth: F.Tuple[str] | None = None,
        database: str | None = None,
    ):
        """Create `GraphDS` instance."""
        super().__init__(endpoint, auth=tuple(auth), database=database)

        self.set_database(database)


@unpack_params()
@inject_object()
def compute_embeddings(
    input: DataFrame,
    gdb: GraphDB,
    features: List[str],
    api_key: str,
    batch_size: int,
    attribute: str,
    endpoint: str,
    model: str,
    concurrency: int,
):
    """Function to orchestrate embedding computation in Neo4j.

    Args:
        input: input df
        gdb: graph database instance
        features: features to include to compute embeddings
        api_key: api key to use
        batch_size: batch size
        attribute: attribute to add
        endpoint: endpoint to use
        model: model to use
        concurrency: number of concurrent calls to execute
    """
    # Due to https://github.com/neo4j-contrib/neo4j-apoc-procedures/issues/4156
    # we first check if we need to do anything here, and if all embeddings are already calculated
    # we do not do anything
    with gdb.driver() as driver:
        q = driver.execute_query(
            "match (n:Entity) where n.embedding is null return count(*) as count",
            database_=gdb._database,
        )
        rec = q.records[0]
        count = rec.get("count")
        if count == 0:
            # we don't have to embed anything anymore, thus skipping the work below
            logger.warning(
                "we actually have embedded everything already or there is an issue with the data. Continuing without taking action"
            )
            return {"success": "true"}
        else:
            logger.warning("we still have %s embeddings left to calculate", str(count))

    # fmt: off
    # Register functions
    create_function("iterate", {"name": "apoc.periodic.iterate"}, func_raw=True)
    create_function("openai_embedding", {"name": "apoc.ml.openai.embedding"}, func_raw=True)
    create_function("set_property", {"name": "apoc.create.setProperty"}, func_raw=True)

    # Build query
    p = Pypher()

    # Due to f-string limitations
    empty = '\"\"'

    # The apoc iterate is a rather interesting function, that takes stringified
    # cypher queries as input. The first determines the subset of nodes on
    # include, whereas the second query defines the operation to execute.
    # https://neo4j.com/labs/apoc/4.1/overview/apoc.periodic/apoc.periodic.iterate/
    p.CALL.iterate(
        # Match every :Entity node in the graph
        # FUTURE 'embedding' is hard coded because we had an issue with the $attribute inside of the `` brackets
        cypher.stringify(cypher.MATCH.node("p", labels="Entity").WHERE.p.property("embedding").IS_NULL.RETURN.p),
        # For each batch, execute following statements, the $_batch is a special
        # variable made accessible to access the elements in the batch.
        cypher.stringify(
            [
                # Apply OpenAI embedding in a batched manner, embedding
                # is applied on the concatenation of supplied features for each node.
                cypher.CALL.openai_embedding(f"[item in $_batch | {'+'.join(f'coalesce(item.p.{item}, {empty})' for item in features)}]", "$apiKey", "{endpoint: $endpoint, model: $model}").YIELD("index", "text", "embedding"),
                # Set the attribute property of the node to the embedding
                cypher.CALL.set_property("$_batch[index].p", "$attribute", "embedding").YIELD("node").RETURN("node"),
            ]
        ),
        # The last argument bridges the variables used in the outer query
        # and the variables referenced in the stringified params.
        cypher.map(
            batchMode="BATCH_SINGLE",
            # FUTURE when this is fixed: https://github.com/neo4j-contrib/neo4j-apoc-procedures/issues/4153 we should be able to max out
            # our capacity towards the service provider
            parallel="false",
            # parallel="false",
            batchSize=batch_size,
            concurrency=concurrency,
            params=cypher.map(apiKey=api_key, endpoint=endpoint, attribute=attribute, model=model),
        ),
    ).YIELD("batch", "operations").UNWIND("batch").AS("b").WITH("b").WHERE("b.failed > 0").RETURN("b.failed")
    # fmt: on

    failed = []
    with gdb.driver() as driver:
        failed = driver.execute_query(str(p), database_=gdb._database, **p.bound_params)

    if len(failed.records) > 0:
        raise RuntimeError("Failed batches in the embedding step")

    return {"success": "true"}


@unpack_params()
@inject_object()
def reduce_dimension(df: DataFrame, transformer, input: str, output: str, skip: bool):
    """Function to apply dimensionality reduction.

    Args:
        df: to apply technique to
        transformer: transformer to apply
        input: name of attribute to transform
        output: name of attribute to store result
        skip: whether to skip the PCA transformation and dimensionality reduction
    Returns:
        DataFrame: A DataFrame with either the reduced dimension embeddings or the original
                   embeddings, depending on the 'skip' parameter.

    Note: 
    - If skip is true, the function returns the original embeddings from the LLM model. 
    - If skip is false, the function returns the embeddings after applying the dimensionality reduction technique.
    
    """
    if skip:
        return df.withColumn(output, F.col(input))

    # Convert into correct type
    df = df.withColumn("features", array_to_vector(input))

    # Link
    transformer.setInputCol("features")
    transformer.setOutputCol("pca_features")

    return (
        transformer.fit(df)
        .transform(df)
        .withColumn(output, vector_to_array("pca_features"))
        .drop("pca_features", "features")
    )


@unpack_params()
@inject_object()
def train_topological_embeddings(
    df: DataFrame,
    edges: DataFrame,
    gds: GraphDataScience,
    projection: Any,
    filtering: Any,
    estimator: Any,
    write_property: str,
) -> Dict:
    """Function to add graphsage embeddings.

    Function leverages the gds library to ochestrate topological embedding computation
    on the nodes of the KG.

    NOTE: The df and edges input are only added to ensure correct lineage

    Args:
        df: nodes df
        edges: edges df
        gds: the gds object
        projection: gds projection to execute on the graph
        filtering: filtering to be applied to the projection
        estimator: estimator to apply
        write_property: node property to write result to
    """
    # Validate whether the GDS graph exists
    graph_name = projection.get("graphName")
    if gds.graph.exists(graph_name).exists:
        graph = gds.graph.get(graph_name)
        gds.graph.drop(graph, False)

    config = projection.pop("configuration", None)
    graph, _ = gds.graph.project(*projection.values(), **config)

    # Filter out treat/GT nodes from the graph
    subgraph_name = filtering.get("graphName")
    filter_args = filtering.pop("args")
    subgraph, _ = gds.graph.filter(subgraph_name, graph, **filter_args)

    # Validate whether the model exists
    model_name = estimator.get("args").get("modelName")
    if gds.model.exists(model_name).exists:
        model = gds.model.get(model_name)
        gds.model.drop(model)

    # Initialize the model
    model, _ = getattr(gds.beta, estimator.get("model")).train(
        subgraph, **estimator.get("args")
    )

    return {"success": "true"}


@inject_object()
@unpack_params()
def write_topological_embeddings(
    model: DataFrame,
    gds: GraphDataScience,
    projection: Any,
    estimator: Any,
    filtering: Any,
    write_property: str,
) -> Dict:
    """Write topological embeddings."""
    # Retrieve the graph
    graph_name = filtering.get("graphName")
    graph = gds.graph.get(graph_name)

    # Retrieve the model
    model_name = estimator.get("args").get("modelName")
    model = gds.model.get(model_name)

    # Write model output back to graph
    model.predict_write(graph, writeProperty=write_property)

    return {"success": "true"}


def extract_nodes_edges(
    nodes: DataFrame, edges: DataFrame
) -> tuple[DataFrame, DataFrame]:
    """Simple node/edge extractor function.

    Args:
        nodes: the nodes from the KG
        edges: the edges from the KG
    """
    return {"enriched_nodes": nodes, "enriched_edges": edges}
