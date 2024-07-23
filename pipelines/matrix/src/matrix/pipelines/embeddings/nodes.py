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
        super().__init__(
            endpoint,
            auth=tuple(auth),
        )

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
        cypher.stringify(cypher.MATCH.node("p", labels="Entity").WHERE.p.property('$attribute').IS_NULL.RETURN.p),
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
            parallel="true",
            batchSize=batch_size,
            concurrency=concurrency,
            params=cypher.map(apiKey=api_key, endpoint=endpoint, attribute=attribute, model=model),
        ),
    ).YIELD("batch", "operations")
    # fmt: on

    with gdb.driver() as driver:
        summary = driver.execute_query(str(p), **p.bound_params).summary

    return {"success": "true", "time": summary.result_available_after}


@unpack_params()
@inject_object()
def reduce_dimension(df: DataFrame, transformer, input: str, output: str):
    """Function to apply dimensionality reduction.

    Args:
        df: to apply technique to
        transformer: transformer to apply
        input: name of attribute to transform
        output: name of attribute to store result
    Returns:
        Dataframe with reduced dimension
    """
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


@inject_object()
@unpack_params()
def train_topological_embeddings(
    df: DataFrame,
    edges: DataFrame,
    gds: GraphDataScience,
    projection: Any,
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

    # Validate whether the model exists
    model_name = estimator.get("args").get("modelName")
    if gds.model.exists(model_name).exists:
        model = gds.model.get(model_name)
        gds.model.drop(model)

    # Initialize the model
    model, _ = getattr(gds.beta, estimator.get("model")).train(
        graph, **estimator.get("args")
    )

    return {"success": "true"}


@inject_object()
@unpack_params()
def write_topological_embeddings(
    model: DataFrame,
    gds: GraphDataScience,
    projection: Any,
    estimator: Any,
    write_property: str,
) -> Dict:
    """Write topological embeddings."""
    # Retrieve the graph
    graph_name = projection.get("graphName")
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
