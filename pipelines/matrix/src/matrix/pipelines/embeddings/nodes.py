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
from pypher_utils import create_stringified_function

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
):
    """Function to orchestrate embedding computation in Neo4j.

    Args:
        input: input df
        gdb: graph database instance
        features: features to include to compute embeddings
        ai_config: genai config
    """
    # Register functions
    create_stringified_function("iterate", {"name": "apoc.periodic.iterate"})
    create_function(
        "openai_embedding", {"name": "apoc.ml.openai.embedding"}, func_raw=True
    )
    create_function("set_property", {"name": "apoc.create.setProperty"}, func_raw=True)

    # Build query
    p = Pypher()

    p.CALL.iterate(
        # Match every :Entity node in the graph
        cypher.MATCH.node("p", labels="Entity").RETURN.p,
        # For each batch, execute following statements
        [
            # Match OpenAI embedding in a batched manner
            cypher.CALL.openai_embedding(
                "[item in $_batch | item.p.category]",
                "$apiKey",
                "{endpoint: $endpoint, model: $model}",
            ).YIELD("index", "text", "embedding"),
            # Set the attribute property of the node to the embedding
            cypher.CALL.set_property("$_batch[index].p", "$attribute", "embedding")
            .YIELD("node")
            .RETURN("node"),
        ],
        cypher.map(
            batchMode="BATCH_SINGLE",
            batchSize=1000,
            params=cypher.map(
                apiKey="key", endpoint="endpoint", attribute="embedding", model="ada"
            ),
        ),
    ).YIELD("batch", "operations")

    breakpoint()

    with gdb.driver() as driver:
        driver.execute_query(str(p), **p.bound_params)

    return {"success": "true"}


def concat_features(nodes: DataFrame, features: List[str], ai_config: Dict[str, str]):
    """Function setup features for node embeddings.

    Args:
        nodes: nodes dataframe
        features: features to use for node embeddings.
        ai_config: vertex confoiguration to use
    Returns:
        Input features for node computation
    """
    for key, value in ai_config.items():
        nodes = nodes.withColumn(key, F.lit(value))

    return nodes.withColumn(
        "input", F.concat(*[F.col(feature) for feature in features])
    )


@inject_object()
def reduce_dimension(df: DataFrame, transformer):
    """Function to apply dimensionality reduction.

    Args:
        df: to apply technique to
        transformer: transformer to apply
    Returns:
        Dataframe with reduced dimension
    """
    # Convert into correct type
    df = df.filter(F.col("embedding").isNotNull()).withColumn(
        "features", array_to_vector("embedding")
    )

    # Link
    transformer.setInputCol("features")
    transformer.setOutputCol("pca_features")

    return (
        transformer.fit(df)
        .transform(df)
        .withColumn("pca_embedding", vector_to_array("pca_features"))
        .drop("pca_features", "features")
    )


@inject_object()
@unpack_params()
def add_topological_embeddings(
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

    config = projection.pop("config")
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

    # Write model output back to graph
    model.predict_write(graph, writeProperty=write_property)

    return {"success": "true"}
