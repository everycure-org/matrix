"""Nodes for embeddings pipeline."""
import os
from typing import List, Any, Dict

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from neo4j import Driver
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from pyspark.ml.functions import array_to_vector, vector_to_array
from graphdatascience import GraphDataScience, QueryRunner
from neo4j import GraphDatabase
from pypher import __ as cypher, Pypher

from matplotlib.pyplot import plot
import seaborn as sns

from pypher.builder import create_function
from . import pypher_utils

from refit.v1.core.inject import inject_object
from refit.v1.core.unpack import unpack_params
from refit.v1.core.inline_has_schema import has_schema
from refit.v1.core.inline_primary_key import primary_key

from tenacity import (
    retry,
    stop_after_attempt,
    retry_if_exception_type,
)

import matplotlib.pyplot as plt

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


@has_schema(
    schema={
        "label": "string",
        "id": "string",
        "name": "string",
        "property_keys": "array<string>",
        "property_values": "array<string>",
        "kg_sources": "array<string>",
    },
    allow_subset=True,
)
@primary_key(primary_key=["id"])
def ingest_nodes(df: DataFrame) -> DataFrame:
    """Function to create Neo4J nodes.

    Args:
        df: Nodes dataframe
    """
    return (
        df.select("id", "name", "category", "description", "kg_sources")
        .withColumn("label", F.split(F.col("category"), ":", limit=2).getItem(1))
        .withColumn(
            "properties",
            F.create_map(
                F.lit("name"),
                F.col("name"),
                F.lit("category"),
                F.col("category"),
                F.lit("description"),
                F.col("description"),
            ),
        )
        .withColumn("property_keys", F.map_keys(F.col("properties")))
        .withColumn("property_values", F.map_values(F.col("properties")))
    )


class FailedBatchesException(BaseException):
    """Exception to signal failed batches."""

    pass


@unpack_params()
@inject_object()
@retry(
    stop=stop_after_attempt(3),
    retry=retry_if_exception_type(FailedBatchesException),
)
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
            logger.info("we still have %s embeddings left to calculate", str(count))

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
        raise FailedBatchesException("Failed batches in the embedding step")

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
    print(df.show())
    df.show()
    df = df.withColumn("features", array_to_vector(input))

    # Link
    transformer.setInputCol("features")
    transformer.setOutputCol("pca_features")

    res = (
        transformer.fit(df)
        .transform(df)
        .withColumn(output, vector_to_array("pca_features"))
        .drop("pca_features", "features")
    )

    return res


def ingest_edges(nodes, edges: DataFrame):
    """Function to construct Neo4J edges."""
    return (
        edges.select(
            "subject", "predicate", "object", "knowledge_sources", "kg_sources"
        )
        .withColumn("label", F.split(F.col("predicate"), ":", limit=2).getItem(1))
        # we repartition to 1 partition here to avoid deadlocks in the edges insertion of neo4j.
        # FUTURE potentially we should repartition in the future to avoid deadlocks. However
        # with edges, this is harder to do than with nodes (as they are distinct but edges have 2 nodes)
        # https://neo4j.com/docs/spark/current/performance/tuning/#parallelism
        .repartition(1)
    )


@inject_object()
def add_include_in_graphsage(
    df: DataFrame, gdb: GraphDB, drug_types: List[str], disease_types: List[str]
) -> Dict:
    """Function to add include_in_graphsage property.

    Only edges between non drug-disease pairs are included in graphsage.
    """
    with gdb.driver() as driver:
        q = driver.execute_query(
            """
            MATCH (n)-[r]-(m)
            WHERE 
                n.category IN $drug_types 
                AND m.category IN $disease_types
            SET r.include_in_graphsage = 0
            """,
            database_=gdb._database,
            drug_types=drug_types,
            disease_types=disease_types,
        )

    return {"success": "true"}


@unpack_params()
@inject_object()
def train_topological_embeddings(
    df: DataFrame,
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
        gds: the gds object
        filtering: filtering
        projection: gds projection to execute on the graph
        estimator: estimator to apply
        write_property: node property to write result to
    """
    # Validate whether the GDS graph exists
    graph_name = projection.get("graphName")
    if gds.graph.exists(graph_name).exists:
        graph = gds.graph.get(graph_name)
        gds.graph.drop(graph, False)

    config = projection.pop("configuration", {})
    graph, _ = gds.graph.project(*projection.values(), **config)

    # Filter out treat/GT nodes from the graph
    subgraph_name = filtering.get("graphName")
    filter_args = filtering.pop("args")

    # Drop graph if exists
    if gds.graph.exists(subgraph_name).exists:
        subgraph = gds.graph.get(subgraph_name)
        gds.graph.drop(subgraph, False)

    subgraph, _ = gds.graph.filter(subgraph_name, graph, **filter_args)

    # Validate whether the model exists
    model_name = estimator.get("modelName")
    if gds.model.exists(model_name).exists:
        model = gds.model.get(model_name)
        gds.model.drop(model)

    # Initialize the model
    estimator_name = estimator.get("model")
    if estimator_name == "graphSage":
        model, attr = getattr(gds.beta, estimator.get("model")).train(
            subgraph, **estimator.get("graphsage_args")
        )
        losses = attr.modelInfo["metrics"]["iterationLossesPerEpoch"][0]
    elif estimator_name == "node2vec":
        attr = getattr(gds, estimator.get("model")).write(
            subgraph, **estimator.get("node2vec_args"), writeProperty=write_property
        )
        losses = [int(x) for x in attr["lossPerIteration"]]
    else:
        raise ValueError()

    # Plot convergence
    convergence = plt.figure()
    ax = convergence.add_subplot(1, 1, 1)
    ax.plot([x for x in range(len(losses))], losses)

    # Add labels and title
    ax.set_xlabel("Number of Epochs")
    ax.set_ylabel("Average loss per node")
    ax.set_title("Loss Chart")

    return {"success": "true"}, convergence


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
    graph_name = projection.get("graphName")
    graph = gds.graph.get(graph_name)

    # Retrieve the model
    model_name = estimator.get("modelName")
    if model_name == "graphSage":
        model = gds.model.get(model_name)
        # Write model output back to graph
        model.predict_write(graph, writeProperty=write_property)
    return {"success": "true"}


def visualise_pca(nodes: DataFrame, column_name: str):
    """Write topological embeddings."""
    nodes = nodes.select(column_name, "category").toPandas()
    nodes[["pca_0", "pca_1"]] = pd.DataFrame(
        nodes[column_name].tolist(), index=nodes.index
    )
    fig = plt.figure(
        figsize=(
            10,
            5,
        )
    )
    sns.scatterplot(data=nodes, x="pca_0", y="pca_1", hue="category")
    plt.suptitle("PCA scatterpot")
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.legend(
        bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0, fontsize="small"
    )
    plt.tight_layout(rect=[0, 0, 0.85, 1])

    return fig


def extract_nodes_edges(
    nodes: DataFrame, edges: DataFrame
) -> tuple[DataFrame, DataFrame]:
    """Simple node/edge extractor function.

    Args:
        nodes: the nodes from the KG
        edges: the edges from the KG
    """
    return {"enriched_nodes": nodes, "enriched_edges": edges}
