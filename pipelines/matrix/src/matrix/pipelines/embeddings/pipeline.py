"""Embeddings pipeline."""
from typing import List, Any, Dict

from kedro.pipeline import Pipeline, node, pipeline

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from pyspark.ml.functions import array_to_vector, vector_to_array
from graphdatascience import GraphDataScience

from refit.v1.core.inject import inject_object
from refit.v1.core.unpack import unpack_params

# TODO: Extract this into params
gds = GraphDataScience("bolt://127.0.0.1:7687", auth=("neo4j", "admin"))
gds.set_database("everycure")


def concat_features(df: DataFrame, features: List[str], ai_config: Dict[str, str]):
    """Function setup features for node embeddings.

    Args:
        df: nodes dataframe
        features: features to use for node embeddings.
        ai_config: vertex confoiguration to use
    Returns:
        Input features for node computation
    """
    for key, value in ai_config.items():
        df = df.withColumn(key, F.lit(value))

    return df.withColumn("input", F.concat(*[F.col(feature) for feature in features]))


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
    df = df.withColumn("features", array_to_vector("embedding"))

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
    projection: Any,
    estimator: Any,
    write_property: str,
):
    """Function to add graphsage embeddings.

    Function leverages the gds library to ochestrate toplogical embedding computation
    on the nodes of the KG.

    NOTE: The df is only added to ensure correct lineage
    NOTE: Unfortunately, the gds library works

    Args:
        df: input
        projection: gds projection to execute on the graph
        estimator: estimator to apply
        write_property: node property to write result to
    """
    graph_name = projection.get("graphName")
    if gds.graph.exists(graph_name).exists:
        gds.graph.drop(graph_name, False)

    config = projection.pop("config")
    graph, _ = gds.graph.project(*projection.values(), **config)

    model_name = estimator.get("args").get("modelName")
    if gds.model.exists(model_name).exists:
        model = gds.model.get(model_name)
        gds.model.drop(model)

    model, _ = getattr(gds.beta, estimator.get("model")).train(
        graph, **estimator.get("args")
    )
    model.predict_write(graph, writeProperty=write_property)


def create_pipeline(**kwargs) -> Pipeline:
    """Create embeddings pipeline."""
    return pipeline(
        [
            # NOTE: This enriches the current graph with a nummeric property
            node(
                func=concat_features,
                inputs=[
                    "integration.model_input.nodes",
                    "params:embeddings.node.features",
                    "params:embeddings.ai_config",
                ],
                outputs="embeddings.prm.graph.embeddings",
                name="add_node_embeddings",
            ),
            node(
                func=reduce_dimension,
                inputs=[
                    "embeddings.prm.graph.embeddings",
                    "params:embeddings.dimensionality_reduction",
                ],
                outputs="embeddings.prm.graph.pca_embeddings",
                name="apply_pca",
            ),
            node(
                func=add_topological_embeddings,
                inputs={
                    "df": "embeddings.prm.graph.pca_embeddings",
                    "unpack": "params:embeddings.topological",
                },
                outputs=None,
                name="add_topological_embeddings",
            ),
        ]
    )
