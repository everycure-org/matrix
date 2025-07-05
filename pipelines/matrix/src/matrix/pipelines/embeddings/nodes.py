from typing import Any, Dict, List, Sequence

import matplotlib.pyplot as plt
import pandas as pd
import pyspark.sql as ps
import seaborn as sns
from graphdatascience import GraphDataScience
from pyspark.ml.functions import array_to_vector, vector_to_array
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, FloatType, StringType

from matrix.inject import inject_object, unpack_params
from matrix.utils.pandera_utils import Column, DataFrameSchema, check_output

from .graph_algorithms import GDSGraphAlgorithm


class GraphDS(GraphDataScience):
    """Adaptor class to allow injecting the GDS object.

    This is due to a drawback where our functions cannot inject a tuple into
    the constructor of an object.
    """

    def __init__(
        self,
        *,
        endpoint: str,
        auth: ps.functions.Tuple[str] | None = None,
        database: str | None = None,
    ):
        """Create `GraphDS` instance."""
        super().__init__(endpoint, auth=tuple(auth), database=database)

        self.set_database(database)


@check_output(
    schema=DataFrameSchema(
        columns={
            "id": Column(StringType(), nullable=False),
            "label": Column(StringType(), nullable=False),
            "name": Column(StringType(), nullable=True),
            "property_keys": Column(ArrayType(StringType()), nullable=False),
            "property_values": Column(ArrayType(StringType()), nullable=False),
            "upstream_data_source": Column(ArrayType(StringType()), nullable=False),
        }
    )
)
def ingest_nodes(df: ps.DataFrame) -> ps.DataFrame:
    """Function to create Neo4J nodes.

    Args:
        df: Nodes dataframe
    """
    return (
        df.select("id", "name", "category", "description", "upstream_data_source")
        .withColumn("label", ps.functions.col("category"))
        # add string properties here
        .withColumn(
            "properties",
            ps.functions.create_map(
                ps.functions.lit("name"),
                ps.functions.col("name"),
                ps.functions.lit("category"),
                ps.functions.col("category"),
                ps.functions.lit("description"),
                ps.functions.col("description"),
            ),
        )
        .withColumn("property_keys", ps.functions.map_keys(ps.functions.col("properties")))
        .withColumn("property_values", ps.functions.map_values(ps.functions.col("properties")))
        # add array properties here
        .withColumn(
            "array_properties",
            ps.functions.create_map(
                ps.functions.lit("upstream_data_source"),
                ps.functions.col("upstream_data_source"),
            ),
        )
        .withColumn("array_property_keys", ps.functions.map_keys(ps.functions.col("array_properties")))
        .withColumn("array_property_values", ps.functions.map_values(ps.functions.col("array_properties")))
    )


@unpack_params()
@check_output(
    schema=DataFrameSchema(
        columns={
            "id": Column(StringType(), nullable=False),
            "embedding": Column(ArrayType(FloatType()), nullable=False),
            "pca_embedding": Column(ArrayType(FloatType()), nullable=False),
        },
        unique=["id"],
    )
)
def reduce_embeddings_dimension(df: ps.DataFrame, transformer, input: str, output: str, skip: bool) -> ps.DataFrame:
    return reduce_dimension(df, transformer, input, output, skip)


@unpack_params()
@inject_object()
def reduce_dimension(df: ps.DataFrame, transformer, input: str, output: str, skip: bool) -> ps.DataFrame:
    """Function to apply dimensionality reduction.

    Function to apply dimensionality reduction conditionally, if skip is set to true
    the original input will be returned, otherwise the given transformer will be applied.

    Args:
        df: to apply technique to
        transformer: transformer to apply
        input: name of attribute to transform
        output: name of attribute to store result
        skip: whether to skip the PCA transformation and dimensionality reduction

    Returns:
        DataFrame: A DataFrame with either the reduced dimension embeddings or the original
                   embeddings, depending on the 'skip' parameter.
    """
    if skip:
        return df.withColumn(output, ps.functions.col(input).cast("array<float>"))

    # Convert into correct type
    df = df.withColumn("features", array_to_vector(ps.functions.col(input).cast("array<float>")))

    # Link
    transformer.setInputCol("features")
    transformer.setOutputCol("pca_features")

    res = (
        transformer.fit(df)
        .transform(df)
        .withColumn(output, vector_to_array("pca_features"))
        .withColumn(output, ps.functions.col(output).cast("array<float>"))
        .withColumn("pca_embedding", ps.functions.col(output))
        .drop("pca_features", "features")
    )

    return res


def filter_edges_for_topological_embeddings(
    nodes: ps.DataFrame, edges: ps.DataFrame, drug_types: List[str], disease_types: List[str]
) -> ps.DataFrame:
    """Function to filter edges for topological embeddings process.

    The function removes edges connecting drug and disease nodes to avoid data leakage. Currently
    uses the `all_categories` to remove drug-disease edges.

    FUTURE: Ensure edges from ground truth dataset are explicitly removed.

    Args:
        nodes: nodes dataframe
        edges: edges dataframe
        drug_types: list of drug types
        disease_types: list of disease types
    Returns:
        Dataframe with filtered edges
    """

    def _create_mapping(column: str):
        return nodes.alias(column).withColumn(column, ps.functions.col("id")).select(column, "all_categories")

    df = (
        edges.alias("edges")
        .join(_create_mapping("subject"), how="left", on="subject")
        .join(_create_mapping("object"), how="left", on="object")
        # FUTURE: Improve with proper feature engineering engine
        .withColumn(
            "subject_is_drug",
            ps.functions.arrays_overlap(ps.functions.col("subject.all_categories"), ps.functions.lit(drug_types)),
        )
        .withColumn(
            "subject_is_disease",
            ps.functions.arrays_overlap(ps.functions.col("subject.all_categories"), ps.functions.lit(disease_types)),
        )
        .withColumn(
            "object_is_drug",
            ps.functions.arrays_overlap(ps.functions.col("object.all_categories"), ps.functions.lit(drug_types)),
        )
        .withColumn(
            "object_is_disease",
            ps.functions.arrays_overlap(ps.functions.col("object.all_categories"), ps.functions.lit(disease_types)),
        )
        .withColumn(
            "is_drug_disease_edge",
            (ps.functions.col("subject_is_drug") & ps.functions.col("object_is_disease"))
            | (ps.functions.col("subject_is_disease") & ps.functions.col("object_is_drug")),
        )
        .filter(~ps.functions.col("is_drug_disease_edge"))
        .select("edges.*")
    )

    return df


def ingest_edges(nodes: ps.DataFrame, edges: ps.DataFrame) -> ps.DataFrame:
    """Function to construct Neo4J edges."""
    return (
        edges.select(
            "subject",
            "predicate",
            "object",
            "upstream_data_source",
        )
        .withColumn(
            "label",
            ps.functions.when(
                ps.functions.col("predicate").contains(":"),
                ps.functions.split(ps.functions.col("predicate"), ":", limit=2).getItem(1),
            ).otherwise(ps.functions.col("predicate")),
        )
        # we repartition to 1 partition here to avoid deadlocks in the edges insertion of neo4j.
        # FUTURE potentially we should repartition in the future to avoid deadlocks. However
        # with edges, this is harder to do than with nodes (as they are distinct but edges have 2 nodes)
        # https://neo4j.com/docs/spark/current/performance/tuning/#parallelism
        .repartition(1)
    )


@unpack_params()
@inject_object()
def train_topological_embeddings(
    df: ps.DataFrame,
    gds: GraphDataScience,
    topological_estimator: GDSGraphAlgorithm,
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
        gds: the gds object
        filtering: filtering
        projection: gds projection to execute on the graph
        topological_estimator: GDS estimator to apply
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

    # Validate whether the model exists
    model_name = estimator.get("modelName")
    if gds.model.exists(model_name).exists:
        model = gds.model.get(model_name)
        gds.model.drop(model)

    # Initialize the model
    topological_estimator.run(gds=gds, model_name=model_name, graph=graph, write_property=write_property)
    losses = topological_estimator.return_loss()

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
    model: ps.DataFrame,
    gds: GraphDataScience,
    topological_estimator: GDSGraphAlgorithm,
    projection: Any,
    estimator: Any,
    write_property: str,
) -> Dict:
    """Write topological embeddings."""
    # Retrieve the graph
    graph_name = projection.get("graphName")
    graph = gds.graph.get(graph_name)

    # Retrieve the model
    model_name = estimator.get("modelName")
    topological_estimator.predict_write(gds=gds, model_name=model_name, graph=graph, write_property=write_property)
    return {"success": "true"}


def _cast_to_array(df, col: str) -> ps.DataFrame:
    if isinstance(df.schema[col].dataType, ps.types.StringType):
        return df.withColumn(
            col, ps.functions.from_json(ps.functions.col(col), ps.types.ArrayType(ps.types.FloatType()))
        )

    return df


@check_output(
    schema=DataFrameSchema(
        columns={
            "id": Column(StringType(), nullable=False),
            "topological_embedding": Column(ArrayType(FloatType()), nullable=False),
            "pca_embedding": Column(ArrayType(FloatType()), nullable=False),
        },
        unique=["id"],
    )
)
def extract_topological_embeddings(embeddings: ps.DataFrame, nodes: ps.DataFrame, string_col: str) -> ps.DataFrame:
    """Extract topological embeddings from Neo4j and write into BQ.

    Need a conditional statement due to Node2Vec writing topological embeddings as string. Raised issue in GDS client:
    https://github.com/neo4j/graph-data-science-client/issues/742#issuecomment-2324737372.
    """
    x = (
        nodes.alias("nodes")
        .join(embeddings.transform(_cast_to_array, string_col).alias("embeddings"), on="id", how="left")
        .select("nodes.*", "embeddings.pca_embedding", "embeddings.topological_embedding")
        .withColumn("pca_embedding", ps.functions.col("pca_embedding").cast("array<float>"))
        .withColumn("topological_embedding", ps.functions.col("topological_embedding").cast("array<float>"))
    )

    return x


def visualise_pca(nodes: ps.DataFrame, column_name: str) -> plt.Figure:
    """Write topological embeddings."""
    nodes = nodes.select(column_name, "category").toPandas()
    nodes[["pca_0", "pca_1"]] = pd.DataFrame(nodes[column_name].tolist(), index=nodes.index)
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
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0, fontsize="small")
    plt.tight_layout(rect=[0, 0, 0.85, 1])

    return fig


def embeddings_preprocessor(df: DataFrame, key_length: int, combine_cols: Sequence[str], new_col: str) -> DataFrame:
    return df.withColumn(new_col, F.concat_ws("", *combine_cols).substr(startPos=1, length=key_length))
