"""Integration pipeline."""
from typing import List
from datetime import datetime

from kedro.pipeline import Pipeline, node, pipeline
from pyspark.sql import DataFrame

from refit.v1.core.inline_has_schema import has_schema

import pyspark.sql.functions as F

from pypher import __, Pypher, create_function


@has_schema(
    schema={
        "id": "string",
    },
    allow_subset=True,
    relax=True,
)
def extract_nodes(raw_nodes: DataFrame, types: List[str]) -> DataFrame:
    """Function to extract nodes with given type.

    Args:
        raw_nodes: Raw nodes DataFrame.
        types: List of types to filter.
    """
    return raw_nodes.filter(F.col("category").isin(types))


@has_schema(
    schema={
        "source_id": "string",
        "target_id": "string",
    },
    allow_subset=True,
    relax=True,
)
def extract_edges(raw_edges: DataFrame) -> DataFrame:
    """Function to extract edges.

    Args:
        raw_edges: Raw edges DataFrame.
    """
    return raw_edges


def apply_date_filter(data: DataFrame, cutoff_date: datetime) -> None:
    """Function to create int embeddings.

    Args:
        data: Dataframe
        cutoff_date: Cutoff date
    """
    return data.filter(F.col("date_discovered") <= cutoff_date)


def print(data: DataFrame) -> None:
    """Function to print input DataFrame.

    Args:
        data: Dataframe
    """
    data.show()
    return data


def neo4j_drug_disease_query(pypher: Pypher) -> Pypher:
    """Function codifying the drug-disease query.

    Args:
        pypher: root Pypher object
    Returns:
        Pypher object
    """
    create_function("datetime", {"name": "datetime"})

    return (
        pypher.MATCH.node("drug", labels="Drug")
        .rel_out(labels="TREATS")
        .node("disease", "Disease")
        .RETURN(
            __.drug.__id__.ALIAS("drug_label"),
            __.disease.__id__.ALIAS("disease_label"),
            __.disease.__date_discovered__.ALIAS("datetime_str"),
            __.datetime(__.disease.__date_discovered__).ALIAS("datetime"),
        )
    )


def create_pipeline(**kwargs) -> Pipeline:
    """Create integration pipeline."""
    return pipeline(
        [
            node(
                func=extract_nodes,
                inputs=["integration.raw.rtx_kg2.nodes", "params:modelling.drug_types"],
                outputs="integration.prm.drugs",
                name="create_neo4j_drug_nodes",
            ),
            node(
                func=extract_nodes,
                inputs=[
                    "integration.raw.rtx_kg2.nodes",
                    "params:modelling.disease_types",
                ],
                outputs="integration.prm.diseases",
                name="create_neo4j_disease_nodes",
            ),
            node(
                func=extract_edges,
                inputs=["integration.raw.rtx_kg2.edges"],
                outputs="integration.prm.treats",
                name="create_neo4j_edges",
            ),
            # Example reading
            node(
                func=print,
                inputs=["integration.prm.pypher"],
                outputs=None,
                name="print",
            ),
        ]
    )
