"""
This is a boilerplate pipeline 'sample'
generated using Kedro 0.19.7
"""

import pyspark.sql as ps
from kedro.pipeline import Pipeline, pipeline

from matrix import settings
from matrix.inject import inject_object
from matrix.kedro4argo_node import argo_node
from matrix.pipelines.sample.nodes import sample_kg_from_ids, select_sample_ids

from . import nodes


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            argo_node(
                func=select_sample_ids,
                inputs={
                    "gt": "modelling.raw.ground_truth.positives@spark",
                    "drugs": "ingestion.raw.drug_list@spark",
                    "diseases": "ingestion.raw.disease_list@spark",
                    "params": "params:sample.gt_params",
                },
                outputs=f"sampling.ids",
                name=f"generate_core_sample_ids",
            ),
            argo_node(
                func=sample_kg_from_ids,
                inputs={
                    "node_ids": f"sampling.ids",
                    "nodes": f"integration.int.rtx_kg2.nodes.full_data",
                    "edges": f"integration.int.rtx_kg2.edges.full_data",
                    "params": "params:sample.kg_params",
                },
                outputs=[
                    "integration.int.rtx_kg2.nodes",
                    "integration.int.rtx_kg2.edges",
                ],
                name=f"sample_rtx_kg2",
            ),
        ]
    )
