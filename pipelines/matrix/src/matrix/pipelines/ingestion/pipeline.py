from kedro.pipeline import Pipeline, node, pipeline

import matrix.pipelines.ingestion.nodes as nodes
from matrix import settings
from matrix.kedro4argo_node import ArgoNode, ArgoResourceConfig
from matrix.utils.validation import validate

from . import nodes


def create_pipeline(**kwargs) -> Pipeline:
    """Create ingestion pipeline."""
    # Create pipeline per source
    nodes_lst = []

    # Drug list and disease list
    nodes_lst.extend(
        [
            node(
                func=nodes.write_drug_list,
                inputs=["ingestion.raw.drug_list"],
                outputs="ingestion.raw.drug_list.nodes@pandas",
                name="write_drug_list",
                tags=["drug-list"],
            ),
            node(
                func=nodes.write_disease_list,
                inputs=["ingestion.raw.disease_list"],
                outputs="ingestion.raw.disease_list.nodes@pandas",
                name="write_disease_list",
                tags=["disease-list"],
            ),
        ]
    )

    # RTX-KG2 curies
    nodes_lst.append(
        node(
            func=lambda x: x,
            inputs=["ingestion.raw.rtx_kg2.curie_to_pmids@spark"],
            outputs="ingestion.int.rtx_kg2.curie_to_pmids",
            name="write_rtx_kg2_curie_to_pmids",
            tags=["rtx_kg2"],
        )
    )

    # Add ingestion pipeline for each source
    for source in settings.DYNAMIC_PIPELINES_MAPPING().get("integration"):
        if source.get("has_nodes", True):
            if "robokop" in source.get("name", ""):
                nodes_lst.extend(
                    [
                        ArgoNode(
                            name="robokop_preprocessing_nodes",
                            func=nodes.preprocess_robokop_nodes,
                            inputs=f"ingestion.raw.{source['name']}.nodes@lazypolars",
                            outputs=f"ingestion.int.preprocessing.{source['name']}.nodes@polars",
                            tags=[f"{source['name']}"],
                            argo_config=ArgoResourceConfig(
                                memory_limit=256,
                                memory_request=192,
                            ),
                        ),
                        node(
                            func=lambda x: x,
                            inputs=f"ingestion.int.preprocessing.{source['name']}.nodes@spark",
                            outputs=f"ingestion.int.{source['name']}.nodes",
                            name=f"write_{source['name']}_nodes",
                            tags=[f"{source['name']}"]
                        ),
                    ]
                )
            else:
                nodes_lst.append(
                    node(
                        func=lambda x: x,
                        inputs=f"ingestion.raw.{source['name']}.nodes@spark",
                        outputs=f"ingestion.int.{source['name']}.nodes",
                        name=f"write_{source['name']}_nodes",
                        tags=[f"{source['name']}"],
                    )
                )
        if "ground_truth" in source.get("name", ""):
            nodes_lst.append(
                node(
                    func=lambda x, y: [x, y],
                    inputs=[
                        f"ingestion.raw.{source['name']}.positives",
                        f"ingestion.raw.{source['name']}.negatives",
                    ],
                    outputs=[
                        f"ingestion.int.{source['name']}.positive.edges@pandas",
                        f"ingestion.int.{source['name']}.negative.edges@pandas",
                    ],
                    name=f"write_{source['name']}",
                    tags=[f"{source['name']}"],
                )
            )
        elif source.get("has_edges", True):
            if "robokop" in source.get("name", ""):
                nodes_lst.extend(
                    [
                        ArgoNode(
                            name="robokop_preprocessing_edges",
                            func=nodes.preprocess_robokop_edges,
                            inputs=f"ingestion.raw.{source['name']}.edges@lazypolars",
                            outputs=f"ingestion.int.preprocessing.{source['name']}.edges@polars",
                            tags=[f"{source['name']}"],
                            argo_config=ArgoResourceConfig(
                                memory_limit=128,
                                memory_request=64,
                            ),
                        ),
                        node(
                            func=lambda x: x,
                            inputs=f"ingestion.int.preprocessing.{source['name']}.edges@spark",
                            outputs=f"ingestion.int.{source['name']}.edges",
                            name=f"write_{source['name']}_edges",
                            tags=[f"{source['name']}"]
                        ),
                    ]
                )
            else:
                nodes_lst.append(
                    node(
                        func=lambda x: x,
                        inputs=[f"ingestion.raw.{source['name']}.edges@spark"],
                        outputs=f"ingestion.int.{source['name']}.edges",
                        name=f"write_{source['name']}_edges",
                        tags=[f"{source['name']}"],
                    )
                )
        if source.get("validate", False):
            if "robokop" in source.get("name", ""):
                nodes_lst.append(
                    ArgoNode(
                        func=validate,
                        inputs={
                            "nodes": f"ingestion.int.preprocessing.{source['name']}.nodes@polars",
                            "edges": f"ingestion.int.preprocessing.{source['name']}.edges@polars",
                        },
                        outputs=f"ingestion.int.{source['name']}.violations",
                        name=f"validate_{source['name']}",
                        tags=[f"{source['name']}"],
                        argo_config=ArgoResourceConfig(
                            memory_limit=128,
                            memory_request=64,
                        ),
                    )
                )

            else:
                nodes_lst.append(
                    ArgoNode(
                        func=validate,
                        inputs={
                            "nodes": f"ingestion.raw.{source['name']}.nodes@polars",
                            "edges": f"ingestion.raw.{source['name']}.edges@polars",
                        },
                        outputs=f"ingestion.int.{source['name']}.violations",
                        name=f"validate_{source['name']}",
                        tags=[f"{source['name']}"],
                    )
                )
    return pipeline(nodes_lst)
