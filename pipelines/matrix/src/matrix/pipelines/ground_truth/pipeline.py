from kedro.pipeline import Pipeline, node, pipeline

from matrix import settings
from matrix.pipelines.batch import pipeline as batch_pipeline
from matrix.pipelines.integration import nodes as integration_nodes

from ...kedro4argo_node import ArgoNode, ArgoResourceConfig
from . import nodes


def _create_ground_truth_pipeline(
    source: str, is_combined: bool = True, has_positives: bool = True, has_negatives: bool = True
) -> Pipeline:
    pipelines = []

    pipelines.append(
        pipeline(
            [
                ArgoNode(
                    func=nodes.transform,
                    inputs={
                        "transformer": f"params:ground_truth.sources.{source}.transformer",
                        # NOTE: This dynamically wires the nodes and edges into each transformer.
                        # This is due to the fact that the Transformer objects are only created
                        # during node execution time, otherwise we could infer this based on
                        # the transformer.
                        **(
                            {"positive_edges": f"ingestion.int.{source}.edges.positives@spark"} if has_positives else {}
                        ),
                        **(
                            {"negative_edges": f"ingestion.int.{source}.edges.negatives@spark"} if has_negatives else {}
                        ),
                        **({"edges_df": f"ingestion.int.{source}.edges@spark"} if is_combined else {}),
                    },
                    outputs={"edges": f"ground_truth.int.{source}.edges", "nodes": f"ground_truth.int.{source}.nodes"},
                    name=f"transform_{source}_nodes",
                    tags=["standardize"],
                    argo_config=ArgoResourceConfig(
                        memory_request=128,
                        memory_limit=128,
                    ),
                ),
                batch_pipeline.create_pipeline(
                    source=f"source_{source}",
                    df=f"ground_truth.int.{source}.nodes",
                    output=f"ground_truth.int.{source}.nodes.nodes_norm_mapping",
                    bucket_size="params:integration.normalization.batch_size",  # Ensure same normalization settings are used
                    transformer="params:integration.normalization.normalizer",
                    max_workers=120,
                ),
                node(
                    func=integration_nodes.normalize_edges,
                    inputs={
                        "mapping_df": f"ground_truth.int.{source}.nodes.nodes_norm_mapping",
                        "edges": f"ground_truth.int.{source}.edges",
                    },
                    outputs=f"ground_truth.int.{source}.edges.norm@spark",
                    name=f"normalize_{source}_ground_truth_edges",
                ),
            ],
            tags=source,
        )
    )
    return sum(pipelines)


def create_pipeline(**kwargs) -> Pipeline:
    """Create ground truth pipeline."""

    pipelines = []

    # Create pipeline per source
    for source in settings.DYNAMIC_PIPELINES_MAPPING.get("ground_truth"):
        pipelines.append(
            pipeline(
                _create_ground_truth_pipeline(
                    source=source["name"],
                    is_combined=source["is_combined"],
                    has_positives=source["has_positives"],
                    has_negatives=source["has_negatives"],
                ),
                tags=[source["name"]],
            )
        )
    pipelines.append(
        pipeline(
            [
                node(
                    func=nodes.unify_edges,
                    inputs=[
                        *[
                            f'ground_truth.int.{source["name"]}.edges.norm@spark'
                            for source in settings.DYNAMIC_PIPELINES_MAPPING.get("ground_truth")
                        ],
                    ],
                    outputs="ground_truth.prm.unified_edges@spark",
                    name="unify_ground_truth_datasets",
                ),
            ]
        )
    )

    return sum(pipelines)
