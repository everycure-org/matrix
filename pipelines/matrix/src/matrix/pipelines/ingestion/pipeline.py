from kedro.pipeline import Pipeline, node, pipeline

from matrix import settings

from . import nodes


def create_ground_truth_pipeline() -> list:
    """Create pipeline nodes for ground truth processing."""
    return [
        node(
            func=nodes.create_gt,
            inputs={
                "pos_df": "ingestion.raw.ground_truth.positives",
                "neg_df": "ingestion.raw.ground_truth.negatives",
            },
            outputs="ingestion.raw.ground_truth.edges@pandas",
            name="concatenate_gt_dataframe",
            tags=["ground-truth"],
        )
    ]


def create_valid_edge_type_pipeline() -> list:
    """Create pipeline nodes for valid edge type processing."""
    return [
        node(
            func=lambda x: x,
            inputs=["ingestion.raw.valid_edge_types"],
            outputs="ingestion.int.valid_edge_types",
            name="write_valid_edge_types",
            tags=["valid-edge-types"],
        )
    ]


def create_pipeline(**kwargs) -> Pipeline:
    """Create ingestion pipeline."""
    # Create pipeline per source
    nodes_lst = []

    # Ground truth
    nodes_lst.extend(create_ground_truth_pipeline())
    nodes_lst.extend(create_valid_edge_type_pipeline())

    # Drug list and disease list
    nodes_lst.extend(
        [
            node(
                func=lambda x: x,
                inputs=["ingestion.raw.drug_list"],
                outputs="ingestion.raw.drug_list.nodes@pandas",
                name="write_drug_list",
                tags=["drug-list"],
            ),
            node(
                func=lambda x: x,
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
            nodes_lst.append(
                node(
                    func=lambda x: x,
                    inputs=[f'ingestion.raw.{source["name"]}.nodes@spark'],
                    outputs=f'ingestion.int.{source["name"]}.nodes',
                    name=f'write_{source["name"]}_nodes',
                    tags=[f'{source["name"]}'],
                )
            )

        if source.get("has_edges", True):
            nodes_lst.append(
                node(
                    func=lambda x: x,
                    inputs=[f'ingestion.raw.{source["name"]}.edges@spark'],
                    outputs=f'ingestion.int.{source["name"]}.edges',
                    name=f'write_{source["name"]}_edges',
                    tags=[f'{source["name"]}'],
                )
            )

    return pipeline(nodes_lst)
