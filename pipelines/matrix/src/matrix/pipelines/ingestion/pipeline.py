from kedro.pipeline import Pipeline, node, pipeline

from matrix import settings


def create_ground_truth_pipeline() -> list:
    """Create pipeline nodes for ground truth processing."""
    return [
        node(
            func=lambda x, y: [x, y],
            inputs=["ingestion.raw.kgml_xdtd_ground_truth.positives", "ingestion.raw.kgml_xdtd_ground_truth.negatives"],
            outputs=[
                "ingestion.raw.kgml_xdtd_ground_truth.positive.edges@pandas",
                "ingestion.raw.kgml_xdtd_ground_truth.negative.edges@pandas",
            ],
            name="write_kgml_xdtd_gt",
            tags=["kgml_xdtd_ground_truth"],
        ),
        node(
            func=lambda x, y: [x, y],
            inputs=[
                "ingestion.raw.ec_ground_truth.positives",
                "ingestion.raw.ec_ground_truth.negatives",
            ],
            outputs=[
                "ingestion.raw.ec_ground_truth.positive.edges@pandas",
                "ingestion.raw.ec_ground_truth.negative.edges@pandas",
            ],
            name="write_ec_indications_list",
            tags=["ec_ground_truth"],
        ),
    ]


def create_pipeline(**kwargs) -> Pipeline:
    """Create ingestion pipeline."""
    # Create pipeline per source
    nodes_lst = []

    # Ground truth
    nodes_lst.extend(create_ground_truth_pipeline())

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
        if source.get("has_edges", True) and not (
            source.get("has_positive_edges", False) or source.get("has_negative_edges", False)
        ):
            nodes_lst.append(
                node(
                    func=lambda x: x,
                    inputs=[f'ingestion.raw.{source["name"]}.edges@spark'],
                    outputs=f'ingestion.int.{source["name"]}.edges',
                    name=f'write_{source["name"]}_edges',
                    tags=[f'{source["name"]}'],
                )
            )

        if source.get("has_positive_edges", False):
            nodes_lst.append(
                node(
                    func=lambda x: x,
                    inputs=f'ingestion.raw.{source["name"]}.positive.edges@spark',
                    outputs=f'ingestion.int.{source["name"]}.positive.edges',
                    name=f'write_{source["name"]}_positive_edges',
                    tags=[f'{source["name"]}'],
                ),
            )
        if source.get("has_negative_edges", False):
            nodes_lst.append(
                node(
                    func=lambda x: x,
                    inputs=f'ingestion.raw.{source["name"]}.negative.edges@spark',
                    outputs=f'ingestion.int.{source["name"]}.negative.edges',
                    name=f'write_{source["name"]}_negative_edges',
                    tags=[f'{source["name"]}'],
                )
            )
    return pipeline(nodes_lst)
