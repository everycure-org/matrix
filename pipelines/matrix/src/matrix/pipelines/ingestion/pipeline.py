from kedro.pipeline import Pipeline, node, pipeline

from matrix import settings


def create_pipeline(**kwargs) -> Pipeline:
    """Create ingestion pipeline."""
    # Create pipeline per source
    nodes_lst = []

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
                    name=f'write_{source["name"]}',
                    tags=[f'{source["name"]}'],
                )
            )
        elif source.get("has_edges", True):
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
