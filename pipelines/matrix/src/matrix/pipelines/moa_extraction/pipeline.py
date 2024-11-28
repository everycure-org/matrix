"""
MOA extraction pipeline.

Note: Several dummy variables are used to ensure that the pipeline runs in linear order for each num_hops. This is so that the argo workflow can be fused correctly.
"""

from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline


import matrix.pipelines.embeddings.nodes as embeddings_nodes
from matrix.kedro4argo_node import ArgoNode, ArgoResourceConfig

from . import nodes
from matrix import settings

moa_extraction_settings = settings.DYNAMIC_PIPELINES_MAPPING.get("moa_extraction")
num_hops_lst = [model["num_hops"] for model in moa_extraction_settings]


def _preprocessing_pipeline() -> Pipeline:
    preprocessing_strands_lst = []
    for num_hops in num_hops_lst:
        preprocessing_strands_lst.append(
            pipeline(
                [
                    ArgoNode(
                        func=embeddings_nodes.ingest_nodes,
                        inputs=["integration.prm.filtered_nodes"],
                        outputs=f"moa_extraction.input_nodes.{num_hops}_hop",
                        name=f"moa_extraction_ingest_neo4j_input_nodes_{num_hops}_hop",
                        tags=["moa_extraction.create_neo4j_db"],
                        argo_config=ArgoResourceConfig(
                            cpu_request=8,
                            cpu_limit=8,
                            memory_limit=192,
                            memory_request=120,
                        ),
                    ),
                    node(
                        func=embeddings_nodes.ingest_edges,
                        inputs=[f"moa_extraction.input_nodes.{num_hops}_hop", "integration.prm.filtered_edges"],
                        outputs=f"moa_extraction.input_edges.{num_hops}_hop",
                        name=f"ingest_neo4j_input_edges_{num_hops}_hop",
                        tags=["moa_extraction.create_neo4j_db"],
                    ),
                    node(
                        func=nodes.add_tags,
                        inputs={
                            "runner": f"params:moa_extraction.gdb_{num_hops}_hop",
                            "drug_types": "params:moa_extraction.tagging_options.drug_types",
                            "disease_types": "params:moa_extraction.tagging_options.disease_types",
                            "batch_size": "params:moa_extraction.tagging_options.batch_size",
                            "verbose": "params:moa_extraction.tagging_options.verbose",
                            "edges_dummy": f"moa_extraction.input_edges.{num_hops}_hop",
                        },
                        outputs=f"moa_extraction.reporting.add_tags_{num_hops}_hop",
                        tags=["moa_extraction.tagging"],
                        name=f"add_tags_{num_hops}_hop",
                    ),
                    node(
                        func=nodes.get_one_hot_encodings,
                        inputs={
                            "runner": f"params:moa_extraction.gdb_{num_hops}_hop",
                            "tags_dummy": f"moa_extraction.reporting.add_tags_{num_hops}_hop",
                        },
                        outputs=[
                            f"moa_extraction.feat.category_encoder_{num_hops}_hop",
                            f"moa_extraction.feat.relation_encoder_{num_hops}_hop",
                        ],
                        name=f"get_one_hot_encodings_{num_hops}_hop",
                    ),
                    node(
                        func=nodes.map_drug_mech_db,
                        inputs={
                            "runner": f"params:moa_extraction.gdb_{num_hops}_hop",
                            "drug_mech_db": "moa_extraction.raw.drug_mech_db",
                            "mapper": f"params:moa_extraction.path_mapping.mapper_{num_hops}_hop",
                            "drugmechdb_entities": "moa_extraction.raw.drugmechdb_entities",
                            "one_hot_encodings_dummy": f"moa_extraction.feat.category_encoder_{num_hops}_hop",
                        },
                        outputs=f"moa_extraction.int.{num_hops}_hop_indication_paths",
                        name=f"map_{num_hops}_hop",
                    ),
                    node(
                        func=nodes.report_mapping_success,
                        inputs={
                            "drugmechdb_entities": "moa_extraction.raw.drugmechdb_entities",
                            "drug_mech_db": "moa_extraction.raw.drug_mech_db",
                            "mapped_paths": f"moa_extraction.int.{num_hops}_hop_indication_paths",
                        },
                        outputs=f"moa_extraction.reporting.{num_hops}_hop_mapping_success@yaml",
                        name=f"report_mapping_success_{num_hops}_hop",
                    ),
                    node(
                        func=nodes.make_splits,
                        inputs={
                            "paths_data": f"moa_extraction.int.{num_hops}_hop_indication_paths",
                            "splitter": f"params:moa_extraction.splits.splitter_{num_hops}_hop",
                            "mapping_report_dummy": f"moa_extraction.reporting.{num_hops}_hop_mapping_success@yaml",
                        },
                        outputs=f"moa_extraction.prm.{num_hops}_hop_splits",
                        name=f"make_splits_{num_hops}_hop",
                    ),
                ],
                tags=[
                    "argowf.fuse",
                    f"argowf.fuse-group.moa_extraction_{num_hops}_hop",
                    "argowf.template-neo4j",
                    "moa_extraction.preprocessing",
                ],
            )
        )
    return sum(
        [
            *preprocessing_strands_lst,
        ]
    )


def create_pipeline(**kwargs) -> Pipeline:
    return _preprocessing_pipeline()
