"""Fabricator pipeline."""
from matrix import settings
from kedro.pipeline import Pipeline, node, pipeline
from . import nodes as nd


def create_pipeline(**kwargs) -> Pipeline:
    """Create requests pipeline."""
    nodes = []
    for model in settings.DYNAMIC_PIPELINES_MAPPING.get("modelling"):
        if not model["run_inference"]:
            continue
        nodes.append(
            pipeline(
                [
                    node(
                        func=lambda x: x,
                        inputs="ingestion.raw.drug_list",
                        outputs="inference.sheet.drug_list",
                        name=f"ingest_drug_list",
                    ),
                    node(
                        func=lambda x: x,
                        inputs="ingestion.raw.disease_list",
                        outputs="inference.sheet.disease_list",
                        name=f"ingest_disease_list",
                    ),
                    node(
                        func=nd.resolve_input_sheet,
                        inputs={
                            "sheet": "inference.sheet.normalized_inputs",
                        },
                        outputs="inference.request.type",
                        name="select_request_type",
                    ),
                    node(
                        func=nd.run_inference,
                        inputs={
                            "model": f'modelling.{model["model_name"]}.models.model',
                            "embeddings": "inference.embed.nodes",
                            "infer_type": "inference.request.type",
                            "drug_nodes": "ingestion.raw.drug_list",
                            "disease_nodes": "ingestion.raw.disease_list",
                            "train_df": "modelling.model_input.splits",
                            "sheet": "inference.sheet.normalized_inputs",
                        },
                        outputs=f"model_outputs.{model['model_name']}.predictions",
                        name=f"run_inference",
                    ),
                    node(
                        func=nd.visualise_treat_scores,
                        inputs={
                            "scores": "model_outputs.node.predictions",
                            "infer_type": "inference.request.type",
                        },
                        outputs=f"model_outputs.{model['model_name']}.visualisations",
                        name=f"visualise_inference",
                    ),
                ]
            )
        )
    return pipeline(nodes)
