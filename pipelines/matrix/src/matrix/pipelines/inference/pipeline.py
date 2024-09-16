"""Fabricator pipeline."""
from matrix import settings
from kedro.pipeline import Pipeline, node, pipeline
from . import nodes as nd
from ..matrix_generation import nodes as matrix_gen


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
                        outputs="inference.raw.drug_list",
                        name=f"ingest_drug_list",
                    ),
                    node(
                        func=lambda x: x,
                        inputs="ingestion.raw.disease_list",
                        outputs="inference.raw.disease_list",
                        name=f"ingest_disease_list",
                    ),
                    node(
                        func=nd.resolve_input_sheet,
                        inputs={
                            "input_sheet": "inference.raw.normalized_inputs",
                            "drug_sheet": "inference.raw.drug_list",
                            "disease_sheet": "inference.raw.disease_list",
                        },
                        outputs=[
                            "inference.int.request_type",
                            "inference.int.drug_list",
                            "inference.int.disease_list",
                        ],
                        name="resolve_input_sheet",
                    ),
                    node(
                        func=matrix_gen.generate_pairs,
                        inputs=[
                            "inference.int.drug_list",
                            "inference.int.disease_list",
                            "modelling.model_input.splits",
                        ],
                        outputs=f"inference.model_input.drug_disease_pairs",
                        name="generate_pairs_per_request",
                    ),
                    node(
                        func=matrix_gen.make_predictions_and_sort,
                        inputs=[
                            "modelling.feat.rtx_kg2",
                            "inference.model_input.drug_disease_pairs",
                            f"modelling.{model['model_name']}.model_input.transformers",
                            f"modelling.{model['model_name']}.models.model",
                            f"params:modelling.{model['model_name']}.model_options.model_tuning_args.features",
                            "params:inference.score_col_name",
                            "params:inference.matrix_generation_options.batch_by",
                        ],
                        outputs=f"inference.{model['model_name']}.predictions",
                        name=f"request_{model['model_name']}_predictions_and_sort",
                    ),
                    node(
                        func=matrix_gen.generate_report,
                        inputs=[
                            f"inference.{model['model_name']}.predictions",
                            "params:inference.matrix_generation_options.n_reporting",
                            "inference.raw.drug_list",
                            "inference.raw.disease_list",
                            "modelling.model_input.splits",
                            "params:inference.score_col_name",
                        ],
                        outputs=f"inference.{model['model_name']}.report",
                        name=f"add_metadata",
                    ),
                    # FUTURE: add describe_scores node once we get input from the medical team
                    # node(func=nd.describe_scores)
                    node(
                        func=nd.visualise_treat_scores,
                        inputs={
                            "scores": f"inference.{model['model_name']}.predictions",
                            "infer_type": "inference.int.request_type",
                        },
                        outputs=f"inference.{model['model_name']}.visualisations",
                        name=f"visualise_inference",
                    ),
                ]
            )
        )
    return pipeline(nodes)
