from kedro.pipeline import Pipeline

from matrix.pipelines.create_sample.pipeline import create_pipeline as create_create_sample_pipeline
from matrix.pipelines.data_release.pipeline import create_pipeline as create_data_release_pipeline
from matrix.pipelines.embeddings.pipeline import create_pipeline as create_embeddings_pipeline
from matrix.pipelines.evaluation.pipeline import create_pipeline as create_evaluation_pipeline
from matrix.pipelines.fabricator.pipeline import create_pipeline as create_fabricator_pipeline
from matrix.pipelines.filtering.pipeline import create_pipeline as create_filtering_pipeline
from matrix.pipelines.ingest_to_N4J.pipeline import create_pipeline as create_ingest_to_N4J_pipeline
from matrix.pipelines.ingestion.pipeline import create_pipeline as create_ingestion_pipeline
from matrix.pipelines.integration.pipeline import create_pipeline as create_integration_pipeline
from matrix.pipelines.matrix_generation.pipeline import (
    create_pipeline as create_matrix_pipeline,
)
from matrix.pipelines.matrix_transformations.pipeline import (
    create_pipeline as create_matrix_transformations_pipeline,
)
from matrix.pipelines.modelling.pipeline import create_pipeline as create_modelling_pipeline
from matrix.pipelines.preprocessing.pipeline import create_pipeline as create_preprocessing_pipeline
from matrix.pipelines.run_comparison.pipeline import create_pipeline as create_run_comparison_pipeline
from matrix.pipelines.sentinel.pipeline import create_pipeline as create_sentinel_pipeline


def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        Mapping from a pipeline name to a ``Pipeline`` object.
    """
    # Define pipeline combinations
    pipelines = {
        # Individual pipelines
        "preprocessing": create_preprocessing_pipeline(),  # Run manually for clinical trials and medical KG artifacts
        "fabricator": create_fabricator_pipeline(),
        "ingestion": create_ingestion_pipeline(),
        "integration": create_integration_pipeline(),
        "filtering": create_filtering_pipeline(),
        "embeddings": create_embeddings_pipeline(),
        "data_release": create_data_release_pipeline(),
        "modelling": create_modelling_pipeline(),
        "matrix_generation": create_matrix_pipeline(),
        "matrix_transformations": create_matrix_transformations_pipeline(),
        "pre_transformed_evaluation": create_evaluation_pipeline(
            matrix_input="matrix_generation", score_col_name="treat score"
        ),
        "transformed_evaluation": create_evaluation_pipeline(
            matrix_input="matrix_transformations", score_col_name="transformed_treat_score"
        ),
        "create_sample": create_create_sample_pipeline(),
        "ingest_to_N4J": create_ingest_to_N4J_pipeline(),
        "sentinel_kg_release_patch": create_sentinel_pipeline(is_patch=True),
        "sentinel_kg_release": create_sentinel_pipeline(is_patch=False),
        "run_comparison": create_run_comparison_pipeline(),
        # "inference": create_inference_pipeline(),  # Run manually based on medical input
    }

    # Higher order pipelines
    # fmt: off
    pipelines["data_engineering"] = (
          pipelines["ingestion"]
        + pipelines["integration"]
    )

    pipelines["feature"] = (
        pipelines["filtering"]
        + pipelines["embeddings"]
    )
    pipelines["evaluation"] = (
        pipelines["pre_transformed_evaluation"] 
        + pipelines["transformed_evaluation"]
    )
    pipelines["modelling_run"] = (
          pipelines["modelling"]
        + pipelines["matrix_generation"]
        + pipelines["matrix_transformations"]
        + pipelines["evaluation"]
    )
    pipelines["feature_and_modelling_run"] = (
        pipelines["feature"]
        + pipelines["modelling_run"]
    )
    pipelines["__default__"] = (
          pipelines["data_engineering"]
        + pipelines["feature_and_modelling_run"]
    )

    pipelines["kg_release_and_matrix_run"] = (
        pipelines["data_engineering"]
        + pipelines["data_release"]
        + pipelines["ingest_to_N4J"]
        + pipelines["feature_and_modelling_run"]
        + pipelines["sentinel_kg_release"]
    )

    pipelines["kg_release_patch_and_matrix_run"] = (
        pipelines["data_engineering"]
        + pipelines["data_release"]
        + pipelines["feature_and_modelling_run"]
        + pipelines["sentinel_kg_release_patch"]
    )

    # Test pipelines
    pipelines["test"] = (
        pipelines["fabricator"]
        + pipelines["__default__"]
        + pipelines["data_release"]
        + pipelines["ingest_to_N4J"]
    )
    pipelines["test_sample"] = (
        pipelines["feature_and_modelling_run"]
    )
    # fmt: on

    return pipelines
