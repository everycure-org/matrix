from __future__ import annotations

from kedro.pipeline import Pipeline

from core_entities.pipelines.disease_list import (
    create_pipeline as create_disease_list_pipeline,
)
from core_entities.pipelines.disease_list import (
    create_publish_hf_pipeline as create_disease_list_publish_hf_pipeline,
)
from core_entities.pipelines.disease_list import (
    create_publish_pipeline as create_disease_list_publish_pipeline,
)
from core_entities.pipelines.disease_llm_tags import (
    create_disease_categories_pipeline,
    create_disease_labels_pipeline,
    create_disease_prevalence_pipeline,
    create_disease_txgnn_pipeline,
    create_disease_umn_pipeline,
)
from core_entities.pipelines.disease_mondo import (
    create_pipeline as create_disease_mondo_pipeline,
)
from core_entities.pipelines.drug_list import (
    create_pipeline as create_drug_list_pipeline,
)
from core_entities.pipelines.drug_list import (
    create_publish_hf_pipeline as create_drug_list_publish_hf_pipeline,
)
from core_entities.pipelines.drug_list import (
    create_publish_pipeline as create_drug_list_publish_pipeline,
)


def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    pipelines = {
        "disease_mondo": create_disease_mondo_pipeline(),
        "disease_list": create_disease_mondo_pipeline() + create_disease_list_pipeline(),
        "publish_disease_list": create_disease_list_publish_pipeline(),
        "drug_list": create_drug_list_pipeline(),
        "disease_categories": create_disease_categories_pipeline(),
        "publish_drug_list": create_drug_list_publish_pipeline(),
        "publish_disease_list_hf": create_disease_list_publish_hf_pipeline(),
        "publish_drug_list_hf": create_drug_list_publish_hf_pipeline(),
        "disease_labels": create_disease_labels_pipeline(),
        "disease_umn": create_disease_umn_pipeline(),
        "disease_prevalence": create_disease_prevalence_pipeline(),
        "disease_txgnn": create_disease_txgnn_pipeline(),
    }

    pipelines["__default__"] = pipelines["disease_list"] + pipelines["drug_list"]
    return pipelines
