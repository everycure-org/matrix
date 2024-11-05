from enum import Enum


class NodeTags(Enum):
    NEO4J = "neo4j"  # TODO: What is the meaning of this tag?
    BIGQUERY = "bigquery"  # TODO: What is the meaning of this tag?
    ARGO_FUSE_NODE = "argowf.fuse"
    ARGO_FUSE_TOPOLOGICAL_EMBEDDINGS = "argowf.fuse-group.topological_embeddings"
    ARGO_NEO4J_TEMPLATE = "argowf.template-neo4j"
    ARGO_FUSE_TOPOLOGICAL_PCA = "argowf.fuse-group.topological_pca"
    ARGO_MEM_100G = "argowf.mem-100g"
    RTX_KG2 = "rtx_kg2"
    EC_MEDICAL_TEAM = "ec_medical_team"
    ROBOKOP = "robokop"
    STANDARDIZE = "standardize"
    FILTER = "filtering"
    EC_MEDICAL_KG = "ec-medical-kg"
    EC_CLINICAL_TRIALS_DATA = "ec-clinical-trials-data"
    DRUG_LIST = "drug-list"
    DISEASE_LIST = "disease-list"
    INFERENCE_INPUT = "inference-input"


def fuse_group_tag(model: str, *args: str) -> str:
    return f"argowf.fuse-group.{model}{'.'.join(arg for arg in args if arg)}"
