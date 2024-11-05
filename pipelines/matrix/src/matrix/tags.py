from enum import Enum

ARGO_NODE_PREFIX = "argowf."
ARGO_FUSE_GROUP_PREFIX = f"{ARGO_NODE_PREFIX}fuse-group."


class NodeTags(Enum):
    NEO4J = "neo4j"  # TODO: What is the meaning of this tag?
    BIGQUERY = "bigquery"  # TODO: What is the meaning of this tag?

    ## Argo tags - these trigger specific behavior in Argo
    ARGO_FUSE_NODE = f"{ARGO_NODE_PREFIX}fuse"
    ARGO_FUSE_TOPOLOGICAL_EMBEDDINGS = f"{ARGO_FUSE_GROUP_PREFIX}topological_embeddings"
    ARGO_NEO4J_TEMPLATE = f"{ARGO_NODE_PREFIX}template-neo4j"
    ARGO_FUSE_TOPOLOGICAL_PCA = f"{ARGO_FUSE_GROUP_PREFIX}topological_pca"
    ARGO_MEM_100G = f"{ARGO_NODE_PREFIX}mem-100g"  # TODO: This should be a K8s tag.

    # Kubernetes tags - these define affinity rules for K8s
    K8S_GPU = "gpu_node"

    ## Other tags
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
    return f"{ARGO_FUSE_GROUP_PREFIX}{model}{'.'.join(arg for arg in args if arg)}"
