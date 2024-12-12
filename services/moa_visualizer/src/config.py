from typing import Dict
from pydantic_settings import BaseSettings
from pydantic import Field, HttpUrl
from pathlib import Path
import streamlit as st
# from enum import Enum


# class BiolinkClass(str, Enum):
#     """Enumeration of supported biolink classes."""
#     SMALL_MOLECULE = "small_molecule"
#     DISEASE = "disease"
#     PROTEIN = "protein"


# class ColourMap(BaseSettings):
#     """Pastel colours for biolink classes."""
#     colour_map: Dict[BiolinkClass, str] = {
#         BiolinkClass.SMALL_MOLECULE: '#77DD77',
#         BiolinkClass.DISEASE: '#B3EBF2',
#         BiolinkClass.PROTEIN: '#FF964F',
#     }


class OntologyUrls(BaseSettings):
    """URLs for different ontologies."""

    MONDO: HttpUrl = Field("http://purl.obolibrary.org/obo/", description="MONDO ontology base URL")
    UniProtKB: HttpUrl = Field("https://www.uniprot.org/uniprotkb/", description="UniProtKB database URL")
    CHEMBL: HttpUrl = Field("https://www.ebi.ac.uk/chembl/", description="ChEMBL database URL")

    class Config:
        frozen = True


class NodeColumns(BaseSettings):
    """Configuration for individual pathway node columns."""

    predicates: str
    id: str
    name: str
    type: str


class DisplayColumns(BaseSettings):
    """Configuration for display columns."""

    # Note this was generated using AI assistance
    all_columns: list[str] = Field(default_factory=list, description="All column display names in order")
    all_keys: list[str] = Field(default_factory=list, description="All field names including node columns")
    n_nodes: int = Field(default=0, description="Number of nodes")

    drug_name: str = Field("Drug", description="Display name for drug column")
    disease_name: str = Field("Disease", description="Display name for disease column")
    MOA_score: str = Field("MoA Score", description="Display name for MOA score")
    feedback: str = Field("Feedback", description="Display name for feedback column")
    drug_disease_pair_id: str = Field("Drug-Disease Pair ID", description="Display name for drug-disease pair ID")
    drug_id: str = Field("Drug Curie", description="Display name for drug ID")
    disease_id: str = Field("Disease Curie", description="Display name for disease ID")
    node_columns: Dict[int, NodeColumns] = Field(
        default_factory=lambda: {
            i: NodeColumns(predicates=f"Edge {i}", id=f"Node {i} ID", name=f"Node {i} Name", type=f"Node {i} Type")
            for i in range(1, 4)  # TODO dynamic number of nodes, or just set
            # max number of nodes in config
        },
        description="Mapping of node number to column names",
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        updated_values = {
            "all_columns": self.get_all_columns(),
            "all_keys": self.get_all_keys(),
            "n_nodes": len(self.node_columns),
        }
        object.__setattr__(self, "__dict__", {**self.__dict__, **updated_values})

    @property
    def basic_fields(self) -> list[str]:
        """Get the basic fields that are common between display names and keys,
        Ignores nodes as these are variable."""
        return ["drug_name", "disease_name", "MOA_score", "feedback", "drug_disease_pair_id", "drug_id", "disease_id"]

    def get_all_columns(self) -> list[str]:
        """Get all column display names in order."""
        basic_display_fields = [getattr(self, field) for field in self.basic_fields]

        node_fields = [
            field for node in self.node_columns.values() for field in [node.predicates, node.id, node.name, node.type]
        ]

        return basic_display_fields + node_fields

    def get_all_keys(self) -> list[str]:
        """Get all field names including node column fields."""
        node_fields = [
            f"node_columns_{i}_{field}" for i in range(1, 4) for field in ["predicates", "id", "name", "type"]
        ]
        return self.basic_fields + node_fields


class Settings(BaseSettings):
    """Main configuration settings."""

    data_input_path: Path = Field(
        default="mtrx-us-central1-hub-dev-storage/kedro/data/releases/v0.2.5-rtx-only/runs/feature-moa-extraction-4472893d/datasets/moa_extraction/reporting",
        env="DATA_INPUT_PATH",
        description="Path to input data directory",
    )
    moa_db_path: Path = Field(
        default="data/moa_extraction.db", env="MOA_DB_PATH", description="Path to MOA database file"
    )
    gcp_project: str = Field(default="project-silc", env="GCP_PROJECT", description="GCP project identifier")
    moa_info_img: Path = Field(default="assets/moa_info.svg", env="MOA_INFO_IMG", description="Path to MOA info image")
    moa_info_text: str = Field(
        default="""
Mechanism of action (MOA) prediction models aim to predict the biological 
pathways by which a given drug acts on a given disease. The MOA extraction 
module is an example of a path-based approach to MOA prediction. It extracts 
paths in the knowledge graph that are likely to be relevant to the MOA of a 
given drug-disease pair. The MOA extraction module is loosely inspired by the 
[KGML-xDTD](https://github.com/chunyuma/KGML-xDTD) MOA module which extracts 
paths in the KG using an adversarial actor-critic reinforcement learning 
algorithm. In contrast, we use a simpler approach of extracting paths using a 
supervised binary classifier.

The main component of the MOA extraction system is a binary classifier that 
predicts whether a given path in the knowledge graph is likely to represent a 
mechanism of action for a given drug-disease pair. The binary classifier is 
trained on a dataset of manually annotated MoA pathways. The prediction 
process is illustrated by the following diagram:
""",
        description="Explainer for MOA information",
    )

    class Config:
        env_file = ".env"
        case_sensitive = True
        validate_assignment = True


@st.cache_resource
def initialise_settings():
    """Initialize and cache settings for the Streamlit app."""
    return {"settings": Settings(), "ont_urls": OntologyUrls(), "display_cols": DisplayColumns()}


configs = initialise_settings()
settings = configs["settings"]
ont_urls = configs["ont_urls"]
display_cols = configs["display_cols"]
