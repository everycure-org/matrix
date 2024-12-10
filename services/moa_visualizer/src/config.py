import os

# TODO
# Start a colour map for biolink classes
# Pastel colours
# colour_map = {
#     'biolink:SmallMolecule': '#77DD77',
#     'biolink:Disease': '#B3EBF2',
#     'biolink:Protein': '#FF964F',
# }

DATA_INPUT_PATH = os.environ.get(
    "DATA_INPUT_PATH",
    "mtrx-us-central1-hub-dev-storage/kedro/data/releases/v0.2.5-rtx-only/runs/feature-moa-extraction-4472893d/datasets/moa_extraction/reporting",
)
MOA_DB_PATH = os.environ.get("MOA_DB_PATH", "data/moa_extraction.db")
GCP_PROJECT = os.environ.get("GCP_PROJECT", "project-silc")
MOA_INFO_IMG = os.environ.get("MOA_INFO_IMG", "assets/moa_info.svg")

# Ontologies and their URLs
ONT_URLS = {
    "MONDO": "http://purl.obolibrary.org/obo/",
    "UniProtKB": "https://www.uniprot.org/uniprotkb/",
    "CHEMBL": "https://www.ebi.ac.uk/chembl/",
}

# Define columns that we wat to surface to the user, with cleaner display names
DISPLAY_COLS = {
    "drug_name": "Drug",
    "disease_name": "Disease",
    "MOA_score": "MoA Score",
    "Feedback": "Feedback",
    "drug_disease_pair_id": "Drug-Disease Pair ID",
    "drug_id": "Drug Curie",  # making assumptions these actually are curies
    "disease_id": "Disease Curie",
    "predicates_1": "Edge 1",
    "intermediate_id_1": "Node 1 ID",
    "intermediate_name_1": "Node 1 Name",
    "intermediate_type_1": "Node 1 Type",
    "predicates_2": "Edge 2",
    "intermediate_id_2": "Node 2 ID",
    "intermediate_name_2": "Node 2 Name",
    "intermediate_type_2": "Node 2 Type",
    "predicates_3": "Edge 3",
    "intermediate_id_3": "Node 3 ID",
    "intermediate_name_3": "Node 3 Name",
    "intermediate_type_3": "Node 3 Type",
}


# Info text
MOA_INFO_TEXT = """
Mechanism of action (MOA) prediction models aim to predict the biological 
pathways by which a given drug acts on a given disease. The MOA extraction 
module is an example of a path-based approach to MOA prediction. It extracts 
paths in the the knowledge graph that are likely to be relevant to the MOA of a 
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

"""
