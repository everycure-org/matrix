import random
import uuid
from typing import List, Tuple

import pandas as pd
from bmt import toolkit

tk = toolkit.Toolkit()


def filter_by_category(ids: List[str], categories: List[str], category: str, **kwargs):
    zipped = list(zip(ids, categories))
    drugs = [id for id, cat in zipped if cat == category]
    return drugs


def get_ancestors_for_category_delimited(category: str, delimiter: str) -> str:
    output = delimiter.join(tk.get_ancestors(category, formatted=True))
    return output


def generate_random_biolink_entities(num_rows: int, **kwargs) -> List[str]:
    """Generate a list of random biolink entities."""
    return [generate_random_biolink_entity(**kwargs) for _ in range(num_rows)]


def generate_random_biolink_entity(**kwargs) -> str:
    """Generate a random biolink entity."""
    return random.choice(tk.get_all_entities(formatted=True, **kwargs))


def generate_random_biolink_entities_list(num_rows: int, **kwargs) -> List[List[str]]:
    """Generate a list of random biolink entities."""
    e = generate_random_biolink_entities(num_rows, **kwargs)
    return [tk.get_ancestors(e, formatted=True, **kwargs) for e in e]


def get_random_biolink_predicates(subject_categories: List[str], object_categories: List[str]) -> str:
    """Given a subject and object category, get a random predicate that is valid."""
    pairs = list(zip(subject_categories, object_categories))
    return [random.choice(get_valid_predicates(s, o)) for s, o in pairs]


def get_valid_predicates(subject_category: str, object_category: str) -> list[str]:
    subject_candidates = tk.get_all_predicates_with_class_domain(subject_category, formatted=True, check_ancestors=True)
    object_candidates = tk.get_all_predicates_with_class_range(object_category, formatted=True, check_ancestors=True)
    return list(set(subject_candidates).intersection(set(object_candidates)))


def create_subject_predicate_object_mapping() -> pd.DataFrame:
    # create subject -> predicate mapping
    subject_mapping = pd.DataFrame({"subj_category": tk.get_all_entities(formatted=True)})
    subject_mapping["predicate"] = subject_mapping.apply(
        lambda row: tk.get_all_predicates_with_class_domain(row["subj_category"], formatted=True, check_ancestors=True),
        axis=1,
    )
    subject_mapping = subject_mapping.explode("predicate")

    # create predicate -> object mapping
    object_mapping = pd.DataFrame({"predicate": subject_mapping["predicate"].unique()}).dropna()
    object_mapping["obj_category"] = object_mapping.apply(
        lambda row: tk.get_slot_range(row["predicate"], formatted=True), axis=1
    )
    object_mapping = object_mapping.explode("obj_category")

    # cross join
    subject_predicate_object = subject_mapping.merge(object_mapping, on="predicate", how="inner")
    return subject_predicate_object


def generate_biolink_sample_kg(nodes_per_type: int, edge_count: int) -> Tuple[pd.DataFrame]:
    """
    Generates an initial knowledge graph with a given number of nodes per type and edges.
    We use this to generate various biolink compatible KGs
    """
    nodes_per_side = int(nodes_per_type / 2)

    possible_triples = create_subject_predicate_object_mapping()
    edges = possible_triples.sample(
        int(edge_count / nodes_per_type)
    )  # we will explode the edges by IDs and each edge brings in 2 nodes

    # generate a list of unique strings for each subject and object to use as ids
    edges["subject"] = edges.apply(lambda row: [uuid.uuid4() for _ in range(nodes_per_side)], axis=1)
    edges["object"] = edges.apply(lambda row: [uuid.uuid4() for _ in range(nodes_per_side)], axis=1)
    edges = edges.explode(["subject", "object"])

    # Prepare the nodes table
    subjects = edges[["subject", "subj_category"]].rename(columns={"subject": "id", "subj_category": "category"})
    objects = edges[["object", "obj_category"]].rename(columns={"object": "id", "obj_category": "category"})
    nodes = pd.concat([subjects, objects])
    nodes["all_categories"] = nodes.apply(lambda row: tk.get_ancestors(row["category"], formatted=True), axis=1)

    # Clean up the edges table
    edges = edges[["subject", "predicate", "object"]]
    return nodes, edges
