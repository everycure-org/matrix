from linkml_runtime.utils.schemaview import SchemaView
import pandas as pd
from generate_biolink_imports import get_biolink_model_schemaview

VALID_EDGE_TYPE_TABLE = "valid_biolink_edge_types.tsv"

def generate_valid_biolink_edge_types():
    """
    Generate a table of valid edge types for the Biolink model.
    This includes both predicate-based and class definition-based edge types.

    """
    sv = get_biolink_model_schemaview()

    predicate_valid_triples = []

    predicates = sv.slot_descendants('related to')
    for predicate in predicates:
        slot = sv.get_slot(predicate)

        all_subject_categories = [sv.get_uri(class_name) for class_name in sv.class_descendants(slot.domain)] if slot.domain else None
        all_object_categories = [sv.get_uri(class_name) for class_name in sv.class_descendants(slot.range)] if slot.range else None
        if all_subject_categories and all_object_categories:
            for subject_category in all_subject_categories:
                for object_category in all_object_categories:
                    predicate_valid_triples.append({
                        "subject_category": subject_category,
                        "predicate": sv.get_uri(predicate),
                        "object_category": object_category
                    })

    class_definition_valid_triples = []

    for class_name in sv.class_descendants('association'):
        cls = sv.get_class(class_name)
        slot_usage = cls.slot_usage
        if slot_usage is None:
            continue
        if 'subject' not in slot_usage or slot_usage['subject'] is None or slot_usage['subject'].range is None:
            continue
        if 'predicate' not in slot_usage or slot_usage['predicate'].subproperty_of is None:
            continue
        if 'object' not in slot_usage or slot_usage['object'] is None or slot_usage['object'].range is None:
            continue


        subject_range = slot_usage['subject'].range
        all_subject_categories = [sv.get_uri(class_name) for class_name in sv.class_descendants(subject_range)]
        all_predicates = [sv.get_uri(predicate_name) for predicate_name in sv.slot_descendants(slot_usage['predicate'].subproperty_of)]
        object_range = slot_usage['object'].range
        all_object_categories = [sv.get_uri(class_name) for class_name in sv.class_descendants(object_range)]
        for subject_category in all_subject_categories:
            for predicate in all_predicates:
                for object_category in all_object_categories:
                    class_definition_valid_triples.append({
                        "subject_category": subject_category,
                        "predicate": predicate,
                        "object_category": object_category
                    })


    predicate_valid_triples_df = pd.DataFrame(predicate_valid_triples)
    association_valid_triples_df = pd.DataFrame(class_definition_valid_triples)
    all_valid_triples_df = pd.merge(predicate_valid_triples_df, association_valid_triples_df,
                                    how='outer', on=['subject_category','predicate', 'object_category'])

    output_path = f"src/matrix_validator/resources/{VALID_EDGE_TYPE_TABLE}"
    all_valid_triples_df.to_csv(output_path, sep='\t', index=False)


if __name__ == "__main__":
    generate_valid_biolink_edge_types()