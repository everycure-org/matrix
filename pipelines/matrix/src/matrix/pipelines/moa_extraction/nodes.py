"""Module with nodes for evaluation."""

from typing import List

from refit.v1.core.inject import inject_object


from .neo4j_runners import Neo4jRunner


def _tag_edges_between_types(
    runner: Neo4jRunner,
    type_1_lst: List[str],
    type_2_lst: List[str],
    tag: str,
    batch_size: int,
    verbose: bool,
    prefix: str = "_moa_extraction_",
) -> None:
    """Tag edges between two types.

    Note that this function is agnostic of directionality, that is, it will
    add the tag to an edge from the first to the second node regardless of
    the direction.

    Args:
        runner: The Neo4j runner.
        type_1_lst: List of types for the first node.
        type_2_lst: List of types for the second node.
        tag: The tag to add.
        batch_size: The batch size to use for the query.
        verbose: Whether to print the number of batches completed.
        prefix: The prefix to add to the tag.
    """

    def update_relationships_in_batches(query_with_limit: str):
        """Update relationships in batches.

        Args:
            query_with_limit: The query to run with a limit and a return statement for the count of edges updated.
        """
        total_updated = 0
        while True:
            result = runner.run(query_with_limit)
            updated = result[0][0]
            total_updated += updated
            if updated < batch_size:
                break
            if verbose:
                print(f"{int(total_updated/batch_size)} batches completed.")
        return total_updated

    # Reset the tag for all edges
    if verbose:
        print(f"Resetting tag {tag} for all edges. Batch size: {batch_size}...")
    update_relationships_in_batches(f"""
        MATCH ()-[r]-()
        WHERE r.{prefix}{tag} IS NULL OR r.{prefix}{tag} = true
        WITH r LIMIT {batch_size}
        SET r.{prefix}{tag} = false
        RETURN count(r) AS updated
    """)
    # Set the tag for the specific edges
    for type_1 in type_1_lst:
        for type_2 in type_2_lst:
            if verbose:
                print(f"Setting tags for {type_1} to {type_2}...")
            update_relationships_in_batches(f"""
                MATCH (n1: {type_1})-[r]-(n2: {type_2})
                WHERE r.{prefix}{tag} = false
                WITH r LIMIT {batch_size}
                SET r.{prefix}{tag} = true
                RETURN count(r) AS updated
            """)


@inject_object()
def add_tags(
    runner: Neo4jRunner,
    drug_types: List[str],
    disease_types: List[str],
    batch_size: int,
    verbose: bool,
    prefix: str = "_moa_extraction_",
) -> None:
    """Add tags to the Neo4j database.

    Args:
        runner: The Neo4j runner.
        drug_types: List of KG node types representing drugs.
        disease_types: List of KG node types representing diseases.
        batch_size: The batch size to use for the query.
        verbose: Whether to print the number of batches completed.
        prefix: The prefix to add to the tag.
    """
    _tag_edges_between_types(runner, drug_types, disease_types, "drug_disease", batch_size, verbose, prefix)
    _tag_edges_between_types(runner, disease_types, disease_types, "disease_disease", batch_size, verbose, prefix)


# def get_one_hot_encoding(
