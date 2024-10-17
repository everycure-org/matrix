import pandas as pd

from typing import Dict, Any, List, Optional


def unnest(predicates: List[Dict[str, Any]], parents: Optional[List[str]] = None, depth: int = 1):
    """
    Function to unnest biolink predicates.

    Args:
        predicates: predicates to unnest
        parents: list of parents in hierarchy
        depth: depth in the hierarchy
    Returns:
        Unnested dataframe
    """

    slices = []
    for predicate in predicates:
        name = predicate.get("name")

        # Recurse the children
        if children := predicate.get("children"):
            slices.append(unnest(children, parents=[*parents, name], depth=depth + 1))

        slices.append(pd.DataFrame([[name, parents, depth]], columns=["name", "parents", "depth"]))

    return pd.concat(slices, ignore_index=True)
