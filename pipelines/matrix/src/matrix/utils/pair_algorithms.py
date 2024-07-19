from typing import List
from random import choices
import numpy as np


def efficient_alternate_pairs(
    drugs: List[str], diseases: List[str], pairs: List[tuple[str]], multi=1
) -> List[tuple[str]]:
    """Returns a new set of pairs that were pairwise swapped efficiently.

    This algorithm works by creating 2 random pools to choose drugs and diseases from.
    We then iterate until we have found enough new pairs that are not overlapping with the old pairs.

    To efficiently compare, we use a set for the new drugs.

    Say we get n pairs.

    To create new pairs, we grab n random drugs and n random diseases. We then swap out the diseases
    of the original pairs with the newly selected ones and remember that matrix. Then we do the same
    for drugs and remember it as well. Now we have 2 new pairs for each old pair: One with the drug
    swapped out and one with the disease swapped out. We add these newly discovered pairs to the new_pairs
    set.

    Next we remove all original pairs from the new_pairs set and check if we got enough. If yes, we
    return the desired count, if not we repeat until we got em all.

    Args:
        drugs: A list of drugs to select from
        diseases: A list of diseases to select from
        pairs: a list of known pairs that we don't want to pick again
    """
    pairs_s = set(pairs)
    new_pairs = set()
    target_length = multi * len(pairs)

    pairs_m = np.array(pairs)

    # keep grabbing till we got our list length
    i = 0
    while len(new_pairs) < target_length:
        start_pairs_length = len(new_pairs)
        print(f"round {i}: we got {len(new_pairs)} and want {target_length}")
        # 1. grab a random set of drugs and diseases of length*pairs length
        new_drugs = np.array(choices(drugs, k=len(pairs)))
        new_diseases = np.array(choices(diseases, k=len(pairs)))

        # 2. create new pairs, once swapped out drugs, once swapped out diseases
        new_pairs_ndrugs = np.column_stack((new_drugs, pairs_m[:, 1]))
        new_pairs_ndiseases = np.column_stack((new_diseases, pairs_m[:, 1]))

        # 3. add to the new_pairs, ensuring we add no item that is part of the original list
        new_pairs_m = np.concatenate((new_pairs_ndiseases, new_pairs_ndrugs))
        new_pairs.update(set([tuple(pair) for pair in new_pairs_m.tolist()]))
        new_pairs = new_pairs - pairs_s
        print(
            f"round {i} ended: we got {len(new_pairs)} and finding {len(new_pairs_m)} new pairs"
        )

        if start_pairs_length == len(new_pairs):
            # if we have not found a single new pair we conservatively assume we are all out of pairs and abort.
            raise ValueError("Could not find any more random pairs, giving up")

    return list(new_pairs)[0:target_length]
