import abc
from typing import List, Tuple

import numpy as np
from scipy.stats import hypergeom, spearmanr


class NamedFunction(abc.ABC):
    """Class representing a named vectorised function.

    Used in the computation of ranking-based evaluation metrics.
    """

    def generate(self):
        """Returns function."""
        ...

    def name(self):
        """Returns name of the function."""
        ...


class MRR(NamedFunction):
    """Class representing a named vectorised function for the computation of MRR."""

    @staticmethod
    def generate():
        """Returns function."""
        return lambda rank: 1 / rank

    @staticmethod
    def name():
        """Returns name of the function."""
        return "mrr"


class HitK(NamedFunction):
    """Class representing a named vectorised function for the computation of Hit@k."""

    def __init__(self, k) -> None:
        """Initialise instance of Hitk object.

        Args:
            k: Value for k.
        """
        self.k = k

    def generate(self):
        """Returns function."""
        return lambda rank: np.where(rank <= self.k, 1, 0)

    def name(self):
        """Returns name of the function."""
        return "hit-" + str(self.k)


class RecallAtN(NamedFunction):
    """Class representing a named vectorised function for the computation of Recall@n."""

    def __init__(self, n) -> None:
        """Initialise instance of RecallAtN object.

        Args:
            n: Value for n.
        """
        self.n = n

    def generate(self):
        """Returns function."""
        return lambda rank: np.where(rank <= self.n, 1, 0)

    def name(self):
        """Returns name of the function."""
        return "recall-" + str(self.n)


class AUROC(NamedFunction):
    """Class representing a named vectorised function for the computation of AUROC metric."""

    def __init__(self) -> None:
        """Initialise instance of AUROC object."""

    def generate(self):
        """Returns function."""
        return lambda quantile: 1 - quantile

    def name(self):
        """Returns name of the function."""
        return "auroc"


class HypergeomAtN(NamedFunction):
    """Class representing a named vectorised function for the computation of Hypergeom At N."""

    def __init__(self, n) -> None:
        """Initialise instance of RCScore object.

        Args:
            n: Value for n.
        """
        self.n = n

    def generate(self):
        """Returns function that computes hypergeom at N.

        Returns:
            Function that takes a set of items and returns the ratio of set size to N.
        """

        def hypergeom_func(rank_sets: Tuple, common_items: List):
            rank_set1, rank_set2 = rank_sets

            # Get top-k items from each model based on raw rankings
            rank_set1 = rank_set1.head(self.n)
            rank_set2 = rank_set2.head(self.n)
            common_items = [
                id
                for id in common_items["pair_id"].values
                if ((id in rank_set1["pair_id"].values) and (id in rank_set2["pair_id"].values))
            ]

            # Get ranks for common items
            ranks1 = set([rank_set1[rank_set1.pair_id == item]["rank"].values.tolist()[0] for item in common_items])
            ranks2 = set([rank_set2[rank_set2.pair_id == item]["rank"].values.tolist()[0] for item in common_items])
            # Overlap
            overlap = len(ranks1 & ranks2)
            # Total number of pairs
            N = len(set(rank_set1["pair_id"]) | set(rank_set2["pair_id"]))

            # Calculate expected overlap by chance
            expected_overlap = (self.n * self.n) / N
            return {"enrichment": overlap / expected_overlap, "pvalue": hypergeom.sf(overlap - 1, N, self.n, self.n)}

        return hypergeom_func

    def name(self):
        """Returns name of the function."""
        return f"hypergeom_at_{self.n}"


class SpearmanAtN(NamedFunction):
    """Class representing a named vectorised function for the computation of Hypergeom At N."""

    def __init__(self, n) -> None:
        """Initialise instance of RCScore object.

        Args:
            n: Value for n.
        """
        self.n = n

    def generate(self):
        """Returns function that computes commonality at N.

        Returns:
            Function that takes a set of items and returns the ratio of set size to N.
        """

        def spearman_corr(rank_sets: Tuple, common_items: List):
            rank_set1, rank_set2 = rank_sets
            rank_set1 = rank_set1.head(self.n)
            rank_set2 = rank_set2.head(self.n)
            common_items = [
                id
                for id in common_items["pair_id"].values
                if ((id in rank_set1["pair_id"].values) and (id in rank_set2["pair_id"].values))
            ]
            # Get ranks for common items
            ranks1 = [rank_set1[rank_set1.pair_id == item]["rank"].values.tolist()[0] for item in common_items]
            ranks2 = [rank_set2[rank_set2.pair_id == item]["rank"].values.tolist()[0] for item in common_items]
            if len(ranks1) > 1:  # Ensure there are enough pairs to calculate correlation
                out = spearmanr(ranks1, ranks2)
                return {"correlation": out.correlation, "pvalue": out.pvalue}
            else:
                return {"correlation": float("nan"), "pvalue": float("nan")}

        return spearman_corr

    def name(self):
        """Returns name of the function."""
        return f"spearman_at_{self.n}"


class CommonalityAtN(NamedFunction):
    """Class representing a named vectorised function for the computation of commonality at N."""

    def __init__(self, n) -> None:
        """Initialise instance of CommonalityAtN object.

        Args:
            n: Value for n.
        """
        self.n = n

    def generate(self):
        """Returns function that computes commonality at N.

        Returns:
            Function that takes a set of items and returns the ratio of set size to N.
        """

        def commonality_func(matrices: List):
            main_set = set(matrices[0]["pair_id"])
            matrices = [matrix.head(self.n) for matrix in matrices]
            for matrix in matrices:
                main_set = main_set.intersection(set(matrix["pair_id"]))
            return len(main_set) / self.n

        return commonality_func

    def name(self):
        """Returns name of the function."""
        return f"commonality_at_{self.n}"
