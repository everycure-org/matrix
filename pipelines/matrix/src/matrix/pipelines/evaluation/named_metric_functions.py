import abc

import numpy as np
from scipy.stats import spearmanr


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


class RCScore(NamedFunction):
    """Class representing a named vectorised function for the computation of RCScore."""

    def __init__(self, n) -> None:
        """Initialise instance of RCScore object.

        Args:
            n: Value for n.
        """
        self.n = n


class HypergeomAtN(NamedFunction):
    """Class representing a named vectorised function for the computation of Hypergeom At N."""

    def __init__(self, n) -> None:
        """Initialise instance of RCScore object.

        Args:
            n: Value for n.
        """
        self.n = n


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

        def spearman_corr(rank_sets, common_items):
            rank_set1, rank_set2 = rank_sets
            rank_set1 = rank_set1.head(self.n)
            rank_set2 = rank_set2.head(self.n)
            # Get ranks for common items
            ranks1 = [rank_set1[rank_set1.id == item].rank for item in common_items]
            ranks2 = [rank_set2[rank_set2.id == item].rank for item in common_items]

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

        def commonality_func(matrices):
            # Logic to compute commonality
            main_set = set(matrices[0]["id"])
            matrices = [matrix.head(self.n) for matrix in matrices]
            for matrix in matrices:
                main_set = main_set.intersection(set(matrix["id"]))

            if len(main_set) > 0:
                return len(main_set) / self.n
            else:
                return float("nan")

        return commonality_func

    def name(self):
        """Returns name of the function."""
        return f"commonality_at_{self.n}"
