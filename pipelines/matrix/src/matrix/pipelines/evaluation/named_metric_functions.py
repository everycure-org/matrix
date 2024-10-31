"""Module containing a collection of named functions.

Used for the computation of ranking metrics.
"""

import numpy as np
import abc


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
