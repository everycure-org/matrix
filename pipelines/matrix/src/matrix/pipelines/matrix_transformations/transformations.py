import logging
from abc import ABC, abstractmethod

import pyspark.sql as ps
from pyspark.sql import functions as F
from pyspark.sql.window import Window

logger = logging.getLogger(__name__)


class MatrixTransformation(ABC):
    """
    Base class for all matrix transformations in the pipeline.
    """

    @abstractmethod
    def apply(self, matrix_df: ps.DataFrame) -> ps.DataFrame:
        """Apply the transformation to the matrix.

        Args:
            matrix_df: Input DataFrame to transform

        Returns:
            Transformed DataFrame
        """
        pass


class RankBasedFrequentFlyerTransformation(MatrixTransformation):
    def __init__(
        self,
        matrix_weight: float = 0.001,
        drug_weight: float = 1.0,
        disease_weight: float = 1.0,
        decay_matrix: float = 0.05,
        decay_drug: float = 0.05,
        decay_disease: float = 0.05,
        score_col: str = "treat score",
        perform_sort: bool = True,
    ):
        """Initialize the frequent flyer transformation.

        Args:
            matrix_weight: Weight assigned to matrix-wide rank
            drug_weight: Weight assigned to drug-specific rank
            disease_weight: Weight assigned to disease-specific rank
            decay_matrix: The negative power applied to the matrix-wide quantile
            decay_drug: The negative power applied to the drug-specific quantile
            decay_disease: The negative power applied to the disease-specific quantile
            score_col: Column in the input dataframe containing the treat score
            perform_sort: Whether to sort the output dataframe by the transformed score
        """
        self.matrix_weight = matrix_weight
        self.drug_weight = drug_weight
        self.disease_weight = disease_weight
        self.decay_matrix = decay_matrix
        self.decay_drug = decay_drug
        self.decay_disease = decay_disease
        self.score_col = score_col
        self.perform_sort = perform_sort

    def apply(self, matrix_df: ps.DataFrame) -> ps.DataFrame:
        """Apply the frequent flyer transformation to the matrix.

        Args:
            matrix_df: Input DataFrame to transform

        Returns:
            Transformed DataFrame
        """
        # Count entities
        N_drug = matrix_df.select("source").distinct().count()
        N_disease = matrix_df.select("target").distinct().count()
        N_matrix = matrix_df.count()

        logger.info(f"Computing ranks for matrix with {N_drug} drugs, {N_disease} diseases, and {N_matrix} matrix rows")

        # Define windows for ranking
        drug_window = Window.partitionBy("source").orderBy(F.col(self.score_col).desc())
        disease_window = Window.partitionBy("target").orderBy(F.col(self.score_col).desc())

        matrix_df = (
            matrix_df.withColumn("rank_drug", F.rank().over(drug_window))
            .withColumn("quantile_drug", F.col("rank_drug") / N_drug)
            .withColumn("rank_disease", F.rank().over(disease_window))
            .withColumn("quantile_disease", F.col("rank_disease") / N_disease)
        )

        # Compute transformed score
        matrix_df = (
            matrix_df.withColumn(f"untransformed_{self.score_col}", F.col(self.score_col))
            .withColumn(f"untransformed_rank", F.col("rank"))
            .withColumn(
                self.score_col,
                F.pow(F.col("quantile_rank"), -self.decay_matrix) * self.matrix_weight
                + F.pow(F.col("quantile_drug"), -self.decay_drug) * self.drug_weight
                + F.pow(F.col("quantile_disease"), -self.decay_disease) * self.disease_weight,
            )
        )

        # Recalculate rank and quantile_rank based on the new score
        score_window = Window.orderBy(F.col(self.score_col).desc())
        matrix_df = matrix_df.withColumn("rank", F.rank().over(score_window)).withColumn(
            "quantile_rank", F.col("rank") / N_matrix
        )

        # Sort if requested
        if self.perform_sort:
            matrix_df = matrix_df.orderBy(F.col("rank").desc())

        return matrix_df


class AlmostPureRankBasedFrequentFlyerTransformation(MatrixTransformation):
    def __init__(
        self,
        decay: float,
        score_col: str = "treat score",
        perform_sort: bool = True,
    ):
        """Initialize the frequent flyer transformation.

        Args:
            decay: The negative power applied to the all component scores
            score_col: Column in the input dataframe containing the treat score
            perform_sort: Whether to sort the output dataframe by the transformed score
        """
        self.decay = decay
        self.score_col = score_col
        self.perform_sort = perform_sort

    def apply(self, matrix_df: ps.DataFrame) -> ps.DataFrame:
        """Apply the frequent flyer transformation to the matrix.

        Args:
            matrix_df: Input DataFrame to transform

        Returns:
            Transformed DataFrame
        """

        return RankBasedFrequentFlyerTransformation(
            matrix_weight=0.001,
            drug_weight=1.0,
            disease_weight=1.0,
            decay_matrix=self.decay,
            decay_drug=self.decay,
            decay_disease=self.decay,
            score_col=self.score_col,
            perform_sort=self.perform_sort,
        ).apply(matrix_df)
