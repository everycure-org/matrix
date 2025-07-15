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
    def apply(self, matrix_df: ps.DataFrame, score_col: str) -> ps.DataFrame:
        """Apply the transformation to the matrix.

        Args:
            matrix_df: Input DataFrame to transform

        Returns:
            Transformed DataFrame
        """
        pass


class NoTransformation(MatrixTransformation):
    """No transformation applied to the matrix."""

    def apply(self, matrix_df: ps.DataFrame, score_col: str) -> ps.DataFrame:
        return matrix_df


class RankBasedFrequentFlyerTransformation(MatrixTransformation):
    def __init__(
        self,
        matrix_weight: float,
        drug_weight: float,
        disease_weight: float,
        decay_matrix: float,
        decay_drug: float,
        decay_disease: float,
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
        """
        self.matrix_weight = matrix_weight
        self.drug_weight = drug_weight
        self.disease_weight = disease_weight
        self.decay_matrix = decay_matrix
        self.decay_drug = decay_drug
        self.decay_disease = decay_disease

    def apply(self, matrix_df: ps.DataFrame, score_col: str) -> ps.DataFrame:
        """Apply the frequent flyer transformation to the matrix.

        Args:
            matrix_df: Input DataFrame to transform. Must contain the following columns:
                - source: The source entity (e.g. drug)
                - target: The target entity (e.g. disease)
                - quantile_rank: The quantile rank of the score

        Returns:
            Transformed DataFrame
        """
        # Count entities
        N_drug = matrix_df.select("source").distinct().count()
        N_disease = matrix_df.select("target").distinct().count()
        N_matrix = matrix_df.count()

        logger.info(f"Computing ranks for matrix with {N_drug} drugs, {N_disease} diseases, and {N_matrix} matrix rows")

        # Define windows for ranking
        drug_window = Window.partitionBy("source").orderBy(F.col(score_col).desc())
        disease_window = Window.partitionBy("target").orderBy(F.col(score_col).desc())

        matrix_df = (
            matrix_df.withColumn("rank_drug", F.rank().over(disease_window))
            .withColumn("quantile_drug", F.col("rank_drug") / N_drug)
            .withColumn("rank_disease", F.rank().over(drug_window))
            .withColumn("quantile_disease", F.col("rank_disease") / N_disease)
            .withColumn(
                f"transformed_treat_score",
                F.pow(F.col("quantile_rank"), -self.decay_matrix) * self.matrix_weight
                + F.pow(F.col("quantile_drug"), -self.decay_drug) * self.drug_weight
                + F.pow(F.col("quantile_disease"), -self.decay_disease) * self.disease_weight,
            )
            .withColumnRenamed(score_col, f"untransformed_treat_score")
            .withColumnRenamed("rank", f"untransformed_rank")
        )

        # Recalculate rank and quantile_rank based on the new score
        score_window = Window.orderBy(F.col(f"transformed_treat_score").desc())
        matrix_df = matrix_df.withColumn("rank", F.row_number().over(score_window)).withColumn(
            "quantile_rank", F.col("rank") / N_matrix
        )
        return matrix_df


class AlmostPureRankBasedFrequentFlyerTransformation(RankBasedFrequentFlyerTransformation):
    def __init__(self, decay: float, epsilon: float = 0.001):
        """Initialize the almost pure rank-based frequent flyer transformation.

        This transformation is a special case of the frequent flyer transformation where:
            - Matrix weight is set to a small epsilon value (default: 0.001)
            - Decay factor is constant across all components (matrix, drug, and disease)
            - Drug and disease weights are equal (both set to 1.0)

        This configuration emphasizes the importance of drug and disease-specific rankings
        while minimizing the influence of the overall matrix ranking.

        Args:
            decay: The negative power applied to all component scores (matrix, drug, and disease)
            epsilon: Small value used for the matrix weight (default: 0.001)

        Returns:
            None
        """
        super().__init__(
            matrix_weight=epsilon,
            drug_weight=1.0,
            disease_weight=1.0,
            decay_matrix=decay,
            decay_drug=decay,
            decay_disease=decay,
        )

    def apply(self, matrix_df: ps.DataFrame, score_col: str) -> ps.DataFrame:
        """Apply the frequent flyer transformation to the matrix.

        Args:
            matrix_df: Input DataFrame to transform
            score_col: Column in the input dataframe containing the treat score

        Returns:
            Transformed DataFrame
        """
        return super().apply(matrix_df, score_col)


class UniformRankBasedFrequentFlyerTransformation(RankBasedFrequentFlyerTransformation):
    def __init__(self, decay: float, epsilon: float = 0.001):
        """Initialize the uniform rank-based frequent flyer transformation.

        This transformation is a special case of the frequent flyer transformation where:
            - All weights are equal (matrix, drug, and disease weights = 1.0)
            - Decay factor is constant across all components (matrix, drug, and disease)

        This configuration treats all ranking components (matrix-wide, drug-specific, and disease-specific)
        with equal importance in the final score calculation.

        Args:
            decay: The negative power applied to the all component scores
        """
        super().__init__(
            matrix_weight=1.0,
            drug_weight=1.0,
            disease_weight=1.0,
            decay_matrix=decay,
            decay_drug=decay,
            decay_disease=decay,
        )

    def apply(self, matrix_df: ps.DataFrame, score_col: str) -> ps.DataFrame:
        """Apply the frequent flyer transformation to the matrix.

        Args:
            matrix_df: Input DataFrame to transform
            score_col: Column in the input dataframe containing the treat score

        Returns:
            Transformed DataFrame
        """
        return super().apply(matrix_df, score_col)
