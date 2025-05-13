from abc import ABC, abstractmethod

import pyspark.sql as ps
from pyspark.sql import functions as F
from pyspark.sql.window import Window


class Transformation(ABC):
    def __init__(self, matrix_df):
        self.matrix_df = matrix_df

    @abstractmethod
    def apply(self):
        pass


class FrequentFlyerTransformation(Transformation):
    def __init__(self, matrix_df):
        super().__init__(matrix_df)

    def apply(self):
        return frequent_flyer_transformation(self.matrix_df)


def frequent_flyer_transformation(
    matrix_df: ps.DataFrame,
    gamma: float = 0.05,
    epsilon: float = 0.001,
    score_col: str = "treat score",
    output_col: str = "transformed_score",
    perform_sort: bool = True,
) -> ps.DataFrame:
    """Compute the almost pure ranking-based transformation for a matrix of drug-disease treat scores.

    Args:
        matrix_df: Dataframe representing input drug-disease pair matrix
        gamma: Weight assigned to the matrix-wide rank
        epsilon: Weight assigned to the drug-specific rank
        score_col: Column in the input dataframe matrix_df containing the treat score
        output_col: Name of the column in the output dataframe
        perform_sort: Whether to sort the output dataframe by the transformed score
    """
    return give_rank_transformed_matrix(
        matrix_df,
        matrix_weight=epsilon,
        drug_weight=1,
        disease_weight=1,
        decay_matrix=gamma,
        decay_drug=gamma,
        decay_disease=gamma,
        score_col=score_col,
        output_col=output_col,
        perform_sort=perform_sort,
    )


def give_rank_transformed_matrix(
    matrix_df: ps.DataFrame,
    matrix_weight: int = 1,
    drug_weight: int = 1,
    disease_weight: int = 1.5,
    decay_matrix: int = 0.001,
    decay_drug: int = 0.001,
    decay_disease: int = 0.001,
    score_col: str = "treat score",
    output_col: str = "transformed_score",
    perform_sort: bool = True,
) -> ps.DataFrame:
    """Compute rank-based transformation for a matrix of drug-disease treat scores.

    Args:
        matrix_df: Dataframe representing input drug-disease pair matrix
        matrix_weight: Weight assigned to matrix-wide rank
        drug_weight: Weight assigned to drug-specific rank
        disease_weight: Weight assigned to disease-specific rank
        decay_matrix: The negative power applied to the matrix-wide quantile.
            Defines to which extent the top ranks are emphasised compared to the lower ranks.
        decay_drug: The negative power applied to the drug-specific quantile.
            Defines to which extent the top ranks are emphasised compared to the lower ranks.
        decay_disease: The negative power applied to the disease-specific quantile.
            Defines to which extent the top ranks are emphasised compared to the lower ranks.
        score_col: Column in the input dataframe matrix_df containing the treat score
        output_col: Name of the column in the output dataframe
        perform_sort: Whether to sort the output dataframe by the transformed score

    Returns:
        Input ranks with extra column for the transformed score along with new columns for the ranks and quantile ranks.
    """
    # Count entities
    N_drug = matrix_df.select("source").distinct().count()
    N_disease = matrix_df.select("target").distinct().count()
    N_matrix = matrix_df.count()

    # Define windows for ranking
    drug_window = Window.partitionBy("source").orderBy(F.col(score_col).desc())
    disease_window = Window.partitionBy("target").orderBy(F.col(score_col).desc())
    matrix_window = Window.orderBy(F.col(score_col).desc())

    # Compute ranks
    matrix_df = matrix_df.withColumn("rank_drug", F.rank().over(drug_window))
    matrix_df = matrix_df.withColumn("rank_disease", F.rank().over(disease_window))
    matrix_df = matrix_df.withColumn("rank_matrix", F.rank().over(matrix_window))

    # Compute quantile ranks
    matrix_df = matrix_df.withColumn("quantile_drug", F.col("rank_drug") / N_drug)
    matrix_df = matrix_df.withColumn("quantile_disease", F.col("rank_disease") / N_disease)
    matrix_df = matrix_df.withColumn("quantile_matrix", F.col("rank_matrix") / N_matrix)

    # Compute transformed score
    matrix_df = matrix_df.withColumn(
        output_col,
        F.pow(F.col("quantile_matrix"), -decay_matrix) * matrix_weight
        + F.pow(F.col("quantile_drug"), -decay_drug) * drug_weight
        + F.pow(F.col("quantile_disease"), -decay_disease) * disease_weight,
    )

    # Sort if requested
    if perform_sort:
        matrix_df = matrix_df.orderBy(F.col(output_col).desc())

    return matrix_df
