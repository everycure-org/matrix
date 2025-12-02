import logging
from functools import reduce

import pyspark.sql as ps
from matrix_inject.inject import inject_object
from matrix_pandera.validator import Column, DataFrameSchema, check_output
from pyspark.sql import functions as F

logger = logging.getLogger(__name__)


@check_output(
    schema=DataFrameSchema(
        columns={
            "drug_id": Column(str, nullable=False),
            "disease_id": Column(str, nullable=False),
        },
        unique=["drug_id", "disease_id"],
    )
)
@inject_object()
def concatenate_datasets(
    datasets_to_include: dict[str, dict[str, bool]], **all_datasets: dict[str, ps.DataFrame]
) -> ps.DataFrame:
    """
    Concatenate datasets based the input parameters.

    datasets_to_include: For each available dataset, contains a dictionary of boolean values for each pair type
        (positives, negatives) determining whether to include positive (y = 1) and/or negative (y = 0) pairs.
    all_datasets: A dictionary of all available datasets.
        Columns required: "subject", "object", "y"

    Returns:
        A dataframe with the unique drug-disease pairs from the union of the included datasets.
    """
    y_values_required = {
        dataset_name: [
            y
            for y, pair_type in zip([0, 1], ["negatives", "positives"])
            if datasets_to_include[dataset_name][pair_type]
        ]
        for dataset_name in all_datasets.keys()
    }

    dataframes_to_concatenate = [
        df.filter(F.col("y").isin(y_values_required[dataset_name])).select(
            F.col("subject").alias("drug_id"), F.col("object").alias("disease_id")
        )
        for dataset_name, df in all_datasets.items()
        if len(y_values_required[dataset_name]) > 0
    ]
    return reduce(lambda df1, df2: df1.union(df2), dataframes_to_concatenate).distinct()
