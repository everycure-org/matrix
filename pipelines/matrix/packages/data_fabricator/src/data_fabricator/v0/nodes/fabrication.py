# (c) McKinsey & Company 2016 – Present
# All rights reserved
#
#
# This material is intended solely for your internal use and may not be reproduced,
# disclosed or distributed without McKinsey & Company's express prior written consent.
# Except as otherwise stated, the Deliverables are provided ‘as is’, without any express
# or implied warranty, and McKinsey shall not be obligated to maintain, support, host,
# update, or correct the Deliverables. Client guarantees that McKinsey’s use of
# information provided by Client as authorised herein will not violate any law
# or contractual right of a third party. Client is responsible for the operation
# and security of its operating environment. Client is responsible for performing final
# testing (including security testing and assessment) of the code, model validation,
# and final implementation of any model in a production environment. McKinsey is not
# liable for modifications made to Deliverables by anyone other than McKinsey
# personnel, (ii) for use of any Deliverables in a live production environment or
# (iii) for use of the Deliverables by third parties; or
# (iv) the use of the Deliverables for a purpose other than the intended use
# case covered by the agreement with the Client.
# Client warrants that it will not use the Deliverables in a "closed-loop" system,
# including where no Client employee or agent is materially involved in implementing
# the Deliverables and/or insights derived from the Deliverables.
"""Node to fabricate data using the data_fabricator."""


from typing import Any, Dict, List, Union

import pandas as pd
import pyspark

from ..core.fabricator import MockDataGenerator

# List of prefix used in the dataframe name to indicate that the dataframe should not be
# persisted since it is only used as a temporary dataframe to generate the output
# dataframes.
_IGNORE_DATAFRAMES_WITH_PREFIX = ["_TEMP_", "_CATALOG_"]


def fabricate_datasets(
    fabrication_params: Dict[str, Any],
    ignore_prefix: List[str] = None,
    seed: int = None,
    **source_dfs: Dict[str, Union[pd.DataFrame, pyspark.sql.DataFrame]],
) -> Dict[str, pd.DataFrame]:
    """Fabricates datasets.

    This node passes configuration to ``MockDataGenerator`` data fabricator to fabricate
    datasets.

    Args:
        fabrication_params: Fabrication parameters to pass to ``MockDataGenerator``.
        source_dfs: Optional real-world dataframes to add to ``MockDataGenerator``.

    Returns:
        A dictionary with the fabricated pandas dataframes.
    """
    ignore_prefix = (
        ignore_prefix if ignore_prefix is not None else _IGNORE_DATAFRAMES_WITH_PREFIX
    )
    mock_generator = MockDataGenerator(instructions=fabrication_params, seed=seed)

    if source_dfs:
        for df_name, df in source_dfs.items():
            if isinstance(df, pd.DataFrame):
                mock_generator.all_dataframes[df_name] = df
            elif isinstance(df, pyspark.sql.DataFrame):
                mock_generator.all_dataframes[df_name] = df.toPandas()

    mock_generator.generate_all()

    output = {
        name: mock_generator.all_dataframes[name]
        for name in fabrication_params
        if not any(name.startswith(prefix) for prefix in ignore_prefix)
    }

    return output
