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
"""Function to reverse engineer data fabrication parameters given data."""
from typing import Mapping, Union

import pandas as pd
import pyspark


def reverse_engineer_df(
    df: Union[pyspark.sql.DataFrame, pd.DataFrame], num_rows: int = 10
) -> Mapping:
    """Given a spark dataframe, generates data fabricator config.

    Complex dtypes like arrays and structs are not allowed.

    Args:
        df: The dataframe to reverse engineer config for.
        num_rows: The number of rows.

    Returns:
        A dictionary that can be used for data fabrication.

    Raises:
        ValueError: If spark dataframe contains complex column types.
        NotImplementedError: If a pandas dataframe is passed.
        TypeError: If neither a spark or pandas dataframe is passed.
    """
    if isinstance(df, pyspark.sql.DataFrame):
        # check for banned dtypes
        all_dtypes = {x[1] for x in df.dtypes}
        for dtype in all_dtypes:
            if "array" in dtype or "struct" in dtype:
                raise ValueError(f"dtype {dtype} not allowed. Kindly drop this column.")

        sampled_rows = [row.asDict() for row in df.limit(num_rows).collect()]
        all_columns = df.columns
    elif isinstance(df, pd.DataFrame):
        raise NotImplementedError("Not implemented. Kindly pass in a spark dataframe.")
    else:
        raise TypeError("Either pass in a pandas or spark dataframe.")

    data_fabrication_config = {}
    data_fabrication_config["num_rows"] = num_rows
    data_fabrication_config["columns"] = {}
    for column in all_columns:
        the_column_values = []
        for row in sampled_rows:
            the_column_values.append(row[column])

        data_fabrication_config["columns"][column] = {
            "type": "generate_values",
            "sample_values": list(sorted(list(set(the_column_values)))),
        }

    return data_fabrication_config


def reverse_engineer_tables(
    dictionary_of_dfs: Mapping[str, Union[pyspark.sql.DataFrame, pd.DataFrame]],
    num_rows: int = 10,
) -> Mapping:
    """Generates data fabricator config given a dictionary of dataframes.

    It is recommended to simply `yaml.safe_dump` the config into console and then
    copy paste the config into YAML. For PK/FK relationships, generate the config
    first then manually update them within the config after pasting.

    Args:
        dictionary_of_dfs: A dictionary of dataframes where the key correspnds to the
            table name.
        num_rows: The number of rows to generate config from.

    Returns:
        A dictionary that can be passed to the data fabricator.
    """
    tables_config = {}

    for table_name, df in dictionary_of_dfs.items():
        tables_config[table_name] = reverse_engineer_df(df=df, num_rows=num_rows)

    return tables_config
