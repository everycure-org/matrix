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
"""Hydra instantiator node to handle dependency injection."""
# flake8: noqa
# pylint: disable=too-many-nested-blocks

from typing import Any, Dict, List, Literal

import hydra
import pandas as pd

from ..core.mock_generator import BaseTable, MockDataGenerator

# List of prefix used in the dataframe name to indicate that the dataframe should not be
# persisted since it is only used as a temporary dataframe to generate the output
# dataframes.
_IGNORE_DATAFRAMES_WITH_PREFIX = ["_TEMP_", "_CATALOG_"]


def hydra_instantiate_dictionary(
    raw_dict: Dict[str, Any],
    convert_type: Literal["none", "partial", "all"] = "all",
) -> Dict[str, Any]:
    """Instantiate objects and functions using hydra.

    Supports the `_target_` and `_partial_` syntax to inject any dependencies.

    Args:
        raw_dict: Uninstantiated dictionary
        convert_type: instantiate's argument conversion strategy
            to convert OmegaConf types to regular Python types
            https://omegaconf.readthedocs.io/en/2.3_branch/index.html
    Returns:
        Dictionary with dependencies injected and instantiated.
    """
    inst_python_dict = hydra.utils.instantiate(raw_dict, _convert_=convert_type)
    return inst_python_dict


def fabricate_datasets(
    ignore_prefix: List[str] = None,
    seed: int = None,
    **fabricator_params: Dict[str, Any],
) -> Dict[str, pd.DataFrame]:
    """Fabricate datasets.

    This node passes configuration to ``MockDataGenerator`` data fabricator to fabricate
    datasets.

    Args:
        fabricator_params: Fabrication parameters to pass to ``MockDataGenerator``.
            This argument can be one or more dictionaries of tables (for instance,
            when there are more than one `parameter.yaml` file).

    Returns:
        A dictionary with the fabricated pandas dataframes.
    """
    ignore_prefix = (
        ignore_prefix if ignore_prefix is not None else _IGNORE_DATAFRAMES_WITH_PREFIX
    )
    tables_params = hydra_instantiate_dictionary(fabricator_params)

    list_of_tables = [
        table for _, list_of_tables in tables_params.items() for table in list_of_tables
    ]

    mock_generator = MockDataGenerator(tables=list_of_tables, seed=seed)

    mock_generator.generate_all()

    output = {
        name: table.dataframe
        for name, table in mock_generator.tables.items()
        if not any(name.startswith(prefix) for prefix in ignore_prefix)
    }

    return output
