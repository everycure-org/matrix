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
"""converter for v0 config files supporting standard v0 API."""

import copy
import inspect
import re
from typing import Dict, List

import data_fabricator.v1.core.functions
import data_fabricator.v1.core.mock_generator
from data_fabricator.v1.core.mock_generator import BaseTable


def _get_map_functions_to_classes():
    """Get all existent mapping from functions to column classes in V1."""
    clsmembers = inspect.getmembers(
        data_fabricator.v1.core.mock_generator, inspect.isclass
    )
    col_classes = [
        cls[1]
        for cls in clsmembers
        if (
            issubclass(cls[1], data_fabricator.v1.core.mock_generator.BaseColumn)
            and cls[1] != data_fabricator.v1.core.mock_generator.BaseColumn
        )
    ]

    map_func_to_col = {}

    for col in col_classes:
        source = inspect.getsource(col)
        match = re.findall(r"\b\w+(?=\(\n* *num_rows)", source)
        if match:
            map_func_to_col[match[-1]] = str(col).split("'")[1]

    return map_func_to_col


_CLASSES_MAP = _get_map_functions_to_classes()


# flake8: noqa: C901
def _process_columns(columns_dict: Dict[str, str]):
    """Process columns dictionary."""
    new_dict = {}
    for col_key, col_val in columns_dict.items():
        new_col_val = {}
        function_name = col_val.pop("type")

        # changes type for object using classes map
        new_col_val["_target_"] = _CLASSES_MAP[function_name]
        # changes probability_null kwargs for v1 compatible
        for param in ("prob_null", "null_value", "seed"):
            if param in col_val:
                if "prob_null_kwargs" not in new_col_val:
                    new_col_val["prob_null_kwargs"] = {}
                new_col_val["prob_null_kwargs"][param] = col_val.pop(param)
        # special np_random case
        if function_name == "numpy_random":
            new_col_val["distribution"] = col_val.pop("distribution")
            if "numpy_seed" in col_val:
                new_col_val["numpy_seed"] = col_val.pop("numpy_seed")
            new_col_val["np_random_kwargs"] = col_val.copy()
            col_val = {}

        # update dtype if v0 integer flag is defined:
        if "integer" in col_val and col_val["integer"]:
            col_val.pop("integer")
            col_val.pop("dtype", None)
            new_col_val["dtype"] = "Int64"
        # update dtype explicitly for integer sample values
        if function_name == "generate_values":
            if "dtype" not in col_val and all(
                isinstance(val, int) for val in col_val["sample_values"]
            ):
                new_col_val["dtype"] = "Int64"
        # populate remaining params
        for param in col_val:
            if re.match(r".+_func$", param):
                # changes string functions to function instances
                new_col_val[param] = (
                    col_val[param]
                    if "lambda" in col_val[param]
                    else {
                        "_target_": "data_fabricator.v1.core.functions."
                        + col_val[param],
                        "_partial_": True,
                    }
                )
            else:
                new_col_val[param] = col_val[param]

        new_dict[col_key] = new_col_val

    return new_dict


def v0_converter(yaml_dict: Dict[str, str]) -> Dict[str, List[BaseTable]]:
    """Convert v0 config yaml parsed dictionary to v1 data_fabricator."""
    yaml_dict_copy = copy.deepcopy(yaml_dict)
    converted_dict = {"tables": []}

    for key, value in yaml_dict_copy.items():
        if not isinstance(value, Dict):
            # for other kwargs not tables config
            converted_dict[key] = value
            continue
        # set parameters object and name.
        current_table = {}
        current_table[
            "_target_"
        ] = "data_fabricator.v1.core.mock_generator.create_table"
        current_table["name"] = key
        current_table.update(value)
        current_table["columns"] = _process_columns(
            copy.deepcopy(current_table["columns"])
        )
        converted_dict["tables"].append(copy.deepcopy(current_table))

    return converted_dict
