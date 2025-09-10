# (c) McKinsey & Company 2016 – Present
# All rights reserved
#
#
# This material is intended solely for your internal use and may not be reproduced,
# disclosed or distributed without McKinsey & Company's express prior written consent.
# Except as otherwise stated, the Deliverables are provided 'as is', without any express
# or implied warranty, and McKinsey shall not be obligated to maintain, support, host,
# update, or correct the Deliverables. Client guarantees that McKinsey's use of
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
"""This module provides tests for fabrication node."""
import pandas as pd
import yaml
import pyspark.sql as ps

from data_fabricator.v0.nodes.fabrication import fabricate_datasets

# pylint: skip-file
# flake8: noqa


def test_param_dataset_only():
    string = """
    customers:
        num_rows: 200
        columns:
            hcp_id:
                type: generate_unique_id
                prefix: hcp
                id_start_range: 1
                id_end_range: 201
            hcp_name:
                type: faker
                provider: name
            hcp_company:
                type: faker
                provider: company
            hcp_phone_number:
                type: faker
                provider: phone_number
                localisation: ja-JP
            ftr1:
                type: generate_random_numbers
                start_range: 0
                end_range: 1
                prob_null: 0.5
            ftr2:
                type: generate_random_numbers
                start_range: 1
                end_range: 100
                integer: True
            ftr3:
                type: generate_random_numbers
                start_range: 200
                end_range: 1000
                integer: True
            ftr4:
                type: generate_random_numbers
    """

    parsed_dict = yaml.safe_load(string)

    output_dict = fabricate_datasets(parsed_dict)

    assert output_dict["customers"].shape == (200, 8)
    assert output_dict["customers"]["hcp_id"].nunique() == 200
    assert len(output_dict.keys()) == 1


def test_param_dfs_provided():
    string = """
    customers:
        num_rows: 10
        columns:
            hcp_id:
                type: generate_unique_id
                prefix: hcp
                id_start_range: 1
                id_end_range: 201
            trigger_ids:
                type: generate_random_arrays
                sample_values: [1, 2, 3, 4]
                allow_duplicates: False
                length: 2
            ftr1:
                type: generate_random_numbers
                start_range: 0
                end_range: 1
                prob_null: 0.5
            ftr_df:
                type: row_apply
                list_of_values: input_pd.pd_input_col  # refers to injected dataframe
                row_func: "lambda x: x"
                resize: True
                seed: 1
            ftr_pyspark:
                type: row_apply
                list_of_values: input_pyspark.pyspark_input_col  # refers to injected dataframe
                row_func: "lambda x: x"
                resize: True
                seed: 1
    """

    parsed_dict = yaml.safe_load(string)

    spark = ps.SparkSession.builder.getOrCreate()

    dfs = {
        "input_pd": pd.DataFrame(data={"pd_input_col": ["one", "two"]}),
        "input_pyspark": spark.createDataFrame(
            data=pd.DataFrame(data={"pyspark_input_col": [1, 2, 3]})
        ),
    }

    output_dict = fabricate_datasets(parsed_dict, **dfs)

    assert len(output_dict.keys()) == 1
    assert output_dict["customers"]["ftr_df"].nunique() == 2
    assert output_dict["customers"]["ftr_pyspark"].nunique() == 3
    for i in range(10):
        assert len(output_dict["customers"]["trigger_ids"][i]) == 2
    assert output_dict["customers"]["trigger_ids"].dtype == "object"


def test_exclude_dataset_default():
    string = """
    _TEMP_customers:
        num_rows: 10
        columns:
            hcp_id:
                type: generate_unique_id
                prefix: hcp
                id_start_range: 1
                id_end_range: 201
            ftr1:
                type: generate_random_numbers
                start_range: 0
                end_range: 1
                prob_null: 0.5
    """

    parsed_dict = yaml.safe_load(string)

    output_dict = fabricate_datasets(parsed_dict)

    assert len(output_dict.keys()) == 0


def test_exclude_dataset_custom():
    string = """
    EXCLUDE_ME_customers:
        num_rows: 10
        columns:
            hcp_id:
                type: generate_unique_id
                prefix: hcp
                id_start_range: 1
                id_end_range: 201
            ftr1:
                type: generate_random_numbers
                start_range: 0
                end_range: 1
                prob_null: 0.5
    """

    parsed_dict = yaml.safe_load(string)

    output_dict = fabricate_datasets(parsed_dict, ignore_prefix=["EXCLUDE_ME"])

    assert len(output_dict.keys()) == 0
