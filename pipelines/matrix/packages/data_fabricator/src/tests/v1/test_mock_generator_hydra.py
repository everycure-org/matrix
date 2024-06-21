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
"""This module provides tests for data fabrication utils with hydra."""

# pylint: skip-file
# flake8: noqa

import pandas as pd
import pytest

pytest.importorskip("hydra")
from data_fabricator.v1.core.mock_generator import MockDataGenerator
from data_fabricator.v1.nodes.hydra import (
    fabricate_datasets as fabricate_datasets_hydra,
)
from data_fabricator.v1.nodes.hydra import hydra_instantiate_dictionary


def test_foreign_key(load_scenario3):
    load_scenario3 = [
        hydra_instantiate_dictionary(scenario) for scenario in load_scenario3
    ]
    Patient, Events = load_scenario3

    mdg = MockDataGenerator(tables=[Patient, Events])
    mdg.generate_all()

    assert len(set(mdg.tables["events"].dataframe["patient_id"])) == 10
    assert len(mdg.tables["events"].dataframe["patient_id"]) == 100
    # check if foreign keys equal origin table
    col_selection = ["patient_id", "patient_gender"]
    assert (
        mdg.tables["patient"]
        .dataframe.drop_duplicates()
        .sort_values(col_selection)
        .reset_index()[col_selection]
    ).equals(
        (
            mdg.tables["events"]
            .dataframe[col_selection]
            .drop_duplicates()
            .sort_values(col_selection)
            .reset_index()[col_selection]
        )
    )
    assert mdg.tables["events"].patient_id._metadata_["foreign_key"] == True
    assert mdg.tables["events"].patient_id._metadata_["foreign_key_of"] == [
        "patient.patient_id",
        "patient.patient_gender",
    ]
    assert (
        mdg.tables["events"].patient_gender._metadata_["description"]
        == "Patient gender has much what is defined in the patient table."
    )


def test_python_yaml_create(load_scenario1):
    load_scenario1 = [
        hydra_instantiate_dictionary(scenario) for scenario in load_scenario1
    ]
    Students, Faculty, Classes = load_scenario1

    mdg = MockDataGenerator(tables=[Classes, Faculty, Students])
    mdg.generate_all()
    assert len(mdg.tables["students"].dataframe) == 10
    assert len(mdg.tables["faculty"].dataframe) == 5
    assert len(mdg.tables["classes"].dataframe) == 10


def test_python_yaml_create_single_file(load_scenario2):
    tables = [hydra_instantiate_dictionary(scenario) for scenario in load_scenario2]

    mdg = MockDataGenerator(tables=tables)
    mdg.generate_all()
    assert len(mdg.tables["students"].dataframe) == 10
    assert len(mdg.tables["faculty"].dataframe) == 5
    assert len(mdg.tables["classes"].dataframe) == 10


def test_python_yaml_create_column_metadata(load_scenario1):
    load_scenario1 = [
        hydra_instantiate_dictionary(scenario) for scenario in load_scenario1
    ]
    Students, Faculty, Classes = load_scenario1

    mdg = MockDataGenerator(tables=[Classes, Faculty, Students])
    mdg.generate_all()

    assert mdg.tables["classes"].student_id._metadata_["foreign_key"] == True
    assert mdg.tables["students"].student_id._metadata_["primary_key"] == True
    assert mdg.tables["faculty"].faculty_id._metadata_["primary_key"] == True


@pytest.mark.xfail  # unable to replicate locally anymore
def test_dtypes_yaml_pdint64dtype_error(load_scenario4):
    load_scenario4 = [
        hydra_instantiate_dictionary(scenario) for scenario in load_scenario4
    ]

    with pytest.raises(
        ValueError, match=r"The original error is int\(\) argument must be a string.+"
    ):
        mdg = MockDataGenerator(tables=load_scenario4)
        mdg.generate_all()


def test_conditional_values_with_classes(load_scenario5):
    stage3_sales, _ = [
        hydra_instantiate_dictionary(scenario) for scenario in load_scenario5
    ]

    mdg = MockDataGenerator(tables=[stage3_sales])
    mdg.generate_all()
    df1 = mdg.tables["stage3_sales"].dataframe
    assert sorted(df1.loc[df1["product_cd"] == "R03AC02"]["channel_cd"].unique()) == [
        "a",
        "b",
    ]
    assert sorted(df1.loc[df1["product_cd"] == "R03AC03"]["channel_cd"].unique()) == [
        "c",
        "d",
    ]


def test_conditional_string_with_classes(load_scenario5):
    scenario5 = [hydra_instantiate_dictionary(scenario) for scenario in load_scenario5]

    mdg = MockDataGenerator(tables=scenario5)
    mdg.generate_all()
    df2 = mdg.tables["channel_concepts"].dataframe

    assert sorted(df2["channel_description"].unique()) == [
        "alpha",
        "bravo",
        "charlie",
        "delta",
    ]


def test_fabricate_datasets_hydra(load_scenario1, load_scenario3):
    fabricated_tables = fabricate_datasets_hydra(
        scenario1=load_scenario1, scenario3=load_scenario3
    )
    assert len(fabricated_tables.values()) == 5
    for name in ["patient", "events", "faculty", "students", "classes"]:
        assert name in fabricated_tables.keys()
        assert isinstance(fabricated_tables[name], pd.DataFrame)
