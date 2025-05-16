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
"""Fixtures that loads yaml files required for tests."""
from pathlib import Path

import pytest
import yaml


def concatenate_yaml_tables(list_yaml):
    """Concatenate list of yaml files with table generation instructions.

    Reads yaml files and returns
    dictionary objects of object and tables
    """
    list_of_yaml = []
    for filepath in list_yaml:
        with open(Path(__file__).parent / filepath, "r", encoding="utf-8") as file:
            content = yaml.safe_load(file)
            for table in content["tables"]:
                list_of_yaml.append(table)
    return list_of_yaml


@pytest.fixture
def load_scenario1():
    """Load yaml files for test scenario 1.

    Reads yaml files and returns
    dictionary objects as a list
    """
    list_of_yaml = [
        "scenarios/scenario1/student.yml",
        "scenarios/scenario1/faculty.yml",
        "scenarios/scenario1/classes.yml",
    ]
    return concatenate_yaml_tables(list_of_yaml)


@pytest.fixture
def load_scenario2():
    """Load yaml files for test scenario 2.

    Reads yaml files and returns
    dictionary objects as a list
    """
    list_of_yaml = [
        "scenarios/scenario2/single_yaml_config.yml",
    ]

    return concatenate_yaml_tables(list_of_yaml)


@pytest.fixture
def load_scenario3():
    """Load yaml files for test scenario 3.

    Reads yaml files and returns
    dictionary objects as a list
    """
    list_of_yaml = [
        "scenarios/scenario3/scenario3.yml",
    ]

    return concatenate_yaml_tables(list_of_yaml)


@pytest.fixture
def load_scenario4():
    """Load yaml files for test scenario 4.

    Reads yaml files and returns
    dictionary objects as a list
    """
    list_of_yaml = [
        "scenarios/scenario4/scenario4.yml",
    ]

    return concatenate_yaml_tables(list_of_yaml)


@pytest.fixture
def load_scenario5():
    """Load yaml files for test scenario 5.

    Reads yaml files and returns
    dictionary objects as a list
    """
    list_of_yaml = [
        "scenarios/scenario5/scenario5.yml",
    ]

    return concatenate_yaml_tables(list_of_yaml)


@pytest.fixture
def load_scenario6():
    """Load yaml files for test scenario 6.

    Reads yaml files and returns
    dictionary objects as a list
    """
    filepath = "scenarios/scenario6/scenario6.yml"

    with open(Path(__file__).parent / filepath, "r", encoding="utf-8") as file:
        content = yaml.safe_load(file)

    return content
