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
"""This module provides tests for data fabrication utils"""

import inspect
import re
import sys
from dataclasses import fields

import numpy as np
import pandas as pd
import pytest
from pandas.api.types import is_float_dtype, is_int64_dtype
from pandas.core.dtypes.missing import isna

from data_fabricator.v1.core.functions import hash_string
from data_fabricator.v1.core.mock_generator import (
    BaseColumn,
    BaseTable,
    ColumnApply,
    CrossProduct,
    CrossProductWithSeparator,
    Date,
    DropDuplicates,
    Explode,
    Faker,
    ForeignKey,
    JoinedColumn,
    MockDataGenerator,
    NumpyRandom,
    PrimaryKey,
    RandomArrays,
    RandomNumbers,
    RowApply,
    UniqueId,
    ValuesFromSamples,
)
from data_fabricator.v1.nodes.fabrication import fabricate_datasets
from data_fabricator.v1.nodes.hydra import (
    fabricate_datasets as fabricate_datasets_hydra,
)
from data_fabricator.v1.nodes.hydra import hydra_instantiate_dictionary
from data_fabricator.v1.utils import v0_converter

# pylint: skip-file
# flake8: noqa


def test_generate_unique_id():
    id_class = UniqueId()
    ids = id_class.generate(num_rows=200)

    assert len(ids) == 200


def test_generate_values_from_list():
    input_values = ["a", "b", "c"]
    generate_values_class = ValuesFromSamples(sample_values=input_values)
    values = generate_values_class.generate(num_rows=100)

    assert len(values) == 100
    assert len(set(values)) == len(set(input_values))


def test_random_numbers():
    class TestTable1(BaseTable):
        num_rows = 100
        default_column = RandomNumbers()
        int_column = RandomNumbers(
            start_range=1,
            end_range=10,
            prob_null_kwargs={"prob_null": 0.25},
            dtype=pd.Int64Dtype(),
        )
        test_column3 = RandomNumbers(start_range=1, end_range=10, precision=2)

    mdg = MockDataGenerator(tables=[TestTable1])
    mdg.generate_all()

    default_result = mdg.tables["TestTable1"].dataframe["default_column"]
    int_result = mdg.tables["TestTable1"].dataframe["int_column"]
    result2 = mdg.tables["TestTable1"].dataframe["test_column3"].to_list()

    assert len(default_result) == 100
    assert default_result.max() <= 1.0
    assert is_float_dtype(default_result)
    assert len(int_result) == 100
    assert len(str(result2[0]).split(".")[1]) <= 2
    assert int_result.max() <= 10
    assert is_int64_dtype(int_result)


def test_random_arrays():
    class TestTable1(BaseTable):
        num_rows = 10
        test_column1 = RandomArrays(
            sample_values=[10, 100, 1000], prob_null_kwargs={"seed": 1}
        )

    mdg = MockDataGenerator(tables=[TestTable1])
    mdg.generate_all()

    result0 = mdg.tables["TestTable1"].dataframe["test_column1"].to_list()
    assert result0 == [
        [1000],
        [100],
        [100],
        [100, 1000],
        [10],
        [10, 100],
        [1000, 10],
        [100, 1000, 10],
        [10, 100, 1000],
        [10],
    ]


def test_numpy_random():
    class TestTable1(BaseTable):
        num_rows = 10

        test_column1 = NumpyRandom(
            distribution="binomial", numpy_seed=1, np_random_kwargs={"n": 1, "p": 0.5}
        )
        test_column2 = NumpyRandom(
            distribution="pareto", numpy_seed=1, np_random_kwargs={"a": 1}
        )

    mdg = MockDataGenerator(tables=[TestTable1])
    mdg.generate_all()

    result0 = mdg.tables["TestTable1"].dataframe["test_column1"].to_list()
    result1 = mdg.tables["TestTable1"].dataframe["test_column2"].to_list()
    assert result0 == [0, 1, 0, 0, 0, 0, 0, 0, 0, 1]
    rounded = [round(x, 2) for x in result1]
    assert rounded == [0.72, 2.58, 0.0, 0.43, 0.17, 0.1, 0.23, 0.53, 0.66, 1.17]


def test_joined_column_python():
    class Students(BaseTable):
        num_rows = 10
        _metadata_ = {
            "description": "Student table with enrollment info",
        }
        student_id = UniqueId()
        name = Faker(provider="name")
        enrollment_date = Date(start_dt="2019-01-01", end_dt="2020-12-31", freq="M")

    class Classes(BaseTable):
        num_rows = 20
        student_id = RowApply(
            list_of_values="Students.student_id",
            row_func=lambda x: x,
            resize=True,
            prob_null_kwargs={"seed": 1},
        )
        # works with single keys
        student_name = JoinedColumn(
            selected_column="Students.name",
            other_table_key="Students.student_id",
            this_table_key="Classes.student_id",
        )
        # works with composite keys
        enrollment_date = JoinedColumn(
            selected_column="Students.enrollment_date",
            other_table_key=["Students.student_id", "Students.name"],
            this_table_key=["Classes.student_id", "Classes.student_name"],
        )

    mdg = MockDataGenerator(tables=[Classes, Students])
    mdg.generate_all()
    df1 = mdg.tables["Students"].dataframe
    df2 = mdg.tables["Classes"].dataframe

    for k, i in enumerate(df2.student_id):
        assert (df1.loc[df1.student_id == i].name == (df2.iloc[k].student_name)).all()
        assert (
            df1.loc[df1.student_id == i].enrollment_date
            == (df2.iloc[k].enrollment_date)
        ).all()


def test_row_apply_single_vector():
    categorical_values = ["a", "b", "c"]

    class GeneratedValuesClass(BaseTable):
        num_rows = 100000
        sample_generated_values = ValuesFromSamples(sample_values=categorical_values)

    class TestClass(BaseTable):
        num_rows = 100000
        test_column = RowApply(
            list_of_values="GeneratedValuesClass.sample_generated_values",
            row_func=lambda x: x,
        )

    mdg = MockDataGenerator(tables=[GeneratedValuesClass, TestClass])

    mdg.generate_all()
    initial_values = mdg.tables["GeneratedValuesClass"].dataframe[
        "sample_generated_values"
    ]
    derived_values = mdg.tables["TestClass"].dataframe["test_column"]
    assert len(derived_values) == len(initial_values)


def test_row_apply_multiple_vector():
    class TestTable1(BaseTable):
        num_rows = 200
        ids1 = UniqueId()
        ids2 = UniqueId()

    class TestTable2(BaseTable):
        num_rows = 200
        derived_vals1 = RowApply(
            list_of_values=["TestTable1.ids1", "TestTable1.ids2"],
            row_func=lambda x, y: int(x) + int(y),
        )

    mdg = MockDataGenerator([TestTable2, TestTable1])

    mdg.generate_all()

    derived_vals1 = mdg.tables["TestTable2"].dataframe["derived_vals1"]
    assert len(derived_vals1) == 200
    assert min(derived_vals1) == 2
    assert max(derived_vals1) == 400


def test_row_apply_multiple_vector_fails():
    with pytest.raises(Exception, match="List of Values improperly set"):

        class TestTable3(BaseTable):
            num_rows = 200
            derived_vals2 = RowApply(
                list_of_values=[[1, 2, 3, 4, 5], [1, 3, 5, 7, 11]],
                row_func=lambda x, y: x * y,
                resize=True,
            )

        mdg = MockDataGenerator([TestTable3])
        mdg.generate_all()


def test_explode_single_vector():
    def _dummy(n):
        return [i + 1 for i in range(n)]

    class TestTable1(BaseTable):
        num_rows = 3

        l1 = UniqueId()

    class TestTable2(BaseTable):
        test_column1 = Explode(
            list_of_values="TestTable1.l1",
            explode_func=_dummy,
            explode_func_kwargs={"n": 3},
            position=0,
        )
        test_column2 = Explode(
            list_of_values="TestTable1.l1",
            explode_func=_dummy,
            explode_func_kwargs={"n": 3},
            position=1,
        )

    mdg = MockDataGenerator(tables=[TestTable1, TestTable2])
    mdg.generate_all()

    result0 = mdg.tables["TestTable2"].dataframe["test_column1"].to_list()
    result0 = [int(element) for element in result0]
    result1 = mdg.tables["TestTable2"].dataframe["test_column2"]
    result1 = [int(element) for element in result1]
    assert result0 == [1, 1, 1, 2, 2, 2, 3, 3, 3]
    assert result1 == [1, 2, 3, 1, 2, 3, 1, 2, 3]


def test_explode_multiple_vectors():
    def _dummy(n):
        return [i + 1 for i in range(n)]

    class TestTable1(BaseTable):
        num_rows = 3

        l1 = UniqueId()
        l2 = UniqueId()

    class TestTable2(BaseTable):
        test_column1 = Explode(
            list_of_values=["TestTable1.l1", "TestTable1.l2"],
            explode_func=_dummy,
            explode_func_kwargs={"n": 3},
            position=0,
        )
        test_column2 = Explode(
            list_of_values=["TestTable1.l1", "TestTable1.l2"],
            explode_func=_dummy,
            explode_func_kwargs={"n": 3},
            position=1,
        )
        test_column3 = Explode(
            list_of_values=["TestTable1.l1", "TestTable1.l2"],
            explode_func=_dummy,
            explode_func_kwargs={"n": 3},
            position=2,
        )

    mdg = MockDataGenerator(tables=[TestTable1, TestTable2])
    mdg.generate_all()

    result0 = mdg.tables["TestTable2"].dataframe["test_column1"].to_list()
    result0 = [int(element) for element in result0]
    result1 = mdg.tables["TestTable2"].dataframe["test_column2"]
    result1 = [int(element) for element in result1]
    result2 = mdg.tables["TestTable2"].dataframe["test_column3"]
    result2 = [int(element) for element in result2]

    mdg = MockDataGenerator(tables=[TestTable1, TestTable2])
    mdg.generate_all()

    assert result0 == [1, 1, 1, 2, 2, 2, 3, 3, 3]
    assert result1 == [1, 1, 1, 2, 2, 2, 3, 3, 3]
    assert result2 == [1, 2, 3, 1, 2, 3, 1, 2, 3]


def test_column_apply_single():
    class TestTable1(BaseTable):
        num_rows = 200

        l1 = UniqueId()
        test_column = ColumnApply(
            list_of_values="TestTable1.l1",
            resize=True,
            column_func=lambda x: [int(v) for v in x if int(v) < 100],
        )

    mdg = MockDataGenerator(tables=[TestTable1])
    mdg.generate_all()
    result = mdg.tables["TestTable1"].dataframe["test_column"].to_list()
    for element in result:
        assert element < 100


def test_column_apply_single():
    class TestTable1(BaseTable):
        num_rows: 200

        l1 = UniqueId()
        test_column = ColumnApply(
            list_of_values="TestTable1.l1",
            resize=True,
            column_func=lambda x: [int(v) for v in x if int(v) < 100],
        )

    mdg = MockDataGenerator(tables=[TestTable1])
    mdg.generate_all()
    result = mdg.tables["TestTable1"].dataframe["test_column"].to_list()
    for element in result:
        assert element < 100


def test_column_apply_multiple():
    class TestTable1(BaseTable):
        num_rows = 200

        l1 = UniqueId()
        l2 = UniqueId()
        test_column = ColumnApply(
            list_of_values=["TestTable1.l1", "TestTable1.l2"],
            column_func=lambda x, y: [(int(v1) + int(v2)) for v1, v2 in zip(x, y)],
        )

    mdg = MockDataGenerator(tables=[TestTable1])
    mdg.generate_all()
    result = mdg.tables["TestTable1"].dataframe["test_column"].to_list()
    assert result[0] == 2
    assert result[-1] == 400


def test_cross_product():
    class TestTable1(BaseTable):
        num_rows = 2

        l1 = UniqueId(prefix="q")
        l2 = UniqueId(prefix="r")

    class TestTable2(BaseTable):
        test_column1 = CrossProduct(
            list_of_values=["TestTable1.l1", "TestTable1.l2"], position=0
        )
        test_column2 = CrossProduct(
            list_of_values=["TestTable1.l1", "TestTable1.l2"], position=1
        )

    mdg = MockDataGenerator(tables=[TestTable1, TestTable2])
    mdg.generate_all()

    result1 = mdg.tables["TestTable2"].dataframe["test_column1"].to_list()
    result2 = mdg.tables["TestTable2"].dataframe["test_column2"].to_list()

    assert result1 == ["q1", "q1", "q2", "q2"]
    assert result2 == ["r1", "r2", "r1", "r2"]


def test_cross_product_with_separator():
    """Test on using the API for the cross product with separator."""

    class TestTable1(BaseTable):
        num_rows = 2

        l1 = UniqueId(prefix="q")
        l2 = UniqueId(prefix="r")

    class TestTable2(BaseTable):
        test_column1 = CrossProductWithSeparator(
            list_of_values=["TestTable1.l1", "TestTable1.l2"], separator=" "
        )
        test_column2 = CrossProductWithSeparator(
            list_of_values=["TestTable1.l1", "TestTable1.l2"], separator="_"
        )

    mdg = MockDataGenerator(tables=[TestTable1, TestTable2])
    mdg.generate_all()

    result1 = mdg.tables["TestTable2"].dataframe["test_column1"].to_list()
    result2 = mdg.tables["TestTable2"].dataframe["test_column2"].to_list()

    assert result1 == ["q1 r1", "q1 r2", "q2 r1", "q2 r2"]
    assert result2 == ["q1_r1", "q1_r2", "q2_r1", "q2_r2"]


def test_drop_duplicates():
    class TestTable1(BaseTable):
        num_rows = 5

        l1 = ValuesFromSamples([1, 2, 3, 4], prob_null_kwargs={"seed": 1})
        l2 = ValuesFromSamples([3, 4], prob_null_kwargs={"seed": 1})

    class TestTable2(BaseTable):
        test_column1 = DropDuplicates(["TestTable1.l1", "TestTable1.l2"], position=0)
        test_column2 = DropDuplicates(["TestTable1.l1", "TestTable1.l2"], position=1)
        test_column3 = DropDuplicates("TestTable1.l1", position=0)

    mdg = MockDataGenerator(tables=[TestTable1, TestTable2])
    mdg.generate_all()

    result1 = mdg.tables["TestTable2"].dataframe["test_column1"].to_list()
    result2 = mdg.tables["TestTable2"].dataframe["test_column2"].to_list()
    result3 = mdg.tables["TestTable2"].dataframe["test_column3"].to_list()

    assert result1 == [1, 4, 2]
    assert result2 == [3, 4, 3]
    assert result3 == [1, 4, 2]


def test_primary_key():
    id_class = PrimaryKey()
    ids = id_class.generate(num_rows=200)

    assert len(set(ids)) == 200
    assert id_class._metadata_["primary_key"] == True


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


def test_single_foreign_key():
    class patient(BaseTable):
        num_rows = 10
        patient_id = UniqueId()
        patient_gender = ValuesFromSamples(sample_values=["male", "female", "unknown"])

    class events(BaseTable):
        num_rows = 100
        date = Date(start_dt="2019-01-01", end_dt="2020-12-31", freq="M")
        patient_id = ForeignKey(
            foreign_key_cols=["patient.patient_id"], position=0, resize=True
        )

    mdg = MockDataGenerator(tables=[patient, events])
    mdg.generate_all()

    assert len(set(mdg.tables["events"].dataframe["patient_id"])) == 10
    assert len(mdg.tables["events"].dataframe["patient_id"]) == 100
    assert mdg.tables["events"].dataframe["patient_id"].drop_duplicates().shape[0] == 10
    assert mdg.tables["events"].patient_id._metadata_["foreign_key"] == True
    assert mdg.tables["events"].patient_id._metadata_["foreign_key_of"] == [
        "patient.patient_id"
    ]


def test_python_api_create():
    class Students(BaseTable):
        num_rows = 10
        _metadata_ = {
            "description": "Student table with enrollment info",
        }
        student_id = UniqueId()
        name = Faker(provider="name")
        enrollment_date = Date(start_dt="2019-01-01", end_dt="2020-12-31", freq="M")

    class Faculty(BaseTable):
        num_rows = 5
        _metadata_ = {
            "description": "Faculty info along with departments",
        }
        faculty_id = UniqueId()
        name = Faker(provider="name")
        course = ValuesFromSamples(
            sample_values=["engineering", "computer science", "mathematics"]
        )

    class Classes(BaseTable):
        student_id = RowApply(
            list_of_values="Students.student_id", row_func=lambda x: x
        )
        course = RowApply(
            list_of_values="Faculty.course", row_func=lambda x: x, resize=True
        )

    mdg = MockDataGenerator(tables=[Classes, Faculty, Students])
    mdg.generate_all()
    assert len(mdg.tables["Students"].dataframe) == 10
    assert len(mdg.tables["Faculty"].dataframe) == 5
    assert len(mdg.tables["Classes"].dataframe) == 10


def test_metadata_exists_for_all_col_classes():
    # Get all possible column types
    clsmembers = inspect.getmembers(sys.modules[__name__], inspect.isclass)
    col_classes = [
        cls[1]
        for cls in clsmembers
        if (issubclass(cls[1], BaseColumn) and cls[1] != BaseColumn)
    ]

    classes_wo_metadata = [
        cls
        for cls in col_classes
        if "_metadata_" not in [field.name for field in fields(cls)]
    ]

    assert (
        len(classes_wo_metadata) == 0
    ), f"_metadata_ variable missing in: {classes_wo_metadata}"


def test_probability_null_works():
    class TestTable1(BaseTable):
        num_rows = 10
        l1 = UniqueId(prob_null_kwargs={"prob_null": 0.5})
        l2 = Faker(provider="name", prob_null_kwargs={"prob_null": 0.5})
        l3 = Date(
            start_dt="2023-01-01",
            end_dt="2023-12-31",
            freq="M",
            prob_null_kwargs={"prob_null": 0.5},
        )
        l4 = NumpyRandom(
            distribution="binomial",
            np_random_kwargs={"n": 10, "p": 0.5},
            prob_null_kwargs={"prob_null": 0.5},
        )
        l5 = ValuesFromSamples(
            sample_values=[1, 2, 3], prob_null_kwargs={"prob_null": 0.5}
        )
        l6 = RandomNumbers(prob_null_kwargs={"prob_null": 0.5})

    mdg = MockDataGenerator(tables=[TestTable1])
    mdg.generate_all()

    result = mdg.tables["TestTable1"].dataframe
    mean = result.isna().sum().mean()
    percentage = mean / (10)  # 10 rows ,
    assert percentage > 0.4 and percentage < 0.6


def test_prob_null_kwargs_exists_for_all_col_classes():
    # Get all possible column types
    clsmembers = inspect.getmembers(sys.modules[__name__], inspect.isclass)
    col_classes = [
        cls[1]
        for cls in clsmembers
        if (issubclass(cls[1], BaseColumn) and cls[1] != BaseColumn)
    ]

    classes_wo_metadata = [
        cls
        for cls in col_classes
        if "prob_null_kwargs" not in [field.name for field in fields(cls)]
    ]

    assert (
        len(classes_wo_metadata) == 0
    ), f"prob_null_kwargs variable missing in: {classes_wo_metadata}"


def test_dtypes():
    class TestTable1(BaseTable):
        num_rows = 10
        test1 = ValuesFromSamples(
            sample_values=[x for x in range(100)],
            dtype=np.float64,
        )
        test2 = Date(
            start_dt="2023-01-01",
            end_dt="2023-12-31",
            freq="M",
            prob_null_kwargs={"prob_null": 0.5},
            dtype=np.str_,
        )

    mdg = MockDataGenerator(tables=[TestTable1])
    mdg.generate_all()

    result0 = mdg.tables["TestTable1"].dataframe["test1"]
    result1 = mdg.tables["TestTable1"].dataframe["test2"]
    assert isinstance(result0[0], np.float64)
    assert isinstance(result1[0], str)


def test_dtypes_string_na_error():
    class TestTable1(BaseTable):
        num_rows = 1
        test1 = Date(
            start_dt="2023-01-01",
            end_dt="2023-12-31",
            freq="M",
            prob_null_kwargs={"prob_null": 1},
            dtype=pd.StringDtype(),
        )
        test2 = Date(
            start_dt="2023-01-01",
            end_dt="2023-12-31",
            freq="M",
            prob_null_kwargs={"prob_null": 1},
            dtype=np.str_,
        )

    mdg = MockDataGenerator(tables=[TestTable1])
    mdg.generate_all()

    result0 = mdg.tables["TestTable1"].dataframe["test1"]
    result1 = mdg.tables["TestTable1"].dataframe["test2"]

    # pd.StringDtype() casts pd.NA for null_value into string column
    # but is not considered string
    # same happens for other pd nullable types
    assert isna(result0[0])
    with pytest.raises(AssertionError):
        assert isinstance(result0[0], str)

    # np.str_ converts NA to 'NA' which is string but is not considered na
    assert isinstance(result1[0], str)
    with pytest.raises(AssertionError):
        assert isna(result1[0])


def test_hash_string_with_classes():
    class TestTable1(BaseTable):
        num_rows = 10
        input = ValuesFromSamples([1, 2, "a", "b"])
        test_column1 = RowApply(
            list_of_values="TestTable1.input",
            row_func=hash_string,
            row_func_kwargs={
                "buckets": ["x", "y"],
            },
        )
        test_column2 = RowApply(
            list_of_values="TestTable1.input",
            row_func=hash_string,
            row_func_kwargs={
                "buckets": ["x", "y"],
            },
        )

    mdg = MockDataGenerator(tables=[TestTable1])
    mdg.generate_all()

    input_list = mdg.tables["TestTable1"].dataframe["input"].to_list()
    result0 = mdg.tables["TestTable1"].dataframe["test_column1"].to_list()
    result1 = mdg.tables["TestTable1"].dataframe["test_column2"].to_list()

    tuples = [(x, y) for x, y in zip(input_list, result0)]

    unique_tuples = []
    for tpl in tuples:
        if tpl not in unique_tuples:
            unique_tuples.append(tpl)

    assert len(set(result0)) == 2
    assert len(unique_tuples) == len(set(input_list))
    assert result0 == result1


def test_generate_single_date_with_classes():
    class TestTable1(BaseTable):
        num_rows = 10
        start_date = Date(
            start_dt="2022-01-01",
            end_dt="2022-12-31",
            freq="B",
            prob_null_kwargs={"seed": 1},
        )
        end_date = RowApply(
            list_of_values="TestTable1.start_date",
            row_func="lambda x: generate_single_date(start_dt=x,\
                 end_dt= x + datetime.timedelta(days=14))[0]",
            dtype=np.datetime64,
        )

    mdg = MockDataGenerator(tables=[TestTable1])
    mdg.generate_all()
    df = mdg.tables["TestTable1"].dataframe
    assert ((df["end_date"] - df["start_date"]).dt.days <= 14).all()


def test_filter_rows():
    class TestTable1(BaseTable):
        num_rows = 10
        l1 = Faker(provider="name", prob_null_kwargs={"prob_null": 0.2, "seed": 1})
        l2 = Date(
            start_dt="2023-01-01",
            end_dt="2023-12-31",
            freq="M",
            prob_null_kwargs={"prob_null": 0.2, "seed": 1},
            dtype="string",
        )

    class TestTable2(BaseTable):
        test1 = RowApply(
            list_of_values=["TestTable1.l1", "TestTable1.l2"],
            row_func="lambda x,y:y",
            filter="lambda x, y: x.notna() & y.notna()",
        )

    mdg = MockDataGenerator(tables=[TestTable1, TestTable2])
    mdg.generate_all()
    result = mdg.tables["TestTable2"].dataframe["test1"].to_list()
    assert result == [
        "2023-03-31",
        "2023-04-30",
        "2023-05-31",
        "2023-07-31",
        "2023-09-30",
    ]


def test_fabricate_datasets(load_scenario1, load_scenario3):
    load_scenario1 = hydra_instantiate_dictionary(load_scenario1)
    load_scenario3 = hydra_instantiate_dictionary(load_scenario3)
    fabricated_tables = fabricate_datasets(
        scenario1=load_scenario1, scenario3=load_scenario3
    )
    assert len(fabricated_tables.values()) == 5
    for name in ["patient", "events", "faculty", "students", "classes"]:
        assert name in fabricated_tables.keys()
        assert isinstance(fabricated_tables[name], pd.DataFrame)


def test_v0_converter(load_scenario6):
    tables = v0_converter(load_scenario6)
    tables = hydra_instantiate_dictionary(tables)

    mdg = MockDataGenerator(tables=tables["tables"])
    mdg.generate_all()

    assert len(mdg.tables["accounts"].dataframe) == 10
    assert len(mdg.tables["transaction_statements"].dataframe) == 20


def test_bad_table_definition_num_rows_is_zero():
    class TestTable1(BaseTable):
        l1 = Faker(provider="name", prob_null_kwargs={"prob_null": 0.2, "seed": 1})

    class TestTable2(BaseTable):
        test1 = RowApply(
            list_of_values=["TestTable1.l1"],
            row_func="lambda x:x",
            filter="lambda x: x.notna()",
        )

    mdg = MockDataGenerator(tables=[TestTable1, TestTable2])

    with pytest.raises(
        Exception,
        match=re.escape(
            "There are problems with your table definition(s): {'TestTable1': [\"This table doesn't have any dependencies, set num_rows!\"]}"
        ),
    ):
        mdg.generate_all()


def test_bad_table_definition_column_nonexistant():
    class TestTable1(BaseTable):
        num_rows = 2
        l1 = Faker(provider="name", prob_null_kwargs={"prob_null": 0.2, "seed": 1})

    class TestTable2(BaseTable):
        test1 = RowApply(
            list_of_values=["TestTable1.l_not_a_column"],
            row_func="lambda x:x",
            filter="lambda x: x.notna()",
        )

    mdg = MockDataGenerator(tables=[TestTable1, TestTable2])

    with pytest.raises(
        Exception,
        match=re.escape(
            "There are problems with your table definition(s): "
            "{'TestTable2': ['Column test1 is dependent on TestTable1.l_not_a_column, "
            "but cannot find l_not_a_column in TestTable1.']}"
        ),
    ):
        mdg.generate_all()


def test_bad_table_definition_all_errors():
    class TestTable1(BaseTable):
        l1 = Faker(provider="name", prob_null_kwargs={"prob_null": 0.2, "seed": 1})

    class TestTable2(BaseTable):
        test1 = RowApply(
            list_of_values=["TestTable1.l_not_a_column"],
            row_func="lambda x:x",
            filter="lambda x: x.notna()",
        )

    mdg = MockDataGenerator(tables=[TestTable1, TestTable2])

    with pytest.raises(
        Exception,
        match=re.escape(
            "There are problems with your table definition(s): {'TestTable1': "
            '["This table doesn\'t have any dependencies, set num_rows!"], '
            "'TestTable2': ['Column test1 is dependent on TestTable1.l_not_a_column, "
            "but cannot find l_not_a_column in TestTable1.']}"
        ),
    ):
        mdg.generate_all()
