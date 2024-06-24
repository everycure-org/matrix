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
"""Functions to generate fake data declaratively."""
# pylint: disable=too-many-lines
# pylint: disable=attribute-defined-outside-init

from __future__ import annotations

import datetime
import logging
import random
import re
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field, make_dataclass
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from data_fabricator.v1.core.functions import (
    column_apply,
    cross_product,
    cross_product_with_separator,
    drop_duplicates,
    explode,
    faker,
    generate_dates,
    generate_random_arrays,
    generate_random_numbers,
    generate_unique_id,
    generate_values,
    join_column,
    load_function_if_string,
    numpy_random,
    row_apply,
)

logger = logging.getLogger(__name__)


LIST_OF_KNOWN_INT_DTYPES = ["Int64", pd.Int64Dtype()]


@dataclass
class BaseColumn(ABC):
    """Column base class.

    Each Column class must contain a generate method.
    """

    omit_list = {
        "omit_list",
        "_metadata_",
        "column_references",
        "prob_null_kwargs",
        "dtype",
        "filter",
    }

    def __post_init__(self):
        """Process class after init."""
        self._preprocess_dependencies()

    @abstractmethod
    def _generate(self, num_rows=0):
        """Implement the generator for this Column."""

    def generate(self, num_rows=0):
        """Generate column data.

        Args:
            num_rows: Number of rows that need to be generated for this Column

        """
        result = pd.Series(self._generate(num_rows))
        if self.dtype:
            result = result.astype(self.dtype)
        return result

    @property
    def has_dependencies(self):
        """Property getter to check if column has dependencies."""
        return self._metadata_["dependencies"]

    def _preprocess_dependencies(self):
        """Evaluate existence of dependencies in list of values.

        If dependencies exist then do split the strings beforehand and put
        them in correct format for parsing references.
        """
        if "list_of_values" not in asdict(self):
            self._metadata_["dependencies"] = False
        else:
            self._metadata_["dependencies"] = True
            list_of_values = getattr(self, "list_of_values")
            # options are:
            # case 1 - string reference "table.column"

            if isinstance(list_of_values, str):
                list_of_values = [list_of_values]

            # case 2 - list of references : ["t1.col1", "t2.col2",.. ]
            # if logic from v0
            if (
                isinstance(list_of_values, list)
                and isinstance(list_of_values[0], str)
                # match table.name or tab_le.na_me or _Tab_le.name or _TAB_le1.Nam
                and re.match("^[A-Za-z0-9_]+\\.[A-Za-z0-9_]+$", list_of_values[0])
            ):
                self.column_references = [
                    reference.split(".") for reference in list_of_values
                ]

            # case 3 - Not implemented
            else:
                raise AttributeError("List of Values improperly set")

    def add_to_omit_list(self, values_to_add: List[str]):
        """Add value to omit list.

        Args:
            values_to_add (List[str]): List of values to add.
        """
        self.omit_list = self.omit_list.union(set(values_to_add))

    def get_omit_list(self):
        """Return the full omit list.

        Returns:
            List[str]: Returns full omit list.
        """
        return self.omit_list

    def get_instructions(self):
        """Return all instruction variables, excluding metadata."""
        omit_list = self.get_omit_list()
        return {k: v for k, v in self.__dict__.items() if k not in omit_list}

    @staticmethod
    def filter_rows(condition: Callable, columns: List[List[Any]]):
        """Filter columns given a function that returns a mask over array values.

        Uses pandas to filter  more than one column using a condition function
        that returns a mask.

        Args:
            condition: function with conditions
            columns: list of columns (ex. list_of_values)
        """
        # build a dataframe with list of columns
        df = pd.DataFrame({f"{k}": v for k, v in enumerate(columns)})
        # apply filter with dataframe loc function
        filtered_df = df.loc[condition(*[df[c] for c in df])]
        # return the filtered lists
        return [filtered_df[c].to_list() for c in filtered_df]


@dataclass
class UniqueId(BaseColumn):
    """Generate a list of unique integer ids with ``prefix`` prefix.

    Args:
        num_rows: Number of rows to generate
        prefix: prefix to use for your ids. Defaults to empty string "".
        id_start_range: The starting range of the id. Defaults to 1.
        id_end_range: The ending range of the id. Defaults to ``num_rows``.
        id_length: The length of the id. Pads "0" to the left, i.e. an id ``1``, with
            ``id_length`` gives ``0001``.
        _metadata_: additional information about column.
        prob_null_kwargs: arguments for probability_null function wrapper including
            "prob_null" from 0 to 1 , "null_value", and "seed".
        dtype: numpy dtype for the column output.
    """

    prefix: str = ""
    id_start_range: int = 1
    id_end_range: int = None
    id_length: int = None
    _metadata_: Dict[str, str] = field(default_factory=dict)
    prob_null_kwargs: Dict[str, Any] = field(default_factory=dict)
    dtype: type = None

    def _generate(self, num_rows=0):
        """Run the generator for this Column.

        Args:
            num_rows: Number of rows that need to be generated for this Column.

        Returns:
            A list of generated ids.

        Raises:
            ValueError: if ``id_length`` is lower than the length of the ``prefix``
                combined with the length of ``id_end_range``.
        """
        result = generate_unique_id(
            num_rows, **self.get_instructions(), **self.prob_null_kwargs
        )

        return result


@dataclass
class Faker(BaseColumn):
    """Thin wrapper for accessing Faker properties.

    See:
        https://faker.readthedocs.io/en/master/ for more details.

    Args:
        num_rows: Number of rows to generate.
        provider: The Faker provider to access. See:
            https://faker.readthedocs.io/en/stable/providers.html for more details.
        localisation: Language to generate data in. See:
            https://faker.readthedocs.io/en/stable/locales.html for more details.
        provider_args: Arguments for the Faker provider. Refer to the documentation for
            specific Faker providers.
        faker_seed: Set seed for reproducible results.
        _metadata_: additional information about column.
        prob_null_kwargs: arguments for probability_null function wrapper including
            "prob_null" from 0 to 1 , "null_value", and "seed".
        dtype: numpy dtype for the column output.
    """

    provider: str
    provider_args: Dict[str, Any] = field(default_factory=dict)
    localisation: str = "en_US"
    faker_seed: int = 0
    _metadata_: Dict[str, str] = field(default_factory=dict)
    prob_null_kwargs: Dict[str, Any] = field(default_factory=dict)
    dtype: type = None

    def _generate(self, num_rows=0):
        """Generate fake data using the faker package.

        See:
        https://faker.readthedocs.io/en/master/ for more details.

        Args:
            num_rows: Number of rows that need to be generated for this Column

        Returns:
            A new list of values.
        """
        result = faker(num_rows, **self.get_instructions(), **self.prob_null_kwargs)

        return result


@dataclass
class RandomArrays(BaseColumn):
    """Generate random array with sample values.

    Random arrays can either be fixed length or variable length with or without
    duplicate values.

    Args:
        num_rows: The number of rows to generate.
        sample_values: Sample values to use for populating array with.
        allow_duplicates: Whether to allow duplicate values in array. Defaults to False.
        length: Specify fixed length for arrays. Defaults to None.
    """

    sample_values: List[Any]
    allow_duplicates: bool = False
    length: int = None
    _metadata_: Dict[str, str] = field(default_factory=dict)
    prob_null_kwargs: Dict[str, Any] = field(default_factory=dict)
    dtype: type = None

    def _generate(self, num_rows=0):
        """Generate list of random arrays.

        Args:
            num_rows: Number of rows that need to be generated for this Column

        Returns:
            A list of lists.
        """
        result = generate_random_arrays(
            num_rows, **self.get_instructions(), **self.prob_null_kwargs
        )

        return result


@dataclass
class Date(BaseColumn):
    """Create an object that calls generate_dates and hold its attributes.

    Args:
        num_rows: Number of rows that need to be generated for this Column
        start_dt: The start date of the date range.
        end_dt: The end date of the date range.
        freq: Frequency. See
            https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases
            # pylint: disable=line-too-long    # noqa: E501
        num_rows: Number of rows to generate
        sort_dates : sort dates. This is useful when creating master records for a
            dimension table. Not all dates/values should be sequential for pseudo-real data
            For example: Join date of clients can be sequential, since their IDs would be
            sequential as well. But the close dates of a list of accounts can be entirely
            random.
        date_format: Used to format the dates according to a specific format. If set,
            the output type of the Column corresponds to string.
        _metadata_: additional information about column.
        prob_null_kwargs: arguments for probability_null function wrapper including
            "prob_null" from 0 to 1 , "null_value", and "seed".
        dtype: numpy dtype for the column output.
    """

    start_dt: Union[str, datetime.datetime]
    end_dt: Union[str, datetime.datetime]
    freq: str
    sort_dates: bool = True
    date_format: str = None
    _metadata_: Dict[str, str] = field(default_factory=dict)
    prob_null_kwargs: Dict[str, Any] = field(default_factory=dict)
    dtype: type = None

    def _generate(self, num_rows=0):
        """Generate a range of dates.

        Args:
            num_rows: Number of rows that need to be generated for this Column
        """
        result = generate_dates(
            num_rows=num_rows, **self.get_instructions(), **self.prob_null_kwargs
        )
        return result


@dataclass
class ValuesFromSamples(BaseColumn):
    """Generate a list of values given sample values.

    Args:
        num_rows: The number of rows to generate.
        sample_values: A list of values or distribution of values.
        sort_values : sort values. This is useful when creating master records for a
        dimension table. Not all dates/values should be sequential for pseudo-real data
        For example, Join date of clients can be sequential, since their IDs would be
        sequential as well. But the close dates of a list of accounts can be entirely
        random
        _metadata_: additional information about column.
        prob_null_kwargs: arguments for probability_null function wrapper including
            "prob_null" from 0 to 1 , "null_value", and "seed".
        dtype: pandas or numpy dtype for the column output. Note: default integer type
            `int` isn't nullable in pandas. To ensure the column outputs as int with
            nulls, specify dtype as 'Int64', or pd.Int64Dtype() - the pandas nullable
            int type.
    """

    sample_values: Union[List[Any], Dict[str, int]]
    sort_values: bool = True
    _metadata_: Dict[str, str] = field(default_factory=dict)
    prob_null_kwargs: Dict[str, Any] = field(default_factory=dict)
    dtype: type = None

    def _generate(self, num_rows=0):
        """Generate values from passed list.

        Args:
            num_rows: Number of rows that need to be generated for this Column.

        Returns:
            A list of values.

        Raises:
            TypeError: If ``sample_values`` is not of type list or dict.
        """
        result = generate_values(
            num_rows=num_rows, **self.get_instructions(), **self.prob_null_kwargs
        )
        return result


@dataclass
class RandomNumbers(BaseColumn):
    """Generate random numbers between a specified range.

    Args:
        num_rows: The number of rows to generate.
        start_range: The starting range. Defaults to 0.
        end_range: The ending range. Defaults to 1.
        precision: nth decimal place to round a float number. Defaults to None.
        _metadata_: additional information about column.
        prob_null_kwargs: arguments for probability_null function wrapper including
            "prob_null" from 0 to 1 , "null_value", and "seed".
        dtype: pandas or numpy dtype for the column output. Note, default integer type
            `int` isn't nullable in pandas. To ensure the column outputs as int with
            nulls, specify dtype as 'Int64', or pd.Int64Dtype() - the pandas nullable
            int type.
    """

    start_range: int = 0
    end_range: int = 1
    precision: int = None
    _metadata_: Dict[str, str] = field(default_factory=dict)
    prob_null_kwargs: Dict[str, Any] = field(default_factory=dict)
    dtype: type = None

    def _generate(self, num_rows=0):
        """Run generate_random_numbers with the class parameters.

        Args:
            num_rows: Number of rows that need to be generated for this Column.

        Returns:
            A list of numbers of either float or int.
        """
        floored = self.dtype in LIST_OF_KNOWN_INT_DTYPES
        result = generate_random_numbers(
            num_rows=num_rows,
            floored=floored,
            **self.get_instructions(),
            **self.prob_null_kwargs,
        )
        return result


@dataclass
class NumpyRandom(BaseColumn):
    """Create object that calls numpy_random and generate values in a distribution.

    Args:
        distribution: statistical distribution to be used
        numpy_seed: random seed to be used
        np_random_kwargs: dictionary with attributes to be passed to the np.random
            function.
        _metadata_: additional information about column.
        prob_null_kwargs: arguments for probability_null function wrapper including
            "prob_null" from 0 to 1 , "null_value", and "seed".
        dtype: numpy dtype for the column output.

    See: https://numpy.org/doc/1.18/reference/random/generator.html#distributions
    """

    distribution: str
    numpy_seed: int = None
    np_random_kwargs: Dict[str, str] = field(default_factory=dict)
    _metadata_: Dict[str, str] = field(default_factory=dict)
    prob_null_kwargs: Dict[str, Any] = field(default_factory=dict)
    dtype: type = None

    def _generate(self, num_rows=0):
        """Runs numpy_random with the class parameters.

        Args:
            num_rows: Number of rows that need to be generated for this Column

        Returns:
            the generated list of random numbers.
        """
        result = numpy_random(
            num_rows=num_rows,
            distribution=self.distribution,
            numpy_seed=self.numpy_seed,
            **self.np_random_kwargs,
            **self.prob_null_kwargs,
        )
        return result


@dataclass
class JoinedColumn(BaseColumn):
    """Bring data from a column of another table based on key relationship.

    Args:
        selected_column: name of the columns to bring on the other table matching
            the pattern "table_name.column_name".
        this_table_key:  string or list containing the key(s) in this table used
            in the merge. ex.: "other_table.col_key_name".
        other_table_key: string or list containing the key(s) in the other table
            used in the merge. ex: "this_table.col_key_name".
        _metadata_: additional information about column.
        prob_null_kwargs: arguments for probability_null function wrapper including
            "prob_null" from 0 to 1 , "null_value", and "seed".
        dtype: numpy dtype for the column output.
        filter: lambda function with condition to filter rows, ex.
            lambda x,y: x > y or lambda x,y: x.notna() & y.notna()
            obs. use logical & to chain conditions.
    """

    selected_column: str
    this_table_key: Union[str, List[str]]
    other_table_key: Union[str, List[str]]
    _metadata_: Dict[str, str] = field(default_factory=dict)
    prob_null_kwargs: Dict[str, Any] = field(default_factory=dict)
    dtype: type = None
    filter: Union[str, Callable] = None
    # Required for init, to be reset post_init
    list_of_values: List[str] = field(default=List, init=False)

    def __post_init__(self):
        """Process class after init."""
        # put the key of the from_table and the selected_column in
        # place to have dependencies preprocessed.
        if isinstance(self.this_table_key, str):
            self.this_table_key = [self.this_table_key]
        if isinstance(self.other_table_key, str):
            self.other_table_key = [self.other_table_key]
        self.column_names = (
            self.this_table_key + self.other_table_key + [self.selected_column]
        )
        self.list_of_values = self.column_names
        super().__post_init__()

    def _generate(self, num_rows: int = 0):
        """Generate the joined column based on the key relationship.

        Args:
            num_rows: Number of rows that need to be generated for this column

        Returns:
            list with column values mapped from the key relationship.
        """
        result = join_column(
            selected_column=self.selected_column,
            this_table_key=self.this_table_key,
            other_table_key=self.other_table_key,
            column_names=self.column_names,
            list_of_values=self.list_of_values,
        )
        return result


@dataclass
class RowApply(BaseColumn):
    """Create an object that call row_apply and hold its attributes.

    ::

        l1 = [1,2,3]
        l2 = [2,3,4]
        list_of_values = [l1, l2]
        row_func: lambda x, y: x+y

        # row function is applied element wise
        intermediate_result = [row_function(*tpl) for tpl in zip(*list_of_values)]

        # returns
        [
            1+2,
            2+3,
            3+4,
        ]

    Args:
        list_of_values: The list containing elements.
        row_func: The function to be applied to each element.
        row_func_kwargs: Additional kwargs.
        resize: Whether to resize the result. Defaults to False.
        _metadata_: additional information about column.
        prob_null_kwargs: arguments for probability_null function wrapper including
            "prob_null" from 0 to 1 , "null_value", and "seed".
        dtype: numpy dtype for the column output.
        filter: lambda function with condition to filter rows, ex:
            lambda x,y: x > y or lambda x,y: x.notna() & y.notna()
            obs. use logical & to chain conditions.
    """

    list_of_values: Union[str, List[Any]]
    row_func: Union[str, Callable]
    row_func_kwargs: Dict[str, str] = None
    resize: bool = False
    _metadata_: Dict[str, str] = field(default_factory=dict)
    prob_null_kwargs: Dict[str, Any] = field(default_factory=dict)
    dtype: type = None
    filter: Union[str, Callable] = None

    def _generate(self, num_rows: int = 0):
        """Run the row_apply function on the passed list of elements.

        Args:
            num_rows: Number of rows that need to be generated for this column

        Returns:
            original list of values passed throw row_apply.
        """
        result = row_apply(
            num_rows=num_rows, **self.get_instructions(), **self.prob_null_kwargs
        )
        return result


@dataclass
class ForeignKey(RowApply):
    """Convenience class for foreign-key type relationships.

    Args:
        foreign_key_cols: List of references to columns that
        are going to be used as foreign keys.
        position: The nth-position value to be returned.
    """

    foreign_key_cols: List[str] = field(default_factory=list)
    position: int = field(default=0)
    row_func_kwargs: Dict[str, str] = field(default_factory=dict)

    # Required for init, to be reset post_init
    list_of_values: List[str] = field(default=List, init=False)
    row_func: Callable = field(default=Callable, init=False)

    def __post_init__(self):
        """Process class after init."""
        # required for dependency resolution
        self.list_of_values = self.foreign_key_cols
        super().__post_init__()

        # set row_func
        self.row_func = lambda *args: args[self.position]

        # set foreign_key metadata field to True
        self._metadata_["foreign_key"] = True
        self._metadata_["foreign_key_of"] = self.foreign_key_cols
        if "seed" not in self.prob_null_kwargs:
            self.prob_null_kwargs["seed"] = 1

        # add position to omit_list
        self.add_to_omit_list(["position", "foreign_key_cols"])


@dataclass
class PrimaryKey(UniqueId):
    """Thin wrapper for UniqueId to denote primary key.

    Args:
        num_rows: Number of rows to generate
        prefix: prefix to use for your ids. Defaults to empty string "".
        id_start_range: The starting range of the id. Defaults to 1.
        id_end_range: The ending range of the id. Defaults to ``num_rows``.
        id_length: The length of the id. Pads "0" to the left, i.e. an id ``1``, with
            ``id_length`` gives ``0001``.
    """

    prefix: str = ""
    id_start_range: int = 1
    id_end_range: int = None
    id_length: int = None
    _metadata_: Dict[str, str] = field(default_factory=dict)
    prob_null_kwargs: Dict[str, Any] = field(default_factory=dict)
    dtype: type = None

    def __post_init__(self):
        """Process class after init."""
        # set primary_key metadata field to True
        self._metadata_["primary_key"] = True

        # required for dependency resolution
        super().__post_init__()


@dataclass
class ColumnApply(BaseColumn):
    """Generic wrapper to call a function on a list.

    Function does not modify order of elements in list unless resize is True.

    ::

        # Given
        l1 = [1,2,3]
        l2 = [2,3,4]
        list_of_values = [l1, l2]
        column_function = lambda x,y: [(v1, v2) for v1, v2 in zip(x,y)]

        # internally entire column(s) is fed into function
        intermediate_result = column_function(*list_of_values, **column_function_kwargs)

        # intermediate_result
        intermediate_result = [
            (1, 2),
            (2, 3),
            (3, 4),
        ]

        # return [tpl[position] for tpl in intermediate_result]
        # position == 1 returns the first "column"
        # returns [1,2,3]
        # position == 2 returns the second "column"
        # returns [2,3,4]


    Args:
        list_of_values: A list of values or a list containing lists of values.
        column_func: The function to be applied to the list of columns.
        column_func_kwargs: Additional kwargs.
        num_rows: Checks output length if specified. Defaults to None.
        resize: Whether to resize the result. Defaults to False.
        check_all_inputs_same_length: Whether to bypass the same length check. Defaults
            to False. Used mainly with `cross_product`.
        _metadata_: additional information about column.
        prob_null_kwargs: arguments for probability_null function wrapper including
        "prob_null" from 0 to 1 , "null_value", and "seed".
        dtype: numpy dtype for the column output.
        filter: lambda function with condition to filter rows, ex:
            lambda x,y: x > y or lambda x,y: x.notna() & y.notna()
            obs. use logical & to chain conditions.
    """

    list_of_values: Union[List[Any], List[List[Any]]]
    column_func: Union[str, Callable]
    column_func_kwargs: Dict[str, str] = None
    resize: bool = False
    check_all_inputs_same_length: bool = True
    _metadata_: Dict[str, str] = field(default_factory=dict)
    prob_null_kwargs: Dict[str, Any] = field(default_factory=dict)
    dtype: type = None
    filter: Union[str, Callable] = None

    def _generate(self, num_rows: int = 0):
        """Run the column_apply function on the passed list of elements.

        Args:
            num_rows: Number of rows that need to be generated for this column

        Returns:
            A list of values

        Raises:
            ValueError: If vectors in list are not of same length or if
                function does not return the expected number of rows.
        """
        result = column_apply(
            num_rows=num_rows, **self.get_instructions(), **self.prob_null_kwargs
        )
        return result


@dataclass
class Explode(BaseColumn):
    """Run a function for each row in ``list_of_values``.

    ::

        # Given
        l1 = [1, 2]
        l2 = [a, b]
        list_of_values = [l1, l2]
        explode_func: lambda x: [x, y, z]

        # Intermediate result
        intermediate_result = [
            (1, a, x),
            (1, a, y),
            (1, a, z),
            (2, b, x),
            (2, b, y),
            (2, b, z)
        ]

        # return [tpl[position] for tpl in intermediate_result]
        # position == 0
        # returns [1, 1, 1, 2, 2, 2]
        # position == 1
        # returns [a, a, a, b, b, b]
        # position == 2
        # returns [x, y, z, x, y, z]


    Args:
        list_of_values: The list containing elements.
        position: The n-th position value to return.
        explode_func: The function to apply for each element.
        explode_func_kwargs: Any arguments to apply for each function.
        distribution_kwargs: The kwargs to pass to numpy_random. Defaults to None.
        num_rows: Checks output length if specified. Defaults to None.
        _metadata_: additional information about column.
        prob_null_kwargs: arguments for probability_null function wrapper including
        "prob_null" from 0 to 1 , "null_value", and "seed".
        dtype: numpy dtype for the column output.
        filter: lambda function with condition to filter rows, ex:
            lambda x,y: x > y or lambda x,y: x.notna() & y.notna()
            obs. use logical & to chain conditions.
    """

    list_of_values: List[List[Any]]
    position: int
    explode_func: Union[str, Callable]
    explode_func_kwargs: Dict[Any, Any] = None
    distribution_kwargs: Dict[str, Any] = None
    _metadata_: Dict[str, str] = field(default_factory=dict)
    prob_null_kwargs: Dict[str, Any] = field(default_factory=dict)
    dtype: type = None
    filter: Union[str, Callable] = None

    def _generate(self, num_rows: int = 0):
        """Run the explode function on the passed list of elements.

        Args:
            num_rows: Number of rows that need to be generated for this column
        """
        result = explode(
            num_rows=num_rows, **self.get_instructions(), **self.prob_null_kwargs
        )
        return result


@dataclass
class CrossProduct(ColumnApply):
    """Given a list of lists, returns all possible combinations of values.

    ::

        # Given
        l1 = [p1,p2]
        l2 = [q1,q2]
        list_of_values = [l1, l2]

        # intermediate_result
        intermediate_result = [
            (p1, q1),
            (p1, q2),
            (p2, q1),
            (p2, q2),
        ]

        # return [tpl[position] for tpl in intermediate_result]

        # position == 0 returns the first "column"
        # returns [p1, p1, p2, p2]
        # position == 1 returns the second "column"
        # returns [q1, q1, q2, q2]

    Args:
        list_of_lists: A list of list
        position: The n-th position value to return.
        _metadata_: additional information about column.
        prob_null_kwargs: arguments for probability_null function wrapper including
        "prob_null" from 0 to 1 , "null_value", and "seed".
        dtype: numpy dtype for the column output.
        filter: lambda function with condition to filter rows, ex:
            lambda x,y: x > y or lambda x,y: x.notna() & y.notna()
            obs. use logical & to chain conditions.
    """

    list_of_values: Union[List[Any], List[List[Any]]]
    position: int = 0
    # Required for init, to be reset post_init

    column_func: Callable = None

    def __post_init__(self):
        """Process class after init."""
        super().__post_init__()
        self.column_func = cross_product
        self.column_func_kwargs = {"position": self.position}
        self.add_to_omit_list(["position"])


@dataclass
class CrossProductWithSeparator(ColumnApply):
    """Given a list of lists, return possible combinations values.

    This function uses a itertools function in order to create all the
    combinations possible between the lists that are passed as the list
    of lists and replace the separator from it.

    Args:
        list_of_lists: List of lists that you want to combine. The number of final
            rows will be the product(len(lists)).
            separator: The given separator for the result.
            Eg: " " will return ["blue car", "blue bike", ...].

    Returns:
        A list (column) of values exploded with the designed separator

    """

    list_of_values: Union[List[Any], List[List[Any]]]
    separator: str = ""
    # Required for init, to be reset post_init

    column_func: Callable = None

    def __post_init__(self):
        """Process class after init."""
        super().__post_init__()
        self.column_func = cross_product_with_separator
        self.column_func_kwargs = {"separator": self.separator}
        self.add_to_omit_list(["separator"])


@dataclass
class DropDuplicates(ColumnApply):
    """Given a list of lists of all same length, return unique combinations.

    Row ordering is preserved.

    ::

        # given
        [
            [1,2,2,3,3],
            [1,2,2,3,4],
        ]

        # intermediate result
        intermediate_result = [
            (1,1),
            (2,2),
            (3,3),
            (3,4),
        ]

        # return [tpl[position] for tpl in intermediate_result]
        # position == 1 returns the first "column"
        # returns [1,2,3,3]
        # position == 2 returns the second "column"
        # returns [2,3,3,4]

    Args:
        list_of_lists: A list of list of values of equal length.
        position: The n-th position value to return.
    """

    list_of_values: Union[List[Any], List[List[Any]]]
    position: int = 0
    # Required for init, to be reset post_init:
    column_func: Callable = None

    def __post_init__(self):
        """Process class after init."""
        super().__post_init__()
        self.column_func = drop_duplicates
        self.column_func_kwargs = {"position": self.position}
        self.add_to_omit_list(["position"])


@dataclass
class BaseTable:
    """Master table from which data_fabricator objects can be created.

    Each class fabricating data will have num rows, method to get the list of columns
    and generate each column(caveats are when infer/resize are passed) and methods
    to put the data together to make a dataframe.
    """

    def __init_subclass__(cls, **kwargs):
        """Runs in every inheriting class."""
        if not cls.__dict__.get("num_rows", None):
            cls.num_rows = 0
        if not cls.__dict__.get("_metadata_", None):
            cls._metadata_ = {}

    def __post_init__(self):
        """Runs every time an object is created from this class."""
        self.dataframe = pd.DataFrame()

    def get_columns(self):
        """Used to obtain the list of Columns inside the class."""
        columns = self.__class__.__dict__
        return [
            column
            for column in columns
            if issubclass(getattr(self, column).__class__, BaseColumn)
        ]

    def _validate_num_cols_set(self, mock_data_generator: MockDataGenerator):
        """Check if num_cols set or dependencies exist."""
        # Check if num_rows is > 0
        if self.num_rows:
            return
        # Looks for dependencies in the columns
        if not any(
            getattr(self, column).has_dependencies for column in self.get_columns()
        ):
            err_msg = "This table doesn't have any dependencies, set num_rows!"
            logger.warning(
                "Error found in definition of %s, %s", self.__class__.__name__, err_msg
            )
            mock_data_generator.add_error_to_error_dict(
                error_msg=err_msg,
                table_name=self.__class__.__name__,
            )

    def _validate_column_references_are_valid(
        self, mock_data_generator: MockDataGenerator
    ):
        """Check all column references are valid."""
        for column in self.get_columns():
            col = getattr(self, column)
            # If there are no dependencies (_metadata_.dependencies = False)
            # then we can just move on
            if not col._metadata_["dependencies"]:
                continue
            # Check each column reference exists
            for class_name, col_ref in col.column_references:
                try:
                    getattr(mock_data_generator.tables[class_name], col_ref)
                except AttributeError:
                    err_msg = (
                        f"Column {column} is dependent on {class_name}.{col_ref}, "
                        f"but cannot find {col_ref} in {class_name}."
                    )

                    logger.warning(
                        "Error found in definition of %s. %s",
                        self.__class__.__name__,
                        err_msg,
                    )
                    mock_data_generator.add_error_to_error_dict(
                        error_msg=err_msg, table_name=self.__class__.__name__
                    )

    def validate_definition(self, mock_data_generator: MockDataGenerator):
        """Validate table definition.

        Current checks:
        - num_rows set or there are table dependencies
        - all dependencies are valid columns

        To extend functionality, add private function with check.
        """
        logger.info("Validating Table: %s", self.__class__.__name__)

        # Num. of Cols
        self._validate_num_cols_set(mock_data_generator=mock_data_generator)
        #  Col references are valid
        self._validate_column_references_are_valid(
            mock_data_generator=mock_data_generator
        )

    def generate_column(self, column_name: str, generator: MockDataGenerator):
        """Generate the column data following instructions.

        If that column already exists in the dataframe the function returns
        (base case).
        If it doesn't exist it first need to generate this column then move onto
        its dependent column (recursive step).

        Args:
            column_name: Column to generate
            generator: Instance of class containing all the objects
        """
        if column_name in self.dataframe:
            logger.info("Column %s already exists", column_name)
            return

        infer = self.num_rows == 0

        column_object = getattr(self, column_name)

        if column_object.has_dependencies:
            # If there is a list of references (dependencies) we generate each
            # of those values first recursively if they don't exist yet.
            column_object.list_of_values = []
            for df_name, column_reference in column_object.column_references:
                logger.info(
                    "Generating dependency recursively: in table %s column %s",
                    df_name,
                    column_reference,
                )
                generator.tables[df_name].generate_column(
                    column_name=column_reference, generator=generator
                )

                column_object.list_of_values.append(
                    generator.tables[df_name].dataframe[column_reference].to_list()
                )

            if column_object.filter:
                callable_to_apply = load_function_if_string(column_object.filter)

                column_object.list_of_values = column_object.filter_rows(
                    callable_to_apply, column_object.list_of_values
                )

        logger.info(
            "Generating column %s, num_rows: %s, infer rows: %s",
            column_name,
            self.num_rows,
            infer,
        )
        try:
            self.dataframe[column_name] = column_object.generate(self.num_rows)
        except (ValueError, TypeError) as err:
            if re.match(r"[Ii]nt", str(err)):
                raise ValueError(
                    f"The original error is {err}. Try using `pd.Int64Dtype` or"
                    f"`Int64` if you are trying to create an integer column with"
                    f"nulls."
                ) from err

            raise

        if infer:
            self.num_rows = len(self.dataframe[column_name])


class MockDataGenerator:
    """Master class containing all the tables to be instantiated.

    This class is used to generate all the Column, this way
    if there is a dependancy on another dataframes Column this
    can be solved for by accessing the other dataframe.

    Args:
        tables: List of table declaration.
        seed: For deterministic runs. Defaults to None.
    """

    error_dict: Optional[dict[str, list]] = None

    def __init__(self, tables: List[BaseTable], seed: int = None):
        """Init function creating a dictionary containing all the tables.

        Args:
            tables: Passed list of tables
            seed: random seed
        """
        self.seed = seed
        self.tables = {}
        for table in tables:
            self.tables[table.__name__] = table()

        if self.seed:
            random.seed(self.seed)
            np.random.seed(self.seed)

    @property
    def table_definitions_have_errors(self):
        """Check if column has dependencies."""
        return self.error_dict is not None

    def add_error_to_error_dict(self, error_msg: str, table_name: str):
        """Add a new error to the error dictionary."""
        if self.error_dict is None:
            self.error_dict = {}

        if table_name not in self.error_dict:
            self.error_dict[table_name] = []

        self.error_dict[table_name].append(error_msg)

    def validate_table_definitions(self) -> bool:
        """Validate all table definitions."""
        for table in self.tables.values():
            table.validate_definition(self)

    def generate_all(self):
        """Function to generate all the Columns of each table."""
        # Validate table definitions first
        self.validate_table_definitions()

        if self.table_definitions_have_errors:
            raise ValueError(
                f"There are problems with your table definition(s): {self.error_dict}"
            )

        # Then generate tables
        for table in self.tables.values():
            text_info = f"generating {table.__class__.__name__}"
            logger.info(text_info)
            logger.info("_" * len(text_info))

            for column_name in table.get_columns():
                table.generate_column(column_name, self)


def create_table(
    name: str, columns: Dict[str, any], num_rows=0, _metadata_: Dict[str, str] = None
):
    """Function used to inject objects if read as a yaml file.

    Args:
        name: class name to be given to the table declaration.
        columns: dictionary of object specifications for all columns.
        num_rows: number of rows in table.
        _metadata_: can including num_rows, description  and additional info.
    """
    return make_dataclass(
        cls_name=name,
        fields=(),
        bases=(BaseTable,),
        namespace={"_metadata_": _metadata_, "num_rows": num_rows, **columns},
    )
