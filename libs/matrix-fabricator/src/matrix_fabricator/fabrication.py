"""Generate synthetic data declaratively using YAML configurations.

This module provides a framework for generating synthetic data in a declarative way using YAML
configurations. It is designed to work with kedro pipelines and supports various data generation
strategies including:

- Unique IDs with customizable prefixes and formats
- Realistic fake data using the Faker library
- Random numbers from various statistical distributions
- Date sequences and ranges
- Cross products and row-wise operations
- Column references and dependencies between tables

The main workflow involves:
1. Define your data structure in YAML
2. Use the MockDataGenerator class to process the YAML
3. Get pandas DataFrames as output

Example YAML configuration:
```yaml
patients:
    num_rows: 100
    columns:
        id:
            type: generate_unique_id
            prefix: PAT_
            id_length: 8
        name:
            type: faker
            provider: name
        admission_date:
            type: generate_dates
            start_dt: 2023-01-01
            end_dt: 2023-12-31
            freq: D
```

Notes:
    - All generated data is deterministic when using seeds
    - Column references (e.g., table.column) are resolved automatically
    - Supports null injection with configurable probabilities
    - Handles type conversion and validation
"""

import datetime
import hashlib
import importlib
import itertools
import json
import logging
import random
import re
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Union, cast

import numpy as np
import pandas as pd
import pyspark
from faker import Faker
from pandas.api.types import is_list_like

# --- Configuration ---
logger = logging.getLogger(__name__)

# Default path for loading functions if namespace isn't specified
_DEFAULT_OBJ_PATH = __name__


def load_obj(obj_path: str, default_obj_path: str = _DEFAULT_OBJ_PATH) -> Any:
    """Extract an object from a given path.

    Dynamically imports modules and extracts objects (functions, classes, etc.) from them.
    Used internally to load custom functions referenced in the YAML configuration.

    Args:
        obj_path: Path to an object to be extracted, including the object name.
            Format: "module.submodule.object" or just "object" if in default path.
        default_obj_path: Default module path to look in if obj_path doesn't specify one.
            Defaults to this module.

    Returns:
        The extracted object (function, class, etc.)

    Raises:
        ImportError: If the module cannot be imported
        AttributeError: If the object doesn't exist in the module

    Example:
        >>> func = load_obj("numpy.random.normal")  # Loads numpy's normal distribution
        >>> local_func = load_obj("my_function")  # Loads from current module
    """
    try:
        obj_path_list = obj_path.rsplit(".", 1)
        module_path = obj_path_list[0] if len(obj_path_list) > 1 else default_obj_path
        obj_name = obj_path_list[-1]

        if not module_path:
            raise ValueError(f"Invalid module path derived from '{obj_path}'.")

        module_obj = importlib.import_module(module_path)
        if not hasattr(module_obj, obj_name):
            raise AttributeError(f"Object '{obj_name}' not found in module '{module_path}'.")
        return getattr(module_obj, obj_name)
    except (ImportError, AttributeError, ValueError) as e:
        logger.error(f"Failed to load object '{obj_path}': {e}")
        raise


def load_callable_with_libraries(function_string: str) -> Callable:
    """Convert a string function definition into a callable, importing needed libraries.

    Takes a string containing a function definition (typically a lambda) and converts it
    into a callable Python object. Automatically imports any libraries referenced in the
    function string.

    Args:
        function_string: String containing a function definition, typically a lambda.
            Example: "lambda x: math.sqrt(x)" or "lambda x, y: random.randint(x, y)"

    Returns:
        A callable function object

    Raises:
        ValueError: If the string contains import statements (security measure)
        TypeError: If the evaluated string doesn't produce a callable
        ImportError: If referenced libraries can't be imported

    Example:
        >>> func = load_callable_with_libraries("lambda x: math.sqrt(x)")
        >>> func(4)
        2.0

    Warning:
        This function uses eval() internally. Only use with trusted input strings.
        Never expose this directly to user input in production without strict validation.
    """
    logger.debug(f"Attempting to load callable: {function_string}")
    # Basic check for safety, disallow imports within the lambda for now
    if "import " in function_string:
        raise ValueError("Import statements are not allowed within lambda strings for security reasons.")

    libraries_to_import = set()
    # Regex to find patterns like 'somelibrary.somefunction' or 'somelib.submodule.func'
    potential_libs = re.findall(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\.[a-zA-Z_]", function_string)
    for lib in potential_libs:
        try:
            globals()[lib] = importlib.import_module(lib)
            libraries_to_import.add(lib)
            logger.debug(f"Dynamically imported library: {lib}")
        except ImportError:
            logger.debug(f"Could not import '{lib}', assuming it's not a top-level library.")
            pass
        except Exception as e:
            logger.warning(f"Error importing library {lib}: {e}")

    try:
        callable_func = eval(function_string, globals())
        if not callable(callable_func):
            raise TypeError(f"Evaluated string '{function_string}' did not result in a callable object.")
        return callable_func
    except Exception as e:
        logger.error(f"Failed to evaluate function string '{function_string}': {e}")
        raise


def _apply_null_injection(
    data: List[Any], null_config: Optional[Dict[str, Any]], base_seed: Optional[int] = None
) -> List[Any]:
    """Inject null values into a list based on configuration.

    Internal helper function that handles null value injection based on probability.
    Used by column generators to implement the 'inject_nulls' configuration option.

    Args:
        data: The list of data to process
        null_config: Dictionary containing:
            - probability: float between 0 and 1
            - value: value to use for nulls (optional, defaults to None)
            - seed: specific seed for this injection (optional)
        base_seed: Fallback seed if null_config doesn't specify one

    Returns:
        The list with nulls potentially injected according to configuration

    Example:
        >>> data = [1, 2, 3, 4, 5]
        >>> config = {"probability": 0.4, "value": -999, "seed": 42}
        >>> _apply_null_injection(data, config)
        [1, -999, 3, -999, 5]
    """
    if not null_config or not isinstance(null_config, dict):
        return data

    prob_null = null_config.get("probability")
    if prob_null is None or not (0 <= prob_null <= 1):
        logger.warning(f"Invalid or missing 'probability' in null_config: {null_config}. Skipping null injection.")
        return data

    null_value = null_config.get("value", None)
    injection_seed = null_config.get("seed")

    # Use specific injection_seed if provided, otherwise use derived base_seed
    effective_seed = injection_seed if injection_seed is not None else base_seed

    # Always use a dedicated random instance for null injection if a seed is determined
    if effective_seed is not None:
        null_random = random.Random(effective_seed)
        logger.debug(f"Using specific seed {effective_seed} for null injection.")
    else:
        # Fallback to global random state only if no seed available anywhere
        # !!! This might still cause issues if other code modifies global state
        null_random = random.Random()  # Create a new instance without seed
        logger.warning("Using unseeded random instance for null injection as no seed was provided.")

    return [x if null_random.random() > prob_null else null_value for x in data]


def generate_unique_id(
    num_rows: int, prefixes: list[str] | None = None, delimiter: str | None = None, id_length: int = 8
) -> List[str]:
    """Generate unique numeric IDs with optional prefix and fixed length.

    Creates a range of IDs by incrementing from 1, optionally prefixing them,
    and adjusting to a fixed length if specified.

    Args:
        num_rows: The number of IDs to generate.
        prefixes: Optional string to prepend to each ID (e.g., "USER_", "PAT_").
        id_length: If provided, ensures each ID (including prefix) is exactly this length
                  by zero-padding or truncating the numeric part as needed.

    Returns:
        A list of string IDs.

    Raises:
        ValueError: If id_length is specified but too short for prefix + enough digits to certify uniqueness.

    Example:
        >>> # In YAML config:
        >>> # patient_id:
        >>> #   type: generate_unique_id
        >>> #   prefix: PAT_
        >>> #   id_length: 10
        >>> # Would generate: ['PAT_0000123', 'PAT_0000456', ...]
    """
    if num_rows <= 0:
        return []

    numeric_strings = [str(n + 1) for n in range(num_rows)]

    final_ids = []

    for num_str in numeric_strings:
        if prefixes is not None:
            idx = random.randint(0, len(prefixes) - 1)
            prefix = prefixes[idx]
        else:
            prefix = ""

        padded_num_str = num_str.zfill(id_length)

        if delimiter is None:
            delimiter = ""

        final_ids.append(f"{prefix}{delimiter}{padded_num_str}")

    return final_ids


def faker(
    num_rows: int,
    provider: str,
    localisation: Optional[Union[str, List[str]]] = None,
    provider_args: Optional[Dict[str, Union[str, int]]] = None,
    faker_seed: Optional[int] = None,
    seed: Optional[int] = None,  # Derived column seed
) -> List[Any]:
    """Generate realistic fake data using the Faker library.

    Provides access to Faker's extensive collection of data providers for generating
    realistic synthetic data like names, addresses, phone numbers, etc. Supports
    localization for international data formats.

    Args:
        num_rows: Number of fake data entries to generate.
        provider: The Faker provider to use (e.g., 'name', 'address', 'company').
                 See Faker docs for all available providers.
        localisation: Optional locale(s) for data generation. Can be a single locale
                     (e.g., 'en_US') or a list of locales (['en_US', 'ja_JP']).
        provider_args: Optional dict of arguments to pass to the Faker provider.
        faker_seed: Specific seed for this Faker instance. Takes precedence over seed.
        seed: General column seed, used if faker_seed isn't provided.

    Returns:
        A list containing the generated fake data.

    Example:
        >>> # In YAML config:
        >>> # person_name:
        >>> #   type: faker
        >>> #   provider: name
        >>> #   localisation: en_US
        >>> #   faker_seed: 42
        >>> # phone:
        >>> #   type: faker
        >>> #   provider: phone_number
        >>> #   localisation: ja_JP
        >>> #   provider_args:
        >>> #     country_code: true
    """
    if num_rows <= 0:
        return []

    # Use specific faker_seed if provided, otherwise use derived column seed
    effective_seed = faker_seed if faker_seed is not None else seed

    # Create a unique Faker instance for each call to isolate seeding
    faker_obj = Faker(localisation)

    if effective_seed is not None:
        faker_obj.seed_instance(effective_seed)
        logger.debug(f"Seeding Faker instance with seed: {effective_seed} for provider {provider}")

    provider_args = provider_args or {}
    try:
        provider_func = getattr(faker_obj, provider)
        faker_results = [provider_func(**provider_args) for _ in range(num_rows)]
        return faker_results
    except AttributeError:
        logger.error(f"Faker object does not have provider: {provider}")
        raise
    except Exception as e:
        logger.error(f"Error calling faker.{provider} with args {provider_args}: {e}")
        raise


def numpy_random(
    num_rows: int,
    distribution: str,
    numpy_seed: Optional[int] = None,
    seed: Optional[int] = None,  # Derived column seed
    **kwargs,
) -> List[Any]:
    """Generate random numbers using NumPy's probability distributions.

    Provides access to NumPy's extensive collection of probability distributions
    for generating synthetic numerical data. Useful for creating realistic
    statistical distributions in your synthetic datasets.

    Args:
        num_rows: Number of random samples to generate.
        distribution: Name of the NumPy random distribution to use (e.g., 'normal',
                     'uniform', 'poisson', 'exponential'). Must match a method in
                     numpy.random.Generator.
        numpy_seed: Specific seed for this NumPy generation. Takes precedence over seed.
        seed: General column seed, used if numpy_seed isn't provided.
        **kwargs: Additional arguments passed directly to the NumPy distribution function.
                 See NumPy docs for distribution-specific parameters.

    Returns:
        A list containing the generated random numbers.

    Example:
        >>> # In YAML config:
        >>> # ages:
        >>> #   type: numpy_random
        >>> #   distribution: normal
        >>> #   loc: 35    # mean
        >>> #   scale: 10  # standard deviation
        >>> #   numpy_seed: 42
        >>> # counts:
        >>> #   type: numpy_random
        >>> #   distribution: poisson
        >>> #   lam: 5     # mean rate
    """
    if num_rows <= 0:
        return []

    # Use specific numpy_seed if provided, otherwise use derived column seed
    effective_seed = numpy_seed if numpy_seed is not None else seed

    if effective_seed is not None:
        rng = np.random.Generator(np.random.PCG64(effective_seed))
        logger.debug(f"Using numpy seeded generator with seed: {effective_seed}")
    else:
        # Fallback to legacy global random state if no seed
        rng = np.random
        logger.debug("Using global numpy random state.")

    try:
        dist_func = getattr(rng, distribution)
        samples = dist_func(size=num_rows, **kwargs).tolist()
        return samples
    except AttributeError:
        logger.error(f"Numpy random does not have distribution: {distribution}")
        raise
    except Exception as e:
        logger.error(f"Error calling numpy.{distribution} with args {kwargs}: {e}")
        raise


def generate_random_arrays(
    num_rows: int,
    sample_values: List[Any],
    allow_duplicates: bool = False,
    to_json: bool = False,
    delimiter: Optional[str] = None,
    length: Optional[int] = None,
    min_length: int = 1,
    max_length: Optional[int] = None,
    seed: Optional[int] = None,
) -> List[Union[List[Any], str]]:
    """Generate random arrays (lists) by sampling from provided values.

    Creates a list where each element is itself a randomly generated array/list
    containing elements from sample_values. Offers flexible output formats and
    length control.

    Args:
        num_rows: Number of arrays to generate.
        sample_values: Pool of values to sample from when creating arrays.
        allow_duplicates: If True, same value can appear multiple times in one array.
        to_json: If True, each array is JSON-serialized to a string.
        delimiter: If provided, joins array elements with this delimiter into a string.
                 Ignored if to_json=True.
        length: If provided, all arrays will have exactly this length.
                Overrides min_length and max_length.
        min_length: Minimum length for generated arrays. Used if length not specified.
        max_length: Maximum length for generated arrays. Defaults to len(sample_values)
                   if not specified and allow_duplicates=False.
        seed: Optional seed for reproducible generation.

    Returns:
        If to_json=True: List[str] containing JSON arrays
        If delimiter: List[str] containing delimited strings
        Otherwise: List[List[Any]] containing the generated arrays

    Raises:
        ValueError: If min_length > max_length after adjustments
        ValueError: If allow_duplicates=False and requested length exceeds unique values

    Example:
        >>> # In YAML config:
        >>> # tags:
        >>> #   type: generate_random_arrays
        >>> #   sample_values: ["urgent", "review", "approved"]
        >>> #   min_length: 1
        >>> #   max_length: 2
        >>> #   allow_duplicates: false
        >>> #   delimiter: "|"  # Results like: "urgent|review"
        >>> # json_data:
        >>> #   type: generate_random_arrays
        >>> #   sample_values: [1, 2, 3, 4]
        >>> #   length: 2
        >>> #   to_json: true  # Results like: "[1, 4]"
    """
    if num_rows <= 0:
        return []
    if not sample_values:
        logger.warning("generate_random_arrays called with empty sample_values. Returning list of empty lists.")
        return [[] for _ in range(num_rows)]

    # Use local seeded generator if seed provided
    rand_gen = random.Random(seed) if seed is not None else random

    value_pool = list(sample_values)

    results: List[List[Any]] = []
    max_len = max_length if max_length is not None else len(value_pool)
    if length is not None:
        min_len = max_len = length
    else:
        min_len = min(min_length, len(value_pool))
        max_len = min(max_len, len(value_pool))

    if min_len > max_len:
        logger.warning(f"min_length ({min_len}) > max_length ({max_len}). Adjusting max_length.")
        max_len = min_len

    if not allow_duplicates and max_len > len(value_pool):
        logger.warning(
            f"Requested max_length ({max_len}) > available unique values ({len(value_pool)}) without duplicates. Clamping max_length."
        )
        max_len = len(value_pool)
        if min_len > max_len:
            min_len = max_len

    for _ in range(num_rows):
        current_length = rand_gen.randint(min_len, max_len) if length is None else length

        if allow_duplicates:
            row_array = rand_gen.choices(value_pool, k=current_length)
        else:
            if current_length > len(value_pool):
                logger.error(
                    f"Cannot sample {current_length} items without duplicates from pool of size {len(value_pool)}"
                )
                current_length = len(value_pool)
            row_array = rand_gen.sample(value_pool, k=current_length)

        results.append(row_array)

    # Apply formatting after generating all arrays
    if delimiter:
        return [delimiter.join(map(str, row)) for row in results]
    elif to_json:
        return [json.dumps(row) for row in results]

    return results


def generate_values(
    num_rows: int,
    sample_values: Union[List[Any], Dict[str, Union[int, float]]],
    sort_values: bool = False,
    seed: Optional[int] = None,
) -> List[Any]:
    """Generate values by sampling from a list or using weighted choice.

    Provides two main modes of operation:
    1. List mode: Samples values from a list, handling up/down sampling automatically
    2. Dict mode: Uses weighted random choice where dict values are weights

    Args:
        num_rows: Number of values to generate.
        sample_values: Either:
            - A list of values to sample from
            - A dict mapping values to their weights (e.g., {"A": 0.7, "B": 0.3})
        sort_values: If True and using list mode, sorts the output values.
        seed: Optional seed for reproducible generation.

    Returns:
        A list containing the sampled/generated values.

    Raises:
        TypeError: If sample_values is neither a list nor a dict.
        ValueError: If using dict mode and any weight is negative.

    Example:
        >>> # In YAML config:
        >>> # status:  # Simple sampling
        >>> #   type: generate_values
        >>> #   sample_values: ["active", "inactive", "pending"]
        >>> #   sort_values: true
        >>> # priority:  # Weighted sampling
        >>> #   type: generate_values
        >>> #   sample_values:
        >>> #     high: 1    # 10% chance
        >>> #     medium: 4  # 40% chance
        >>> #     low: 5     # 50% chance
        >>> # referenced:  # Sample from another column
        >>> #   type: generate_values
        >>> #   sample_values: other_table.status_column
    """
    if num_rows <= 0:
        return []

    # Use local seeded generator if seed provided
    rand_gen = random.Random(seed) if seed is not None else random

    if isinstance(sample_values, list):
        if not sample_values:
            logger.warning("generate_values called with empty list sample_values.")
            return [None] * num_rows
        # Handle potential mixed types before sorting
        try:
            # Attempt direct sorting if types are compatible
            unique_elements = sorted(list(set(sample_values)))
        except TypeError:
            logger.debug("Mixed types detected in sample_values for generate_values. Sorting as strings.")
            # Fallback: convert to string for sorting if direct sort fails
            unique_elements = sorted(list(set(map(str, sample_values))))
            # Attempt to convert back to original types where possible, though this might be lossy
            original_type_map = {str(v): v for v in sample_values}
            unique_elements = [original_type_map.get(s, s) for s in unique_elements]

        if len(unique_elements) < num_rows:
            result = rand_gen.choices(unique_elements, k=num_rows)
        else:
            # Even if lengths match, sample to ensure seeded reproducibility matches choices/sample
            result = rand_gen.sample(unique_elements, num_rows)

        if sort_values:
            result.sort()
        return result

    elif isinstance(sample_values, dict):
        # Pass seed down to weighted choice
        return generate_weighted_choice(num_rows=num_rows, weights_dict=sample_values, seed=seed)
    else:
        raise TypeError("sample_values must be a list of values or a dictionary of values with weights.")


def generate_weighted_choice(
    num_rows: int,
    weights_dict: Dict[str, Union[int, float]],
    seed: Optional[int] = None,
) -> List[Any]:
    """Generate list of values based on provided weights."""
    if num_rows <= 0:
        return []
    if not weights_dict:
        logger.warning("generate_weighted_choice called with empty weights_dict.")
        return [None] * num_rows

    elements = list(weights_dict.keys())
    weights = list(weights_dict.values())

    if not all(isinstance(w, (int, float)) and w >= 0 for w in weights):
        raise ValueError("Weights must be non-negative numbers.")
    if sum(weights) <= 0:
        logger.warning("Sum of weights is zero or negative. Cannot perform weighted choice.")
        # Fallback to equal probability
        return random.choices(elements, k=num_rows)

    # Use local seeded generator if seed provided
    rand_gen = random.Random(seed) if seed is not None else random

    list_of_values = rand_gen.choices(elements, weights=weights, k=num_rows)
    return list_of_values


def map_values(
    input_column: List[Any],
    mapping: Dict[Any, Any],
    default_value: Any = None,
) -> List[Any]:
    """Map values from input_column to new values using a mapping dictionary.

    Transforms each value in the input list by looking it up in the mapping
    dictionary. Values not found in the mapping are replaced with default_value.
    Useful for converting codes to labels, IDs to names, etc.

    Args:
        input_column: List of values to transform.
        mapping: Dictionary defining the value mappings.
        default_value: Value to use when input value isn't in mapping.

    Returns:
        A list containing the mapped values.

    Example:
        >>> # In YAML config:
        >>> # status_label:
        >>> #   type: map_values
        >>> #   input_column: my_table.status_code
        >>> #   mapping:
        >>> #     1: "Active"
        >>> #     2: "Inactive"
        >>> #     3: "Pending"
        >>> #   default_value: "Unknown"
        >>> # severity:
        >>> #   type: map_values
        >>> #   input_column: my_table.error_code
        >>> #   mapping:
        >>> #     E001: "Critical"
        >>> #     E002: "High"
        >>> #     W001: "Medium"
        >>> #     W002: "Low"
    """
    return [mapping.get(key, default_value) for key in input_column]


def hash_map(
    input_column: List[Any],
    buckets: List[Any],
) -> List[Any]:
    """Hash values from input_column to deterministically assign them to buckets.

    Uses MD5 hashing to consistently map input values to buckets. The same input
    value will always map to the same bucket, making this useful for creating
    stable assignments or partitions.

    Args:
        input_column: List of values to hash and map.
        buckets: List of possible values to map into. Order matters as it affects
                which inputs map to which buckets.

    Returns:
        A list where each input value has been replaced by its assigned bucket value.

    Raises:
        ValueError: If buckets list is empty.

    Example:
        >>> # In YAML config:
        >>> # user_cohort:  # Stable user assignment to test groups
        >>> #   type: hash_map
        >>> #   input_column: my_table.user_id
        >>> #   buckets: ["control", "treatment_a", "treatment_b"]
        >>> # region:  # Consistent territory assignment
        >>> #   type: hash_map
        >>> #   input_column: my_table.postal_code
        >>> #   buckets: ["North", "South", "East", "West"]

    Note:
        The mapping is deterministic but not uniform. If you need uniform
        distribution across buckets, consider using generate_values with
        equal weights instead.
    """
    if not buckets:
        raise ValueError("Bucket list cannot be empty for hash_map.")

    results = []
    num_buckets = len(buckets)
    for key in input_column:
        hash_object = hashlib.md5(str(key).encode("utf-8"))
        hash_digest = hash_object.hexdigest()
        hash_int = int(hash_digest, 16)
        index = hash_int % num_buckets
        results.append(buckets[index])
    return results


def generate_dates(
    num_rows: int,
    start_dt: Union[str, datetime.datetime],
    end_dt: Union[str, datetime.datetime],
    freq: str,
    sort_dates: bool = True,
    date_format: Optional[str] = None,
) -> List[Union[datetime.datetime, str]]:
    """Generate a sequence of dates within a specified range.

    Uses pandas date_range to create a sequence of dates, then samples from this
    sequence to match the requested number of rows. Handles both upsampling
    (with replacement) and downsampling (without replacement).

    Args:
        num_rows: Number of dates to generate.
        start_dt: Start date of the range. Can be string ('2023-01-01') or datetime.
        end_dt: End date of the range (inclusive).
        freq: Frequency string for pandas date_range. Common options:
            - 'D': Calendar day
            - 'B': Business day
            - 'W': Weekly
            - 'M': Month end
            - 'Q': Quarter end
            - 'Y': Year end
            - 'H': Hourly
            See pandas documentation for full list of frequency aliases.
        sort_dates: If True, sorts the output dates chronologically.
        date_format: If provided, formats dates to strings using this format.
                    Example: '%Y-%m-%d' for '2023-01-01'.

    Returns:
        If date_format is None: List of pandas Timestamps
        If date_format is provided: List of formatted date strings

    Raises:
        ValueError: If date_range creation fails (e.g., invalid freq)

    Example:
        >>> # In YAML config:
        >>> # admission_date:  # Random business days
        >>> #   type: generate_dates
        >>> #   start_dt: 2023-01-01
        >>> #   end_dt: 2023-12-31
        >>> #   freq: B
        >>> #   sort_dates: true
        >>> # report_timestamp:  # Hourly timestamps as strings
        >>> #   type: generate_dates
        >>> #   start_dt: 2023-01-01 00:00:00
        >>> #   end_dt: 2023-01-02 23:59:59
        >>> #   freq: H
        >>> #   date_format: "%Y-%m-%d %H:%M:%S"
    """
    if num_rows <= 0:
        return []
    try:
        unique_dates = list(pd.date_range(start_dt, end_dt, freq=freq))
    except ValueError as e:
        logger.error(f"Error generating date range with freq='{freq}': {e}")
        raise

    if not unique_dates:
        logger.warning(f"Date range parameters resulted in zero dates.")
        return []

    dates: List[datetime.datetime]
    # If num_rows is specified, sample; otherwise, use all unique dates
    if len(unique_dates) == num_rows:
        dates = unique_dates
    elif len(unique_dates) > num_rows:
        dates = random.sample(unique_dates, num_rows)
    else:
        dates = random.choices(unique_dates, k=num_rows)

    if sort_dates:
        dates.sort()

    if date_format:
        return [pd.Timestamp(d).strftime(date_format) for d in dates]
    else:
        # Return as Pandas Timestamps (often more convenient than python datetimes)
        return dates


def cross_product(
    *input_columns: List[Any],
    position: int = 0,
) -> List[Any]:
    """Generate the Cartesian product of multiple columns and extract one dimension.

    Takes multiple input columns and computes their Cartesian product (all possible
    combinations). Returns the values from the specified position in each combination.
    Commonly used within column_apply to create exhaustive combinations of values.

    Args:
        *input_columns: Variable number of input lists to compute product of.
        position: Which position (0-based) to extract from each combination tuple.
                 For example, with position=0, returns elements from first input
                 repeated according to product structure.

    Returns:
        List containing elements from the specified position of each combination
        in the Cartesian product.

    Raises:
        ValueError: If position is out of range for the number of input columns.

    Example:
        >>> # In YAML config:
        >>> # Given:
        >>> #   regions = ['North', 'South']
        >>> #   products = ['A', 'B', 'C']
        >>> # region_repeated:  # ['North', 'North', 'North', 'South', 'South', 'South']
        >>> #   type: column_apply
        >>> #   input_columns: [my_table.regions, my_table.products]
        >>> #   column_func: cross_product
        >>> #   column_func_kwargs:
        >>> #     position: 0
        >>> # product_repeated:  # ['A', 'B', 'C', 'A', 'B', 'C']
        >>> #   type: column_apply
        >>> #   input_columns: [my_table.regions, my_table.products]
        >>> #   column_func: cross_product
        >>> #   column_func_kwargs:
        >>> #     position: 1

    Note:
        Typically used with column_apply and check_all_inputs_same_length=False
        since input columns can have different lengths.
    """
    if not input_columns:
        return []

    if not (0 <= position < len(input_columns)):
        raise ValueError(f"Position {position} is out of range for {len(input_columns)} input columns.")

    product_tuples = list(itertools.product(*input_columns))
    return [tpl[position] for tpl in product_tuples]


def column_apply(
    input_columns: List[List[Any]],
    column_func: Union[str, Callable],
    column_func_kwargs: Optional[Dict[str, Any]] = None,
    num_rows: Optional[int] = None,
    check_all_inputs_same_length: bool = True,
    resize: bool = False,
) -> List[Any]:
    """Apply a function to entire columns at once.

    Takes one or more input columns and passes them as separate arguments to
    column_func. The function operates on the entire columns simultaneously,
    rather than row by row. Useful for operations that need to see all values
    at once (e.g., cumulative calculations, cross products).

    Args:
        input_columns: List of input columns (each a list) to pass to column_func.
        column_func: Function to apply. Can be:
            - A callable object
            - Import path as string (e.g., 'numpy.cumsum')
            - Lambda expression as string (use with caution)
        column_func_kwargs: Optional kwargs to pass to column_func.
        num_rows: If provided, verifies output length matches this value.
                 Required if resize=True.
        check_all_inputs_same_length: If True, verifies all input columns have
                                    same length. Set False for operations like
                                    cross_product that handle varying lengths.
        resize: If True and num_rows provided, resizes output to match num_rows
               via sampling.

    Returns:
        List containing the results of applying column_func.

    Raises:
        ValueError: If check_all_inputs_same_length=True and lengths differ
        ValueError: If num_rows provided and output length doesn't match
        TypeError: If column_func string can't be converted to callable
        Exception: Any exception raised by column_func

    Example:
        >>> # In YAML config:
        >>> # cumulative_sum:
        >>> #   type: column_apply
        >>> #   input_columns: [my_table.value]
        >>> #   column_func: numpy.cumsum
        >>> # all_combinations:
        >>> #   type: column_apply
        >>> #   input_columns: [my_table.users, my_table.products]
        >>> #   column_func: matrix.utils.fabrication.cross_product
        >>> #   check_all_inputs_same_length: false
        >>> #   column_func_kwargs:
        >>> #     position: 0
    """
    if not input_columns:
        return []

    if isinstance(column_func, str):
        if column_func.startswith("lambda"):
            callable_to_apply = load_callable_with_libraries(column_func)
        else:
            callable_to_apply = load_obj(column_func)
    elif callable(column_func):
        callable_to_apply = column_func
    else:
        raise TypeError("column_func must be a callable or a string path/lambda.")

    column_func_kwargs = column_func_kwargs or {}

    if check_all_inputs_same_length:
        expected_length = len(input_columns[0])
        if not all(len(col) == expected_length for col in input_columns):
            raise ValueError("All input columns must have the same length for column_apply (if check enabled).")

    try:
        results = callable_to_apply(*input_columns, **column_func_kwargs)
    except Exception as e:
        logger.error(f"Error applying column_func '{column_func}' with kwargs {column_func_kwargs}: {e}")
        raise

    if not is_list_like(results):
        logger.warning(
            f"column_apply function did not return a list-like object. Got type {type(results)}. Attempting conversion."
        )
        try:
            results = list(results)
        except TypeError:
            raise TypeError(f"Result of column_func could not be converted to a list.")

    if resize:
        if len(results) != num_rows:
            logger.warning("Resizing list from %s to %s", len(results), num_rows)

            # sample without replacement
            if len(results) > num_rows:
                results = random.sample(results, num_rows)

            else:
                results = generate_values(num_rows=num_rows, sample_values=results)

    if num_rows is not None and len(results) != num_rows:
        raise ValueError(f"column_apply function returned {len(results)} rows, but expected {num_rows}.")

    return results


def row_apply(
    input_columns: List[List[Any]],
    row_func: Union[str, Callable],
    row_func_kwargs: Optional[Dict[str, Any]] = None,
    num_rows: Optional[int] = None,
    resize: bool = False,
    seed: Optional[int] = None,
) -> List[Any]:
    """Apply a function row-wise to values from input columns.

    Takes one or more input columns, conceptually 'zips' them together to form
    rows, and applies row_func to each row tuple. Useful for operations that
    compute new values based on corresponding elements from multiple columns.

    Args:
        input_columns: List of input columns (each a list) to process row-wise.
                      All columns must have the same length.
        row_func: Function to apply to each row. Can be:
            - A callable object
            - Import path as string (e.g., 'math.sqrt')
            - Lambda expression as string (e.g., 'lambda x, y: x + y')
            The function should accept arguments matching the number of input columns.
        row_func_kwargs: Optional kwargs passed to row_func on each call.
        num_rows: If provided, verifies output length matches this value.
                 Required if resize=True.
        resize: If True and num_rows provided, resizes input rows to match
               num_rows via sampling BEFORE applying row_func.

    Returns:
        List containing results of applying row_func to each row.

    Raises:
        ValueError: If input columns have different lengths
        ValueError: If num_rows provided and output length doesn't match
        TypeError: If row_func string can't be converted to callable

    Example:
        >>> # In YAML config:
        >>> # full_name:  # Combining first and last names
        >>> #   type: row_apply
        >>> #   input_columns: [my_table.first_name, my_table.last_name]
        >>> #   row_func: "lambda f, l: f'{f} {l}'"
        >>> # bmi:  # Computing BMI from height and weight
        >>> #   type: row_apply
        >>> #   input_columns: [my_table.weight_kg, my_table.height_m]
        >>> #   row_func: "lambda w, h: w / (h * h)"
        >>> # age_group:  # Categorizing ages
        >>> #   type: row_apply
        >>> #   input_columns: [my_table.age]
        >>> #   row_func: "lambda x: 'child' if x < 18 else 'adult'"

    Note:
        If row_func raises an exception for a row, that row gets None in the
        output and the error is logged, but processing continues.
    """
    if not input_columns:
        return []

    if isinstance(row_func, str):
        if row_func.startswith("lambda"):
            callable_to_apply = load_callable_with_libraries(row_func)
        else:
            callable_to_apply = load_obj(row_func)
    elif callable(row_func):
        callable_to_apply = row_func
    else:
        raise TypeError("row_func must be a callable or a string path/lambda.")

    row_func_kwargs = row_func_kwargs or {}

    # Ensure input_columns is always a list of lists
    if input_columns and not isinstance(input_columns[0], list):
        # If the first element isn't a list, assume it's a single resolved column
        # and wrap each column in its own list for consistency.
        # This handles the case where YAML provides a single string reference like 'table.column'
        # which resolves to a list, instead of a list containing that list.
        logger.debug("Wrapping single input column list for row_apply.")
        input_columns = [input_columns]  # Wrap the resolved single list

    expected_length = len(input_columns[0])
    if not all(len(col) == expected_length for col in input_columns):
        raise ValueError("All input columns must have the same length for row_apply.")

    list_of_tuples = list(zip(*input_columns))

    if resize and num_rows is not None and len(list_of_tuples) != num_rows:
        logger.warning(f"Resizing input rows for row_apply from {len(list_of_tuples)} to {num_rows}.")
        resize_random = random.Random(seed) if seed is not None else random

        if len(list_of_tuples) > num_rows:
            list_of_tuples = resize_random.sample(list_of_tuples, num_rows)
        else:
            list_of_tuples = resize_random.choices(list_of_tuples, k=num_rows)

    results = []
    for tpl in list_of_tuples:
        try:
            result = callable_to_apply(*tpl, **row_func_kwargs)
            results.append(result)
        except Exception as e:
            logger.error(f"Error applying row_func '{row_func}' to tuple {tpl} with kwargs {row_func_kwargs}: {e}")
            results.append(None)

    if num_rows is not None and len(results) != num_rows:
        raise ValueError(f"row_apply resulted in {len(results)} rows, but expected {num_rows}.")

    return results


def copy_column(
    source_column: List[Any],
    sample: Optional[Dict[str, Any]] = None,
    seed: Optional[int] = None,  # Derived column seed
) -> List[Any]:
    """
    Copies or samples values from a source column.

    Args:
        source_column: The list of values to copy/sample from.
        sample: Optional dictionary containing:
            'num_rows': The target number of rows.
            'seed': An optional seed for deterministic sampling.
        seed: Derived column seed, used if sample['seed'] is not present.

    Returns:
        A list of values, potentially sampled.
    """
    if not source_column:
        return []

    # If no sampling config, just return the source column as is
    if sample is None:
        return list(source_column)

    # Extract config from sample dict
    num_rows = sample.get("num_rows")
    if num_rows is None:
        # If num_rows not in sample dict, assume we want the original size
        # Or should this raise error? Let's assume original size for now.
        logger.warning("'num_rows' not specified in 'sample' config for copy_column. Returning original column.")
        return list(source_column)

    if num_rows <= 0:
        return []

    # Determine seed: specific sample seed > derived column seed
    sample_specific_seed = sample.get("seed")
    effective_seed = sample_specific_seed if sample_specific_seed is not None else seed

    # Always create a local Random instance, seeded if a seed value was determined
    sample_random = random.Random(effective_seed) if effective_seed is not None else random
    if effective_seed is None:
        logger.warning("Using global random state for copy_column sampling.")

    if len(source_column) == num_rows:
        # Even if size matches, sample/choices ensures seeded reproducibility
        # return list(source_column)
        return sample_random.choices(source_column, k=num_rows)
    elif len(source_column) > num_rows:
        logger.debug(f"Downsampling column from {len(source_column)} to {num_rows} rows.")
        return sample_random.sample(source_column, num_rows)
    else:  # len(source_column) < num_rows
        logger.debug(f"Upsampling column from {len(source_column)} to {num_rows} rows.")
        return sample_random.choices(source_column, k=num_rows)


# --- Orchestration Class ---


class MockDataGenerator:
    """Generate synthetic data based on declarative YAML configurations.

    This class orchestrates the generation of synthetic pandas DataFrames based on
    YAML configuration. It handles column dependencies, references between tables,
    and provides deterministic data generation when seeded.

    The configuration structure supports:
    - Multiple namespaces for organizing related tables
    - Column references across tables (e.g., 'other_table.column')
    - Null value injection with configurable probabilities
    - Type inference and explicit type casting

    Configuration Structure:
        ```yaml
        namespace1:  # Optional namespace (defaults to 'default_fabrication')
            table1:
                num_rows: 100  # Optional if can be inferred from first column
                columns:
                    column1:
                        type: generate_unique_id  # Generator function to use
                        prefix: "ID_"            # Generator-specific args
                        id_length: 8
                        inject_nulls:            # Optional null injection
                            probability: 0.1
                            value: "MISSING"
                        dtype: string            # Optional type casting
                    column2:
                        type: faker
                        provider: name
                        localisation: en_US
                    column3:
                        type: row_apply
                        input_columns: [table1.column1, table1.column2]
                        row_func: "lambda x, y: f'{x}_{y}'"
            table2:
                num_rows: "@table1.num_rows"  # Reference other table's size
                columns:
                    column1:
                        type: copy_column
                        source_column: table1.column1
                        sample:
                            num_rows: "@table2.num_rows"
        ```

    Attributes:
        all_instructions (Dict[str, Any]): The parsed YAML instructions.
        seed (Optional[int]): Global seed for reproducible generation.
        all_dataframes (Dict[str, pd.DataFrame]): Generated DataFrames, keyed by
            'namespace.table_name'.

    Example:
        >>> # Basic usage
        >>> config = {
        ...     "patients": {
        ...         "num_rows": 100,
        ...         "columns": {
        ...             "id": {"type": "generate_unique_id", "prefix": "PAT_"},
        ...             "name": {"type": "faker", "provider": "name"},
        ...             "age": {
        ...                 "type": "numpy_random",
        ...                 "distribution": "normal",
        ...                 "loc": 45,
        ...                 "scale": 15
        ...             }
        ...         }
        ...     }
        ... }
        >>> generator = MockDataGenerator(instructions=config, seed=42)
        >>> dfs = generator.generate_all()
        >>> patients_df = dfs["default_fabrication.patients"]

    Notes:
        - Column generation order follows YAML definition order
        - Tables are generated in YAML definition order
        - References must point to already generated columns/tables
        - Seeding behavior cascades from global seed to column-specific seeds
        - Type inference follows pandas rules unless explicitly overridden
    """

    def __init__(self, instructions: Dict[str, Any], seed: Optional[int] = None):
        """
        Initializes the generator.

        Args:
            instructions: Dictionary loaded from the YAML configuration.
                          Can be {namespace: {table: {...}}}, or {table: {...}}.
            seed: Optional global random seed for reproducibility.
        """
        if not isinstance(instructions, dict):
            raise TypeError("Instructions must be a dictionary.")

        # Detect if instructions are single-namespace format ({table: conf}) and wrap if needed
        first_key = next(iter(instructions), None)
        wrapped_instructions = instructions
        if first_key and isinstance(instructions[first_key], dict) and "columns" in instructions[first_key]:
            assumed_namespace = "default_fabrication"
            logger.warning(
                f"Instructions appear to be single-namespace format {list(instructions.keys())}. "
                f"Wrapping under assumed namespace '{assumed_namespace}'. Cross-namespace references might fail."
            )
            wrapped_instructions = {assumed_namespace: instructions}
        else:
            logger.debug("Assuming multi-namespace structure for instructions.")

        self.all_instructions = wrapped_instructions
        self.seed = seed
        self.all_dataframes: Dict[str, pd.DataFrame] = {}

        if self.seed is not None:
            logger.info(f"Using global seed: {self.seed} for derivations.")

    def generate_all(self):
        """Generates all dataframes defined in the instructions."""
        logger.info("Starting data generation process.")
        instructions_to_process = self.all_instructions

        # Simple sequential generation based on YAML order.
        for namespace, tables in instructions_to_process.items():
            if not isinstance(tables, dict):
                logger.warning(f"Skipping invalid namespace entry '{namespace}': Expected a dictionary of tables.")
                continue
            logger.info(f"Processing namespace: {namespace}")
            for table_name, table_instructions in tables.items():
                if not isinstance(table_instructions, dict) or "columns" not in table_instructions:
                    logger.warning(
                        f"Skipping invalid table entry '{namespace}.{table_name}': Expected dict with 'columns'."
                    )
                    continue
                self.generate_dataframe(namespace, table_name, table_instructions)
        logger.info("Data generation complete.")
        return self.all_dataframes

    def generate_dataframe(self, namespace: str, table_name: str, table_instructions: Dict[str, Any]):
        """Generates a single dataframe."""
        full_table_name = f"{namespace}.{table_name}"
        logger.info(f"Generating dataframe: {full_table_name}")

        if full_table_name in self.all_dataframes:
            logger.warning(f"Dataframe {full_table_name} already generated. Skipping.")
            return

        num_rows = table_instructions.get("num_rows")
        if isinstance(num_rows, str) and num_rows.startswith("@"):
            try:
                num_rows = self._resolve_num_rows_reference(num_rows, namespace)
            except ValueError as e:
                logger.error(f"Cannot generate {full_table_name}: {e}")
                return
        elif not isinstance(num_rows, int) or num_rows < 0:
            num_rows = None
            logger.info(f"num_rows not specified or invalid for {full_table_name}. Will infer from first column.")

        column_definitions = table_instructions.get("columns", {})
        if not column_definitions:
            logger.warning(f"No columns defined for {full_table_name}. Creating empty dataframe.")
            self.all_dataframes[full_table_name] = pd.DataFrame()
            return

        temp_results: Dict[str, pd.array] = {}

        for column_name, column_instr in column_definitions.items():
            logger.info(f"Generating column: {full_table_name}.{column_name}")
            if not isinstance(column_instr, dict):
                logger.error(f"Invalid instruction for column '{column_name}' in '{full_table_name}'. Skipping.")
                continue

            current_col_instr = deepcopy(column_instr)

            if num_rows is not None and "num_rows" not in current_col_instr:
                generator_type = current_col_instr.get("type")
                if generator_type not in ["copy_column", "explode"]:
                    current_col_instr["num_rows"] = num_rows

            try:
                generated_column_data = self.generate_column(
                    namespace,
                    table_name,
                    column_name,
                    current_col_instr,
                    current_num_rows=num_rows,
                )

                if num_rows is None:
                    num_rows = len(generated_column_data)
                    logger.info(f"Inferred num_rows = {num_rows} from column {full_table_name}.{column_name}")

                temp_results[column_name] = generated_column_data
                # Update intermediate dataframe state to allow intra-table references
                self.all_dataframes[full_table_name] = pd.DataFrame(temp_results).copy()

            except Exception as e:
                logger.error(f"Failed to generate column {full_table_name}.{column_name}: {e}", exc_info=True)
                placeholder_val = None
                if num_rows is not None and num_rows > 0:
                    temp_results[column_name] = pd.array([placeholder_val] * num_rows)
                    # Update the partial dataframe even on error for subsequent columns
                    self.all_dataframes[full_table_name] = pd.DataFrame(temp_results).copy()
                else:
                    logger.error(
                        f"Cannot create placeholder for failed column {column_name} as num_rows is unknown or zero."
                    )
                # Re-raise the exception after logging and attempting placeholder
                raise

        final_df = pd.DataFrame(temp_results)
        self.all_dataframes[full_table_name] = final_df
        logger.info(f"Finished generating dataframe: {full_table_name} with shape {final_df.shape}")

    def generate_column(
        self,
        namespace: str,
        table_name: str,
        column_name: str,
        column_instr: Dict[str, Any],
        current_num_rows: Optional[int],
    ) -> pd.array:
        """Generates the data for a single column."""

        # Derive a deterministic seed for this column if a global seed exists
        column_seed: Optional[int] = None
        if self.seed is not None:
            column_seed_str = f"{self.seed}-{namespace}-{table_name}-{column_name}"
            # Use a hash function for better distribution and converting string to int seed
            hash_obj = hashlib.sha256(column_seed_str.encode())
            column_seed = int(hash_obj.hexdigest(), 16) % (2**32 - 1)  # Standard range for seeds
            logger.debug(f"Derived seed {column_seed} for column {namespace}.{table_name}.{column_name}")

        generator_type = column_instr.pop("type", None)
        if not generator_type:
            raise ValueError(f"Missing 'type' for column {namespace}.{table_name}.{column_name}")

        dtype = column_instr.pop("dtype", None)
        null_config = column_instr.pop("inject_nulls", None)
        if generator_type == "generate_random_numbers" and column_instr.get("integer") is True:
            if dtype:
                logger.warning(
                    f"'integer: True' overrides specified dtype '{dtype}' for generate_random_numbers. Using Int64."
                )
            dtype = "Int64"
            # Use pop to remove the key, ensuring it's not passed down if handled here
            column_instr.pop("integer")

        # Resolve references in known argument keys
        resolved_kwargs = {}
        for key, value in column_instr.items():
            if key in ["source_column", "input_columns", "sample_values"]:
                resolved_value = self._resolve_potential_reference(value, namespace)
                resolved_kwargs[key] = resolved_value
            elif (
                key == "sample"
                and isinstance(value, dict)
                and "num_rows" in value
                and isinstance(value["num_rows"], str)
            ):
                resolved_num_rows = self._resolve_num_rows_reference(value["num_rows"], namespace)
                resolved_kwargs[key] = {**value, "num_rows": resolved_num_rows}
            elif key == "num_rows" and isinstance(value, str) and value.startswith("@"):
                resolved_kwargs[key] = self._resolve_num_rows_reference(value, namespace)
            else:
                resolved_kwargs[key] = value

        # Special handling for explode: resolve references within explode_func_kwargs
        if generator_type == "explode" and "explode_func_kwargs" in resolved_kwargs:
            if isinstance(resolved_kwargs["explode_func_kwargs"], dict):
                resolved_explode_kwargs = {}
                for e_key, e_value in resolved_kwargs["explode_func_kwargs"].items():
                    resolved_explode_kwargs[e_key] = self._resolve_potential_reference(e_value, namespace)
                resolved_kwargs["explode_func_kwargs"] = resolved_explode_kwargs
            else:
                logger.warning(
                    f"Expected explode_func_kwargs to be a dict for column {column_name}, but got {type(resolved_kwargs['explode_func_kwargs'])}. Skipping internal reference resolution."
                )

        # Load and call the generator function
        try:
            if generator_type == "copy_column":
                generator_func = copy_column
            elif generator_type == "map_values":
                # Cast to the correct type for map_values
                generator_func = cast(Callable[..., List[Any]], map_values)
            elif generator_type == "hash_map":
                # Cast to the correct type for hash_map
                generator_func = cast(Callable[..., List[Any]], hash_map)
            else:
                generator_func = load_obj(generator_type)

            import inspect

            sig = inspect.signature(generator_func)

            # Pass derived column_seed if function accepts 'seed' and no specific seed is given
            if "seed" in sig.parameters and "seed" not in resolved_kwargs and column_seed is not None:
                # Check for specific seed overrides first (e.g., numpy_seed, faker_seed)
                has_specific_seed = any(k in resolved_kwargs for k in ["numpy_seed", "faker_seed"])
                if not has_specific_seed:
                    resolved_kwargs["seed"] = column_seed

            # Pass current_num_rows if the function expects it and it's not already set
            if "num_rows" in sig.parameters and "num_rows" not in resolved_kwargs and current_num_rows is not None:
                resolved_kwargs["num_rows"] = current_num_rows

            logger.debug(f"Calling {generator_type} with args: {resolved_kwargs}")
            # Add specific log before calling explode
            if generator_type == "explode":
                logger.info(
                    f"EXPLODE CALL: Passing num_rows={resolved_kwargs.get('num_rows')} to explode for column {column_name}"
                )
            generated_data = generator_func(**resolved_kwargs)

        except Exception as e:
            logger.error(f"Error executing generator '{generator_type}' for column {column_name}: {e}")
            raise

        # Apply null injection *after* data generation but *before* type conversion
        if null_config:
            logger.debug(f"Applying null injection for column {column_name} with config: {null_config}")
            # Determine a base seed for null injection consistency if specific seed not in null_config
            # Priority: null_config['seed'] > generator specific seed (e.g., numpy_seed) > column_seed > global seed
            null_injection_base_seed = null_config.get(
                "seed", resolved_kwargs.get("numpy_seed", resolved_kwargs.get("faker_seed", column_seed))
            )
            # The _apply_null_injection function handles the actual seed usage priority internally
            generated_data = _apply_null_injection(generated_data, null_config, base_seed=null_injection_base_seed)

        # Convert result to Pandas Array with specified dtype
        try:
            if dtype and ("datetime" in str(dtype) or "date" in str(dtype)):
                pd_array = pd.array(pd.to_datetime(generated_data, errors="coerce"), dtype=dtype)
            elif dtype == "Int64" and not all(
                isinstance(x, (int, type(None), np.integer)) or (isinstance(x, float) and x.is_integer())
                for x in generated_data
            ):
                pd_array = pd.array(
                    [
                        int(x) if pd.notna(x) and isinstance(x, (int, float)) and float(x).is_integer() else pd.NA
                        for x in generated_data
                    ],
                    dtype=dtype,
                )
            else:
                pd_array = pd.array(generated_data, dtype=dtype)

            if current_num_rows is not None and len(pd_array) != current_num_rows:
                raise ValueError(
                    f"Generated column '{column_name}' has length {len(pd_array)}, expected {current_num_rows}."
                )

            return pd_array
        except Exception as e:
            logger.error(f"Error converting column {column_name} to pandas array with dtype={dtype}: {e}")
            # Fallback to object type if conversion fails
            return pd.array(generated_data, dtype="object")

    def _resolve_potential_reference(self, value: Any, current_namespace: str) -> Any:
        """Resolves a value if it's a column reference string or a list of references."""
        if isinstance(value, str) and re.match(r"^[a-zA-Z0-9_.]+\.[a-zA-Z0-9_:]+$", value):
            # Single reference string like "table.column" or "namespace.table.column"
            return self._resolve_column_reference(value, current_namespace)
        elif (
            isinstance(value, list)
            and value
            and all(isinstance(item, str) and re.match(r"^[a-zA-Z0-9_.]+\.[a-zA-Z0-9_:]+$", item) for item in value)
        ):
            # List of reference strings
            return [self._resolve_column_reference(ref, current_namespace) for ref in value]
        else:
            return value

    def _resolve_column_reference(self, reference: str, current_namespace: str) -> List[Any]:
        """Parses 'namespace.table.column' or 'table.column' and returns the data as a list."""
        logger.debug(f"Resolving column reference: '{reference}' in namespace '{current_namespace}'")
        parts = reference.split(".")

        if len(parts) < 2:
            raise ValueError(
                f"Invalid column reference format: '{reference}'. Expected at least 'table.column' or 'namespace.table.column'."
            )

        column_name = parts[-1]
        table_identifier = ".".join(parts[:-1])

        # Determine the full table name (namespace.table) to look up
        if "." not in table_identifier:
            full_table_name = f"{current_namespace}.{table_identifier}"
        else:
            full_table_name = table_identifier

        # Check for existence, falling back to different lookup strategies
        if full_table_name not in self.all_dataframes:
            if table_identifier in self.all_dataframes:  # Check for direct source_dfs key
                full_table_name = table_identifier
            elif current_namespace == "default_fabrication" and "." not in table_identifier:
                # Check within default wrapper namespace if reference was relative
                potential_key = f"default_fabrication.{table_identifier}"
                if potential_key in self.all_dataframes:
                    full_table_name = potential_key
                    logger.debug(
                        f"Reference '{reference}' resolved via default_fabrication wrapper to '{full_table_name}'"
                    )
                else:
                    raise ValueError(
                        f"Referenced table '{full_table_name}' or '{potential_key}' (derived from '{reference}') not generated yet. Available: {list(self.all_dataframes.keys())}"
                    )
            else:
                raise ValueError(
                    f"Referenced table '{full_table_name}' (derived from '{reference}') has not been generated yet or does not exist. Available: {list(self.all_dataframes.keys())}"
                )

        df = self.all_dataframes[full_table_name]

        if column_name not in df.columns:
            raise ValueError(
                f"Referenced column '{column_name}' not found in table '{full_table_name}'. Available: {list(df.columns)}"
            )

        column_data = df[column_name].tolist()
        logger.debug(f"Successfully resolved reference '{reference}' to list of length {len(column_data)}")
        return column_data

    def _resolve_num_rows_reference(self, reference: str, current_namespace: str) -> int:
        """Parses '@table.num_rows' or '@namespace.table.num_rows' and returns the integer value."""
        logger.debug(f"Resolving num_rows reference: '{reference}' in namespace '{current_namespace}'")

        match = re.match(r"@((?:[a-zA-Z0-9_]+\.)*([a-zA-Z0-9_]+))\.num_rows$", reference)
        if not match:
            raise ValueError(
                f"Invalid num_rows reference format: '{reference}'. Expected '@table.num_rows' or '@namespace.table.num_rows'."
            )

        full_ref_path = match.group(1)
        target_table_name = match.group(2)

        # Determine namespace to search within original instructions
        namespace_to_use = current_namespace
        if "." in full_ref_path:
            namespace_to_use = ".".join(full_ref_path.split(".")[:-1])

        if namespace_to_use not in self.all_instructions:
            if "." not in full_ref_path and "default_fabrication" in self.all_instructions:
                namespace_to_use = "default_fabrication"
                logger.debug(f"Num_rows reference '{reference}' assumed relative within default_fabrication wrapper.")
            else:
                raise ValueError(
                    f"Namespace '{namespace_to_use}' for num_rows reference '{reference}' not found in instructions. Available: {list(self.all_instructions.keys())}"
                )

        if (
            not isinstance(self.all_instructions[namespace_to_use], dict)
            or target_table_name not in self.all_instructions[namespace_to_use]
            or not isinstance(self.all_instructions[namespace_to_use][target_table_name], dict)
            or "num_rows" not in self.all_instructions[namespace_to_use][target_table_name]
        ):
            raise ValueError(
                f"Could not find 'num_rows' definition for referenced table '{namespace_to_use}.{target_table_name}' (derived from '{reference}'). Keys in namespace: {list(self.all_instructions[namespace_to_use].keys())}"
            )

        num_rows_value = self.all_instructions[namespace_to_use][target_table_name]["num_rows"]

        # Resolve nested references recursively
        if isinstance(num_rows_value, str) and num_rows_value.startswith("@"):
            logger.debug(f"Resolving nested num_rows reference: {num_rows_value}")
            return self._resolve_num_rows_reference(num_rows_value, current_namespace)

        if not isinstance(num_rows_value, int):
            raise ValueError(
                f"'num_rows' for referenced table '{namespace_to_use}.{target_table_name}' resolved to non-integer value: {num_rows_value} ({type(num_rows_value)})."
            )

        logger.debug(f"Resolved num_rows reference '{reference}' to {num_rows_value}")
        return num_rows_value


# Prefixes used to identify temporary dataframes that should not be persisted.
_IGNORE_DATAFRAMES_WITH_PREFIX = ["_TEMP_", "_CATALOG_"]


def fabricate_datasets(
    fabrication_params: Dict[str, Any],
    ignore_prefix: Optional[List[str]] = None,
    seed: Optional[int] = None,
    **source_dfs: Dict[str, Union[pd.DataFrame, pyspark.sql.DataFrame]],
) -> Dict[str, pd.DataFrame]:
    """Fabricates datasets.

    This node passes configuration to ``MockDataGenerator`` data fabricator to fabricate
    datasets.

    Args:
        fabrication_params: Fabrication parameters to pass to ``MockDataGenerator``.
        ignore_prefix: List of prefixes for temporary dataframes to ignore in output.
        seed: Optional random seed for reproducibility.
        source_dfs: Optional real-world dataframes to add to ``MockDataGenerator``.
                  Keys should match references used in `fabrication_params` if intended
                  to be used as sources.

    Returns:
        A dictionary with the fabricated pandas dataframes, excluding temporary ones.
    """
    ignore_prefix_list = ignore_prefix if ignore_prefix is not None else _IGNORE_DATAFRAMES_WITH_PREFIX
    mock_generator = MockDataGenerator(instructions=fabrication_params, seed=seed)

    # Add source dataframes to the generator's internal state
    if source_dfs:
        for df_name, df in source_dfs.items():
            if isinstance(df, pd.DataFrame):
                mock_generator.all_dataframes[df_name] = df
            elif isinstance(df, pyspark.sql.DataFrame):
                # Convert Spark dataframes to Pandas for consistency within the generator
                mock_generator.all_dataframes[df_name] = df.toPandas()

    mock_generator.generate_all()

    # Construct the output dictionary, mapping simple table names to generated dataframes
    output = {}
    for namespace, tables in mock_generator.all_instructions.items():
        for table_name in tables.keys():
            full_name = f"{namespace}.{table_name}"
            if not any(table_name.startswith(prefix) for prefix in ignore_prefix_list):
                if full_name in mock_generator.all_dataframes:
                    output[table_name] = mock_generator.all_dataframes[full_name]
                else:
                    logger.warning(f"Expected output dataframe '{full_name}' was not found in generated results.")

    return output
