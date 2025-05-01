# NOTE: This file was partially generated using AI assistance.
import datetime
import hashlib
import importlib
import itertools
import json
import logging
import random
import re
from copy import deepcopy
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, cast

import numpy as np
import pandas as pd
import pyspark
import yaml
from faker import Faker
from pandas.api.types import is_datetime64_any_dtype, is_list_like

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Default path for loading functions if namespace isn't specified
_DEFAULT_OBJ_PATH = __name__


def load_obj(obj_path: str, default_obj_path: str = _DEFAULT_OBJ_PATH) -> Any:
    """
    Extract an object from a given path.
    Dynamically imports modules if necessary.
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
    """
    Evaluates a string function (like a lambda) to convert into a callable.
    Imports libraries mentioned in the string (e.g., 'math.sqrt').
    WARNING: Uses eval(), which can be a security risk if the input string is untrusted.
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
    """
    Injects null values into a list based on configuration.

    Args:
        data: The list of data to process.
        null_config: Dictionary with 'probability', optional 'value', optional 'seed'.
        base_seed: An optional base seed for the random number generator.

    Returns:
        The list with nulls potentially injected.
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
    num_rows: int,
    prefix: str = "",
    id_length: Optional[int] = None,
    seed: Optional[int] = None,  # Add seed parameter for reproducibility
) -> List[str]:
    # NOTE: This function was partially generated using AI assistance.
    """Generate a list of unique-ish random numeric IDs.

    Generates random numeric strings, applies a prefix, and adjusts to a fixed
    length if specified. Uniqueness is highly probable due to large random
    number generation but not strictly guaranteed for all possible inputs.
    The `id_start_range` and `id_end_range` parameters are no longer supported.

    Args:
        num_rows: The number of IDs to generate.
        prefix: A string to prepend to each ID.
        id_length: The total desired length of the final ID string (prefix + number).
                   If specified, numeric parts are zero-padded or truncated to fit.
        seed: An optional seed for the random number generator.

    Returns:
        A list of generated ID strings.

    Raises:
        ValueError: If `id_length` is specified and is not long enough to
                    accommodate the prefix plus at least one digit.
    """
    if num_rows <= 0:
        return []

    # Use a seeded random number generator if a seed is provided
    rng = np.random.default_rng(seed)

    # Generate large random integers to maximize uniqueness
    # Using a large upper bound (e.g., 10**18)
    # Note: For extremely high num_rows, collisions are still possible, though unlikely.
    # True uniqueness guarantee would require checking and regenerating, adding complexity.
    random_numbers = rng.integers(low=0, high=int(1e18), size=num_rows)
    numeric_strings = [str(n) for n in random_numbers]

    final_ids = []

    if id_length is not None:
        if not isinstance(id_length, int) or id_length <= 0:
            raise ValueError("id_length must be a positive integer.")

        numeric_part_length = id_length - len(prefix)
        if numeric_part_length <= 0:
            raise ValueError(
                f"id_length ({id_length}) is too short for the prefix ('{prefix}'). "
                f"It must be longer than the prefix length ({len(prefix)})."
            )

        for num_str in numeric_strings:
            if len(num_str) < numeric_part_length:
                # Pad with leading zeros
                padded_num_str = num_str.zfill(numeric_part_length)
            elif len(num_str) > numeric_part_length:
                # Truncate (take the end part to preserve randomness)
                padded_num_str = num_str[-numeric_part_length:]
            else:
                padded_num_str = num_str
            final_ids.append(f"{prefix}{padded_num_str}")
    else:
        # No length constraint, just prepend prefix with 10 digit number
        final_ids = [f"{prefix}{num_str.zfill(10)}" for num_str in numeric_strings]

    return final_ids


def faker(
    num_rows: int,
    provider: str,
    localisation: Optional[Union[str, List[str]]] = None,
    provider_args: Optional[Dict[str, Union[str, int]]] = None,
    faker_seed: Optional[int] = None,
    seed: Optional[int] = None,  # Derived column seed
) -> List[Any]:
    """Thin wrapper for accessing Faker properties."""
    if num_rows <= 0:
        return []

    # Use specific faker_seed if provided, otherwise use derived column seed
    effective_seed = faker_seed if faker_seed is not None else seed

    # Create a unique Faker instance for each call to isolate seeding if needed
    # NOTE: Using locale might still share some global state depending on Faker version
    faker_obj = Faker(localisation)

    if effective_seed is not None:
        # Seed the specific instance. Seeding global Faker is less reliable.
        faker_obj.seed_instance(effective_seed)
        logger.debug(f"Seeding Faker instance with seed: {effective_seed} for provider {provider}")
    # else: # Removed global seeding fallback
    # logger.warning(f"Using unseeded Faker instance for provider {provider}.")

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


def numpy_random(num_rows: int, distribution: str, numpy_seed: Optional[int] = None, **kwargs) -> List[Any]:
    """Wrapper for numpy.random distributions."""
    if num_rows <= 0:
        return []
    # Use a dedicated Generator for seeded runs
    if numpy_seed is not None:
        rng = np.random.Generator(np.random.PCG64(numpy_seed))
        logger.debug(f"Using numpy seeded generator with seed: {numpy_seed}")
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
    """Generate random arrays (lists) with elements from sample_values."""
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
    """
    Generate values by sampling from a list or using weighted choice from a dict.
    Handles up/down sampling based on num_rows vs available unique values.
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
    """Maps values from input_column based on the provided mapping dictionary."""
    return [mapping.get(key, default_value) for key in input_column]


def hash_map(
    input_column: List[Any],
    buckets: List[Any],
) -> List[Any]:
    """Hashes values from input_column to one of the provided buckets."""
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
    """Generates a range of dates, with up/down sampling."""
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
    """Generates the Cartesian product of the input lists, returns the column at 'position'."""
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
    """Applies a function to the entire list(s) representing column(s)."""
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
    """Applies a function element-wise to rows formed by zipping input columns."""
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
    """Generates fabricated data based on declarative instructions."""

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
        """Resolves a value if it's a column reference string or list of references."""
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
