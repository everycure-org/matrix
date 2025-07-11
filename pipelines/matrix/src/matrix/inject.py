import functools
import importlib
import inspect
import logging
import re
from copy import deepcopy
from inspect import getfullargspec
from types import BuiltinFunctionType, FunctionType
from typing import Any, Dict, List, Tuple

OBJECT_KW = "_object"
INSTANTIATE_KW = "instantiate"
UNPACK_KW = "unpack"

logger = logging.getLogger(__file__)


def _load_obj(obj_path: str, default_obj_path: str = None) -> Any:
    """Extract an object from a given path.

    Args:
        obj_path: Path to an object to be extracted, including the object name.
        default_obj_path: Default object path.

    Returns:
        Extracted object.

    Raises:
        AttributeError: When the object does not have the given named attribute.
    """
    obj_path_list = obj_path.rsplit(".", 1)
    obj_path = obj_path_list.pop(0) if len(obj_path_list) > 1 else default_obj_path
    obj_name = obj_path_list[0]
    module_obj = importlib.import_module(obj_path)
    if not hasattr(module_obj, obj_name):
        raise AttributeError(
            "Object `{}` cannot be loaded from `{}`.".format(  # pylint: disable=consider-using-f-string,# noqa: E501
                obj_name, obj_path
            )
        )
    return getattr(module_obj, obj_name)


def _parse_for_objects(param, exclude_kwargs: List[str] = None) -> Dict:
    """Recursively searches a dictionary and converts declarations to objects."""
    exclude_kwargs = exclude_kwargs or []

    def _find_object_keyword(object_dict: dict):
        object_path = object_dict.pop(OBJECT_KW)
        new_dict = {}

        for key, value in object_dict.items():
            if key in exclude_kwargs:
                # stop recursion if exclude_kwargs found
                new_dict[key] = value
            else:
                if isinstance(value, dict):
                    if OBJECT_KW in value.keys():
                        new_dict[key] = _find_object_keyword(value)
                    else:
                        new_dict[key] = _parse_for_objects(value, exclude_kwargs=exclude_kwargs)
                else:
                    new_dict[key] = _parse_for_objects(value, exclude_kwargs=exclude_kwargs)

        instantiate = new_dict.pop(INSTANTIATE_KW, None)
        obj = _load_obj(object_path)

        # for functions
        if isinstance(obj, (BuiltinFunctionType, FunctionType)):
            if new_dict or instantiate:
                instantiated_obj = functools.partial(obj, **new_dict)
                try:
                    # backwards compatibility check where some invocations of inject_object expect the function to be fully called
                    instantiated_obj = instantiated_obj()
                except TypeError:
                    pass  # keep it as partial
            else:
                instantiated_obj = obj

        # for classes
        elif inspect.isclass(obj):
            if instantiate is False:
                instantiated_obj = obj
            else:
                instantiated_obj = obj(**new_dict)
        else:
            return obj

        return instantiated_obj

    if isinstance(param, dict):
        new_dict = {}
        if OBJECT_KW in param.keys():
            param = deepcopy(param)
            instantiated_obj = _find_object_keyword(param)
            return instantiated_obj

        for key, value in param.items():
            if key in exclude_kwargs:
                # stop recursion if exclude_kwargs found
                new_dict[key] = value
            else:
                if isinstance(value, dict):
                    if OBJECT_KW in value.keys():
                        value = deepcopy(value)
                        new_dict[key] = _find_object_keyword(value)
                    else:
                        new_dict[key] = _parse_for_objects(value, exclude_kwargs=exclude_kwargs)
                else:
                    new_dict[key] = _parse_for_objects(value, exclude_kwargs=exclude_kwargs)
        return new_dict

    if isinstance(param, Tuple):
        return tuple(_parse_for_objects(e, exclude_kwargs=exclude_kwargs) for e in param)
    if isinstance(param, List):
        return [_parse_for_objects(e, exclude_kwargs=exclude_kwargs) for e in param]
    return param


def _inject_object(*args, exclude_kwargs: List[str] = None, **kwargs) -> None:
    """Recursively searches a keyword `object` and load the Python object.

    Declarations of objects follow a certain pattern:
    ::
       # parameters.yml
       params:my_parameters:
           object: path.to.SimpleImputer
           arg: ...

      # nodes.yml
      my_func:
        func: dummy_func
        inputs: params:my_parameters
        outputs: xx

      # nodes.py
      @augment()
      def my_func_pd(*args, **kwargs):
          def dummy_func(*args, **kwargs)
      `dummy_func` will receive an instance of `SimpleImputer(...)` class.
    ::

    """
    parsed_args = _parse_for_objects(args, exclude_kwargs=exclude_kwargs or [])
    dictionary_kwargs = _parse_for_objects(kwargs, exclude_kwargs=exclude_kwargs or [])

    return parsed_args, dictionary_kwargs


def inject_object(exclude_kwargs: List[str] = None):
    """Inject object decorator."""

    def decorate(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            args, kwargs = _inject_object(*args, exclude_kwargs=exclude_kwargs, **kwargs)
            result_df = func(*args, **kwargs)

            return result_df

        return wrapper

    return decorate


def _unpack_params(*args, **kwargs):
    """Unpacks top level dictionaries in args and kwargs by 1 level.

    Most beneficial if used as part of a kedro node. In cases where we need to pass
    lots of parameters to function, we can use this unpack decorator where we use
    unpack arg or kwarg which will contain most of the parameters.

    To enable unpacking, the keyword "unpack" can be provided via a dictionary in
    args and/or kwargs.
    """
    unpacked_kwargs_from_args = {}
    args = list(args)
    remove_list = []
    for i, arg in enumerate(args):
        if isinstance(arg, dict):
            if UNPACK_KW in arg.keys():
                remove_list.append(i)
                new_kwargs = args[i]
                unpacked_kwargs_from_args = {
                    **unpacked_kwargs_from_args,
                    **new_kwargs.pop(UNPACK_KW),
                    **new_kwargs,
                }
    args = [i for j, i in enumerate(args) if j not in remove_list]

    if UNPACK_KW in kwargs:
        unpack_kwargs_from_kwargs = kwargs.pop(UNPACK_KW, {})
    else:
        unpack_kwargs_from_kwargs = {}

    return args, {**unpacked_kwargs_from_args, **kwargs, **unpack_kwargs_from_kwargs}


def unpack_params():  # pylint: disable=missing-return-type-doc
    """Unpack params decorator.

    Returns:
        Wrapper function.
    """

    def decorate(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            args, kwargs = _unpack_params(*args, **kwargs)
            result = func(*args, **kwargs)

            return result

        return wrapper

    return decorate


def _extract_elements_in_list(
    full_list_of_columns: List[str],
    list_of_regexes: List[str],
    raise_exc,
) -> List[str]:
    """Use regex to extract elements in a list."""
    results = []
    for regex in list_of_regexes:
        matches = list(filter(re.compile(regex).match, full_list_of_columns))
        if matches:
            for match in matches:
                if match not in results:
                    logger.info("The regex %s matched %s.", regex, match)
                    results.append(  # helps keep relative ordering as defined in YAML
                        match
                    )
        else:
            if raise_exc:
                raise ValueError(f"The following regex did not return a result: {regex}.")
            logger.warning("The following regex did not return a result: %s", regex)
    return results


def make_list_regexable(
    source_df: str = None,
    make_regexable_kwarg: str = None,
    raise_exc: bool = False,
):
    """Allow processing of regex in input list.
    Args:
        source_df: Name of the dataframe containing actual list columns names.
        make_regexable: Name of list with regexes.
        raise_exc: Whether to raise an exception or just log the warning.
           Defaults to False.
    Returns:
        A wrapper function
    """

    def _decorate(func):
        @functools.wraps(func)
        def _wrapper(
            *args,
            source_df=source_df,
            make_regexable_kwarg=make_regexable_kwarg,
            raise_exc=raise_exc,
            **kwargs,
        ):
            argspec = getfullargspec(func)
            all_args = argspec.args + argspec.kwonlyargs

            if source_df not in all_args:
                raise ValueError("Please provide source dataframe.")

            if (make_regexable_kwarg is not None) and (make_regexable_kwarg in all_args):
                df = kwargs.get(source_df) if source_df in kwargs else args[all_args.index(source_df)]
                make_regexable_list = (
                    kwargs.get(make_regexable_kwarg)
                    if make_regexable_kwarg in kwargs
                    else args[all_args.index(make_regexable_kwarg)]
                )

                if make_regexable_list is not None:
                    df_columns = df.columns
                    new_columns = _extract_elements_in_list(
                        full_list_of_columns=df_columns,
                        list_of_regexes=make_regexable_list,
                        raise_exc=raise_exc,
                    )
                    if not new_columns:
                        raise ValueError(
                            f"No columns were selected using the provided regex patterns: {make_regexable_list} from available columns: {df_columns}"
                        )

                    if make_regexable_kwarg in kwargs:
                        kwargs[make_regexable_kwarg] = new_columns
                    else:
                        args = [
                            (new_columns if i == all_args.index(make_regexable_kwarg) else arg)
                            for (i, arg) in enumerate(args)
                        ]

            result_df = func(*args, **kwargs)

            return result_df

        return _wrapper

    return _decorate
