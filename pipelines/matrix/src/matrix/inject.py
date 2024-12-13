import functools
import importlib
from copy import deepcopy
from types import BuiltinFunctionType, FunctionType
from typing import Any, Dict, List, Tuple

OBJECT_KW = "object"
INSTANTIATE_KW = "instantiate"
UNPACK_KW = "unpack"


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
                instantiated_obj = obj(**new_dict)
            else:
                instantiated_obj = obj

        # for classes
        else:
            if instantiate is False:
                instantiated_obj = obj
            else:
                instantiated_obj = obj(**new_dict)

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
