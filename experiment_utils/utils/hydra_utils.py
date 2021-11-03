import importlib
from functools import partial
from typing import Any


def load_obj(obj_path: str, default_obj_path: str = "") -> Any:
    """Extract an object from a given path.
        Args:
            obj_path: Path to an object to be extracted, including the object name.
            default_obj_path: Default object path.
        Returns:
            Extracted object.
        Raises:
            AttributeError: When the object does not have the given named attribute.
        https://github.com/quantumblacklabs/kedro/blob/9809bd7ca0556531fa4a2fc02d5b2dc26cf8fa97/kedro/utils.py
    """
    obj_path_list = obj_path.rsplit(".", 1)
    obj_path = obj_path_list.pop(0) if len(obj_path_list) > 1 else default_obj_path
    obj_name = obj_path_list[0]
    module_obj = importlib.import_module(obj_path)
    if not hasattr(module_obj, obj_name):
        raise AttributeError(
            "Object `{}` cannot be loaded from `{}`.".format(obj_name, obj_path)
        )
    return getattr(module_obj, obj_name)


def load_partial_func(**kwargs):
    """Return partial func, _target_ as function object with paremeters from hydra config.
    args in config file:
    function: function name, full path or name
    function_path: optional, path to function
    """
    obj_path = kwargs.pop('function', None)
    default_obj_path = kwargs.pop('function_path', '')
    func = load_obj(obj_path, default_obj_path)
    return partial(func, **kwargs)


def load_args_list(**kwargs):
    '''Return list of args. For instantiate list of obj by Hydra'''
    return list(kwargs.values())


def call_class(**kwargs) -> Any:
    """Populate obj from 'class_name', than call it."""
    class_name = kwargs.pop('class_name', None)
    obj = load_obj(class_name)
    return obj(**kwargs)()
