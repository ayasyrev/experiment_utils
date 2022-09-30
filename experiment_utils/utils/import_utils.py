import importlib
from functools import partial
from pathlib import Path, PosixPath
from typing import Any, List, Optional, Union

import experiment_utils
import hydra
import torch
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig
from pydantic import BaseModel


class FuncCfg(BaseModel):
    name: str = ""
    path: str | None = None


def load_obj(func: FuncCfg):
    if func.path is None:
        obj_path, obj_name = func.name.rsplit(".", 1)
    else:
        obj_path = func.path
        obj_name = func.name

    module = importlib.import_module(obj_path)
    obj = getattr(module, obj_name, None)

    if obj is None:
        raise AttributeError(
            f"Object `{obj_name}` cannot be loaded from `{obj_path}`."        )
    return obj


def load_obj_run(func: FuncCfg):
    return load_obj(func)()


def load_obj_partial(func: FuncCfg):
    return partial(load_obj(func), **func.dict(exclude={"name", "path"}))

# def load_obj(obj_path: str, default_obj_path: str = "") -> Any:
#     """Extract an object from a given path.
#         Args:
#             obj_path: Path to an object to be extracted, including the object name.
#             default_obj_path: Default object path.
#         Returns:
#             Extracted object.
#         Raises:
#             AttributeError: When the object does not have the given named attribute.
#         https://github.com/quantumblacklabs/kedro/blob/9809bd7ca0556531fa4a2fc02d5b2dc26cf8fa97/kedro/utils.py
#     """
#     obj_path_list = obj_path.rsplit(".", 1)
#     obj_path = obj_path_list.pop(0) if len(obj_path_list) > 1 else default_obj_path
#     obj_name = obj_path_list[0]
#     module_obj = importlib.import_module(obj_path)
#     if not hasattr(module_obj, obj_name):
#         raise AttributeError(
#             "Object `{}` cannot be loaded from `{}`.".format(obj_name, obj_path)
#         )
#     return getattr(module_obj, obj_name)


# def load_func(**kwargs):
#     """Return func, _target_ as function object with parameters from hydra config.
#     args in config file:
#     function: function name, full path or name
#     function_path: optional, path to function
#     """
#     obj_path = kwargs.pop("function", "")
#     default_obj_path = kwargs.pop("function_path", "")
#     func = load_obj(obj_path, default_obj_path)
#     return func




# def load_partial_func(**kwargs):
#     """Return partial func, _target_ as function object with parameters from hydra config.
#     args in config file:
#     function: function name, full path or name
#     function_path: optional, path to function
#     """
#     obj_path = kwargs.pop("function", "")
#     default_obj_path = kwargs.pop("function_path", "")
#     func = load_obj(obj_path, default_obj_path)
#     return partial(func, **kwargs)


# def load_args_list(**kwargs):
#     """Return list of args. For instantiate list of obj by Hydra"""
#     return list(kwargs.values())


# def call_class(**kwargs) -> Any:
#     """Populate obj from 'class_name', than call it."""
#     class_name = kwargs.pop("class_name", None)
#     class_path = kwargs.pop("class_path", "")
#     obj = load_obj(obj_path=class_name, default_obj_path=class_path)
#     return obj(**kwargs)()


# def class_obj(**kwargs) -> Any:
#     """Populate obj from 'class_name', than call it."""
#     class_name = kwargs.pop("class_name", None)
#     class_path = kwargs.pop("class_path", "")
#     obj = load_obj(obj_path=class_name, default_obj_path=class_path)
#     return obj(**kwargs)


# def load_model(**kwargs) -> Any:
#     """Populate obj from 'class_name', than call it."""
#     model_name = kwargs.pop("model_name", None)
#     model_path = kwargs.pop("model_path", "")
#     obj = load_obj(obj_path=model_name, default_obj_path=model_path)
#     return obj(**kwargs)


# experiment_utils_path = experiment_utils.__path__[0]


# def read_config(
#     overrides: List[str] = [],
#     config_dir_name: str = "conf",
#     config_path: Optional[Union[str, PosixPath, Path]] = None,
#     config_name: str = "config",
# ) -> DictConfig:
#     """Read and Initialise Hydra config when work in jupyter notebook.

#     Args:
#         overrides (List[str], optional): List of overrides. Defaults to [].
#         config_dir_name (str, optional): Name of directory with config structure.
#             Defaults to "conf".
#         config_path (Union[str, PosixPath], optional): Path to look for config dir.
#             Defaults to None, take it from experiment_utils lib.
#         config_name (str, optional): Name for config file. Defaults to "config".

#     Returns:
#         DictConfig: Initialized DictConfig.
#     """
#     if config_path is None:
#         config_path = Path(experiment_utils_path)
#     else:
#         config_path = Path(config_path)
#     config_dir = (config_path / config_dir_name).absolute()
#     with initialize_config_dir(config_dir=str(config_dir), version_base="1.1"):
#         cfg = compose(config_name=config_name, overrides=overrides)
#     return cfg


# def instantiate_model(cfg: DictConfig) -> torch.nn.Module:
#     model: torch.nn.Module = hydra.utils.instantiate(cfg.model, _convert_="all")
#     return model
