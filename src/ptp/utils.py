from copy import copy, deepcopy

from typing import List
from omegaconf import DictConfig, OmegaConf


def instantiate(config: dict | DictConfig, recursive: bool = True, **kwargs):
    """
    (Recursively) instantiate objects from a configuration dictionary.

    Args:
        config (dict): Configuration dictionary with a '_target_' key specifying the class to instantiate.
        recursive (bool): Whether to recursively instantiate nested configurations. Default is True.
        **kwargs: Additional keyword arguments to pass to the class constructor
    Returns:
        Instantiated object or None if config is None.
    """
    if config is None:
        return None
    if isinstance(config, DictConfig):
        config = OmegaConf.to_object(config)
    target = config.pop('_target_')
    module_path, class_name = target.rsplit('.', 1)
    module = __import__(module_path, fromlist=[class_name])
    cls = getattr(module, class_name)
    if recursive:
        for key, value in config.items():
            if isinstance(value, dict) and '_target_' in value:
                config[key] = instantiate(value, recursive=True)
            elif isinstance(value, list):
                config[key] = [instantiate(item, recursive=True) if isinstance(item, dict) and '_target_' in item else item for item in value]
    return cls(**config, **kwargs)


def overwrite(config: dict | DictConfig, updates: dict | DictConfig) -> dict | DictConfig:
    """
    Overwrite values in a configuration dictionary with values from another dictionary.

    Args:
        config (dict | DictConfig): Original configuration dictionary.
        updates (dict | DictConfig): Dictionary with values to overwrite in the original configuration.

    Returns:
        dict: Updated configuration dictionary.
    """
    updated_config = deepcopy(config)
    for key, value in updates.items():
        if key in updated_config and isinstance(updated_config[key], dict) and isinstance(value, dict):
            updated_config[key] = overwrite(updated_config[key], value)
        else:
            updated_config[key] = value
    return updated_config
