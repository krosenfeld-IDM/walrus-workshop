import inspect
from pathlib import Path
import yaml
import os
import re

def get_keyvalue_from_string(string: str):
    """
    Extract the key and value from a string in the format "KeyValue".
    """

    if match := re.match(r'([a-zA-Z]+)([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)', string):
        return match
    else:
        return None

def get_filename_post_fix(trajectory_index: int, kv_dict: dict) -> str:
    return f"traj_{trajectory_index}_" + "_".join([f"{k}_{v:0.0e}" for k, v in kv_dict.items()])    

def load_config(config_path: str | Path | None = None) -> dict:
    """
    Load training configuration from YAML file.

    Args:
        config_path: Path to config file. If None, looks for train.yaml
                     in the local configs directory.

    Returns:
        Dictionary containing configuration
    """
    if config_path is None:
        script_dir = Path(os.getcwd())
        config_path = script_dir / "configs" / "train.yaml"
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config

def filter_kwargs(kwargs, module):
    """
    Filter a dictionary of kwargs to only include keys that the module accepts.
    """

    # Get the list of arguments SAE expects
    module_args = set(inspect.signature(module.__init__).parameters)

    # Create a filtered dict containing ONLY keys that SAE accepts
    model_kwargs = {k: v for k, v in kwargs.items() if k in module_args}
    return model_kwargs
