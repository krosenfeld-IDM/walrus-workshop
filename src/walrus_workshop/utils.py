import inspect
from pathlib import Path
import yaml
import os

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
