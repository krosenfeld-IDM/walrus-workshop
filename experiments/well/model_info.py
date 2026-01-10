"""
Print the model architecture
"""

from walrus_workshop import paths
from omegaconf import OmegaConf
import torch
from hydra.utils import instantiate

# Change the working directory
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Load config
config = OmegaConf.load(paths.configs/"well_config.yaml")

# Load model
checkpoint = torch.load(paths.checkpoints/"walrus.pt", map_location="cpu", weights_only=True)["app"][
    "model"
]

# The dataset objects precompute a number of dataset stats on init, so this may take a little while
data_module = instantiate(
    config.data.module_parameters,
    well_base_path=paths.data/"datasets",
    world_size=1,
    rank=0,
    data_workers=1,
    field_index_map_override=config.data.get(
        "field_index_map_override", {}
    ),  # Use the previous field maps to avoid cycling through the data
    prefetch_field_names=False,
)
field_to_index_map = data_module.train_dataset.field_to_index_map
total_input_fields = max(field_to_index_map.values()) + 1

model = instantiate(
    config.model,
    n_states=total_input_fields,
)
model.load_state_dict(checkpoint)

print(model)