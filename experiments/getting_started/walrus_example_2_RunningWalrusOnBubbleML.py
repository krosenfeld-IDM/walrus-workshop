# Metadata: WellMetadata(dataset_name='bubbleML_PoolBoiling-Subcooled', 
# n_spatial_dims=3, spatial_resolution=(512, 512, 1), 
# scalar_names=[], constant_scalar_names=['inv_reynolds', 'cpgas', 'mugas', 'rhogas', 
# 'thcogas', 'stefan', 'prandtl', 'heater-nucWaitTime', 'heater-wallTemp'], 
# field_names={0: ['gas-interface-sdf', 'temperature'], 
# 1: ['velocity_x', 'velocity_y', 'velocity_z'], 
# 2: []}, constant_field_names={0: [], 1: [], 2: []}, 
# boundary_condition_types=['OPEN', 'WALL', 'PERIODIC'], 
# n_files=2, n_trajectories_per_file=[1, 1], n_steps_per_trajectory=[2001, 2001], 
# grid_type='cartesian')
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf, open_dict
from the_well.benchmark.metrics import make_video
from the_well.data.utils import flatten_field_names
from walrus.data.inflated_dataset import InflatedWellDataset
from walrus.data.multidataset import MixedWellDataset
from walrus.data.well_to_multi_transformer import ChannelsFirstWithTimeFormatter
from walrus.models import IsotropicModel
from walrus.trainer.normalization_strat import (
    normalize_target,
)
from walrus.utils.experiment_utils import align_checkpoint_with_field_to_index_map
from walrus_workshop import paths
# Change working directory
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

checkpoint_base_path = paths.checkpoints
config_base_path = paths.configs
checkpoint_path = os.path.join(checkpoint_base_path, "walrus.pt")
checkpoint_config_path = os.path.join(config_base_path, "extended_config.yaml")
checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)["app"][
    "model"
]
config = OmegaConf.load(checkpoint_config_path)

# Lets start by examining our config file
# print(OmegaConf.to_yaml(config))


# The dataset objects precompute a number of dataset stats on init, so this may take a little while
well_base_path = os.path.join(paths.data, "datasets")
data_module = instantiate(
    config.data.module_parameters,
    well_base_path=well_base_path,
    world_size=1,
    rank=0,
    data_workers=1,
    field_index_map_override=config.data.get(
        "field_index_map_override", {}
    ),  # Use the previous field maps to avoid cycling through the data
    prefetch_field_names=False,
)


import copy
field_to_index_map = data_module.train_dataset.field_to_index_map
new_field_to_index_map = copy.deepcopy(field_to_index_map)
# new_field_to_index_map["gas-interface-sdf"] = max(field_to_index_map.values()) + 1  # New index
# Retrieve the number of fields used in training
# from the mapping of field to index and incrementing by 1
total_input_fields = max(new_field_to_index_map.values()) + 1
# total_input_fields = 67  # We don't have all the Well so we grab from the errors
model: torch.nn.Module = instantiate(
    config.model,
    n_states=total_input_fields,
)

# # Use the Walrus utility to align the checkpoint
# revised_model_checkpoint = align_checkpoint_with_field_to_index_map(
#     checkpoint_state_dict=checkpoint,
#     model_state_dict=model.state_dict(),
#     checkpoint_field_to_index_map=field_to_index_map,
#     model_field_to_index_map=new_field_to_index_map,
# )

# model.load_state_dict(revised_model_checkpoint)
model.load_state_dict(checkpoint)

# Move to the device we want
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()


formatter = ChannelsFirstWithTimeFormatter()
revin = instantiate(config.trainer.revin)()  # This is a functools partial by default


# Before we use the model, let's look at the structure of the data. This will give us important information for how we can use the model when we don't have Well-formatted data.

# In[6]:

bubbleml_config = config

# checkpoint_config_path = os.path.join(".", "configs", "extended_bubbleML_config.yaml")
# bubbleml_config = OmegaConf.load(checkpoint_config_path)

# The dataset object with bubbleml
bubbleml_data_module = instantiate(
    bubbleml_config.data.module_parameters,
    well_base_path=well_base_path,
    world_size=1,
    rank=0,
    data_workers=1,
    field_index_map_override=bubbleml_config.data.get(
        "field_index_map_override", {}
    ),  # Use the previous field maps to avoid cycling through the data
    prefetch_field_names=False,
)

# Grab one trajectory to use as an example
dataset_index = 0
dataset = bubbleml_data_module.rollout_val_datasets[dataset_index].sub_dsets[0]
metadata = dataset.metadata

trajectory_example = next(iter(bubbleml_data_module.rollout_val_dataloaders()[dataset_index]))

print("Metadata:", metadata)
print("Trajectory example keys:", trajectory_example.keys())


# The `MixedWellDataset` object outputs several important fields:
# - `input_fields` - These are time varying state variables to use as input.
# - `out_fields` - Time varying state variables that the model is expected to predict.
# - `constant_fields` - Input values that don't vary with time.
# - `boundary_conditions` - list of lists containing the properties of the borders. These are restricted to topological details "periodic", "open", and "wall"/closed. For instance, the values [[0, 0], [1, 0], [2, 2]] would indicate the first axis (x) has closed boundaries on both sides, while the second (y) has an open boundary at ($y=0$) and closed at ($y=1$). The third (z) is periodic.
# - `padded_field_mask` - When applying rotations, it can be difficult to tell which tensor-valued fields are real and which are from padding into higher dimensions. This lets us know which are which so we can evaluate only the true fields.
# - `field_indices` - This is a new object in Walrus's `MixedWellDataset` that tracks mappings from field names to indices. This tells the model what types of fields it's using at any given time.
# - `metadata` - This is a typical Well metadata object, but as this type of object can sample from multiple data sources, this is passed to make it easier to track what type of data we're working with.
#
# ```
# # Boundary condition codes
# class BoundaryCondition(Enum):
#     WALL = 0
#     OPEN = 1
#     PERIODIC = 2
# ```
#
# Now lets see how we can use the model to forecast the evolution of this field! Let's define a helper function which performs an autoregressive rollout.
#
# We want this helper to:
# - Move our data to the target device
# - Reshape the data into the format expected by Walrus
# - Rollout the model as many steps as we have reference data for, performing appropriate normalization as we go

# In[7]:


from walrus_workshop.rollout import rollout_model

# Now lets make our forecast!

# In[8]:


with torch.no_grad():
    trajectory_example["padded_field_mask"] = trajectory_example[
        "padded_field_mask"
    ].to(device)  # We're going to want this out here too
    inputs, y_ref = formatter.process_input(
        trajectory_example,
        causal_in_time=model.causal_in_time,
        predict_delta=True,
        train=False,
    )
    y_pred, y_ref = rollout_model(
        model,
        revin,
        trajectory_example,
        formatter,
        max_rollout_steps=100,
        device=device,
    )

    # Lets get some extra info so we can visualize our data effectively
    # Remove unused fields
    y_pred, y_ref = (
        y_pred[..., trajectory_example["padded_field_mask"]],
        y_ref[..., trajectory_example["padded_field_mask"]],
    )
    # Collecting names to make detailed output logs
    field_names = flatten_field_names(metadata, include_constants=False)
    used_field_names = [
        f
        for i, f in enumerate(field_names)
        if trajectory_example["padded_field_mask"][i]
    ]


# # Make a figure
# import numpy as np
# import matplotlib.pyplot as plt

# def normalize_data(data):
#     return (data - data.min()) / (data.max() - data.min())

# pred_data = y_pred[0, :, :, :, 0].cpu().numpy()
# ref_data = y_ref[0, :, :, :, 0].cpu().numpy()

# pred_data = np.squeeze(pred_data)
# ref_data = np.squeeze(ref_data)

# print(f"pred_data shape: {pred_data.shape}, dtype: {pred_data.dtype}, min: {pred_data.min()}, max: {pred_data.max()}")
# print(f"ref_data shape: {ref_data.shape}, dtype: {ref_data.dtype}, min: {ref_data.min()}, max: {ref_data.max()}")

# plt.figure()
# plt.subplot(2,2,1)
# plt.pcolormesh(normalize_data(pred_data[-1] - pred_data[0]))
# plt.colorbar()
# plt.title('Prediction Difference')
# plt.subplot(2,2,2)
# plt.pcolormesh(normalize_data(ref_data[-1] - ref_data[0]))
# plt.colorbar()
# plt.title('Reference Difference')
# plt.subplot(2,2,3)
# plt.pcolormesh(normalize_data(pred_data[-1]))
# plt.colorbar()
# plt.title('Prediction')
# plt.subplot(2,2,4)
# plt.pcolormesh(normalize_data(ref_data[-1]))
# plt.colorbar()
# plt.title('Reference')
# plt.savefig("figures/test0.png")
# plt.close()


# In[9]:


from the_well.benchmark.metrics import make_video

output_dir = "./figures/"

print("Making video")
make_video(
    y_pred[0],  # First sample only in batch
    y_ref[0],  # First sample only in batch
    metadata,
    output_dir=output_dir,
    epoch_number="_bubblml_example",  # Misleading parameter name, but duck typing lets it be used for naming the output. Needs upstream fix.
    field_name_overrides=used_field_names,  # Fields actually used
    size_multiplier=1.0,  #
)
