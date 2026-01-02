#!/usr/bin/env python
# coding: utf-8

# # Running Walrus outside of train.py
#
# Hello! In this demo, we'll walk you how to use the Walrus foundation model outside of our provided training code. While the training code has many features
# and options, this can also make it a bit hard to catch the essentials. That is why this notebook exists. We'll walk you through how to use
# Walrus, primarily in inference here, both with Well formatted data and if you have data you can't force into the Well's format to use our included data utilities.
#
# ## Part 1: Using Walrus with Well-style datasets
#
# Prereqs:
# - Since this is using Well-style datasets, this section assumes you have Well-structured data, likely downloaded into a folder structure matching the one you'd get from downloading [The Well](https://github.com/PolymathicAI/the_well).
# - While this example only covers inference, this is using a 1.3B parameter model, so it is necessary to have either enough RAM if using CPU (slow) or VRAM if using GPU to handle a model of this size.
#
# The Walrus codebase is designed with Well-style datasets in mind. While it is not necessary to use these for Walrus,
# we're going to cover this option first since it's going to be a bit more straightforward.
#
# We're also going to take advantage of [Hydra](https://hydra.cc) to translate between OmegaConf configs and datasets and the pretrained Walrus model.
# Hydra is useful for hierarchical instantiation. It lets us break torch modules into individual component types and define models as a hierarchy
# of modular components that can be swapped out completely for other components.
#
# As a first step, we're just going to download the data. We'll just use wget to pull it from huggingface.

#

# In[1]:


import os

# Change to the directory containing this notebook/script
if "__file__" in globals():
    notebook_dir = os.path.dirname(os.path.abspath(__file__))
    print(notebook_dir)
    os.chdir(notebook_dir)
print(f"Current working directory: {os.getcwd()}")

# Setup directory structure
checkpoint_base_path = "./checkpoints/"
config_base_path = "./configs/"
os.makedirs(checkpoint_base_path, exist_ok=True)
os.makedirs(config_base_path, exist_ok=True)

# And we'll download the weights from huggingface
import subprocess

config_file = f"{config_base_path}/extended_config.yaml"
checkpoint_file = f"{checkpoint_base_path}/walrus.pt"

if not os.path.exists(config_file):
    subprocess.run(
        [
            "wget",
            "https://huggingface.co/polymathic-ai/walrus/resolve/main/extended_config.yaml",
            "-O",
            config_file,
        ],
        check=True,
    )

if not os.path.exists(checkpoint_file):
    subprocess.run(
        [
            "wget",
            "https://huggingface.co/polymathic-ai/walrus/resolve/main/walrus.pt",
            "-O",
            checkpoint_file,
        ],
        check=True,
    )


# In[2]:


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

# checkpoint_path = f"{checkpoint_base_path}/walrus.pt"
# checkpoint_config_path = f"{config_base_path}/extended_config.yaml"
checkpoint_path = checkpoint_file
checkpoint_config_path = config_file
# checkpoint_config_path = os.path.join(".", "configs", "bubbleml_poolboil_subcool.yaml")
checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)["app"][
    "model"
]
config = OmegaConf.load(checkpoint_config_path)

# Lets start by examining our config file

print(OmegaConf.to_yaml(config))


# There are a couple key configuration areas since we're using Walrus outside of the training code:
# - **config.data** - Defines the input data configuration used during training this checkpoint.
#     - **config.data.field_index_map_override** - Gives a mapping between physical fields and indices keys in the embedding layer.
# - **config.model** - Defines the structure of the Walrus model itself.
# - **config.trainer.revin** - Tells us what normalization approach to use.
# - **config.trainer.formatter** - Tells us what to use to format the data before feeding it into the model.
#
# We'll start off this process by initializing the data and model. The data size helps us determine the size of the input embedding layer. For ease of use, we'll
# load the full `DataModule` object which has a bit higher overhead compared to loading just the required dataset, but will make the overall process easier.
#
# Since we're using a pretrained checkpoint, we do not need the datamodule to prefetch field information from the data. Instead, we'll be using the
# saved information already in the data config.

# In[3]:


# well_base_path = "/mnt/home/polymathic/ceph/the_well/datasets/"
well_base_path = "/home/krosenfeld/projects/walrus-workshop/notebooks/datasets"
# First we're going to remove non-Well data since that uses absolute paths which are likely not on your system
# with open_dict(config):
# del config.data.module_parameters.well_dataset_info.flowbench_FPO_NS_2D_512x128_harmonics

# The dataset objects precompute a number of dataset stats on init, so this may take a little while
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


# From here, we want to use the `field_to_index_map` object to determine the dimension of the encoder. Since we're
# just using the Well data in this example, this shouldn't change from the pre-existing map, but we'll get it from the dataset
# as an example.

# In[4]:


field_to_index_map = data_module.train_dataset.field_to_index_map
# Retrieve the number of fields used in training
# from the mapping of field to index and incrementing by 1
# total_input_fields = max(field_to_index_map.values()) + 1
total_input_fields = 67  # We don't have all the Well so we grab from the errors
model: torch.nn.Module = instantiate(
    config.model,
    n_states=total_input_fields,
)
model.load_state_dict(checkpoint)

# Move to the device we want
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()


# Now that we have a pretrained model, let's look at how we would use it. First, we need a couple helper objects - the formatter and the normalization object. Since Walrus was trained with reversible normalization, it can be easier to implement outside of the data loader.
#
# The formatter object converts data from the Well convention format to the format ingested by Walrus.
#
# The normalization object normalizes the data before it enters the model and denormalizes the outputs.

# In[5]:


formatter = ChannelsFirstWithTimeFormatter()
revin = instantiate(config.trainer.revin)()  # This is a functools partial by default


# Before we use the model, let's look at the structure of the data. This will give us important information for how we can use the model when we don't have Well-formatted data.

# In[6]:


# Grab one trajectory to use as an example
# dataset_index = 3  # Corresponds to acoustic_scatter_inclusions
dataset_index = 1
dataset = data_module.rollout_val_datasets[dataset_index].sub_dsets[0]
metadata = dataset.metadata

trajectory_example = next(iter(data_module.rollout_val_dataloaders()[dataset_index]))

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


import copy

from walrus.trainer.training import expand_mask_to_match


def rollout_model(
    model,
    revin,
    batch,
    formatter,
    max_rollout_steps=200,
    model_epsilon=1e-5,
    device=torch.device("cpu"),
):
    """Rollout the model for as many steps as we have data for.

    Simplified version of the trainer method for demo purposes.
    """
    metadata = batch["metadata"]
    batch = {
        k: v.to(device) if k not in {"metadata", "boundary_conditions"} else v
        for k, v in batch.items()
    }
    # Extract mask and move to device for loss eval
    if (
        "mask"
        in batch["metadata"].constant_field_names[
            0
        ]  # Assuming all metadata in batch are the same
    ):
        mask_index = batch["metadata"].constant_field_names[0].index("mask")
        mask = batch["constant_fields"][..., mask_index : mask_index + 1]
        mask = mask.to(device, dtype=torch.bool)
    else:
        mask = None

    inputs, y_ref = formatter.process_input(
        batch,
        causal_in_time=model.causal_in_time,
        predict_delta=True,
        train=False,
    )

    # Inputs T B C H [W D], y_ref B T H [W D] C
    T_in = batch["input_fields"].shape[1]
    max_rollout_steps = max_rollout_steps + (T_in - 1)
    rollout_steps = min(
        y_ref.shape[1], max_rollout_steps
    )  # Number of timesteps in target
    train_rollout_limit = 1

    y_ref = y_ref[
        :, :rollout_steps
    ]  # If we set a maximum number of rollout steps, just cut it off now to save memory
    # Create a moving batch of one step at a time
    # moving_batch = copy.deepcopy(batch)
    moving_batch = batch
    y_preds = []
    # Rollout the model - Causal in time gets more predictions from the first step
    for i in range(train_rollout_limit - 1, rollout_steps):
        # Don't fill causal_in_time here since that only affects y_ref
        inputs, _ = formatter.process_input(moving_batch)
        inputs = list(inputs)
        with torch.no_grad():
            normalization_stats = revin.compute_stats(
                inputs[0], metadata#, epsilon=model_epsilon
            )
        # NOTE - Currently assuming only [0] (fields) needs normalization
        normalized_inputs = inputs[:]  # Shallow copy
        normalized_inputs[0] = revin.normalize_stdmean(
            normalized_inputs[0], normalization_stats
        )
        y_pred = model(
            normalized_inputs[0],
            normalized_inputs[1],
            normalized_inputs[2].tolist(),
            metadata=metadata,
        )
        # During validation, don't maintain full inner predictions
        if model.causal_in_time:
            y_pred = y_pred[-1:]  # y_pred is T first, y_ref is not
        # In validation, we want to reconstruct predictions on original scale
        y_pred = inputs[0][-y_pred.shape[0] :].float() + revin.denormalize_delta(
            y_pred, normalization_stats
        )  # Unnormalize delta and add to input
        y_pred = formatter.process_output(y_pred, metadata)[
            ..., : y_ref.shape[-1]
        ]  # Cut off constant channels

        # If we have masked fields, just move them back to zeros
        if mask is not None:
            mask_pred = expand_mask_to_match(mask, y_pred)
            y_pred.masked_fill_(mask_pred, 0)

        y_pred = y_pred.masked_fill(~batch["padded_field_mask"], 0.0)

        # If not last step, update moving batch for autoregressive prediction
        if i != rollout_steps - 1:
            moving_batch["input_fields"] = torch.cat(
                [moving_batch["input_fields"][:, 1:], y_pred[:, -1:]], dim=1
            )
        # For causal models, we get use full predictions for the first batch and
        # incremental predictions for subsequent batches - concat 1:T to y_ref for loss eval
        if model.causal_in_time and i == train_rollout_limit - 1:
            y_preds.append(y_pred)
        else:
            y_preds.append(y_pred[:, -1:])
    y_pred_out = torch.cat(y_preds, dim=1)
    if mask is not None:
        mask_ref = expand_mask_to_match(mask, y_ref)
        y_ref.masked_fill_(mask_ref, 0)
    return y_pred_out, y_ref


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
        max_rollout_steps=200,
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


# In[9]:


from the_well.benchmark.metrics import make_video

output_dir = "./figures/"

make_video(
    y_pred[0],  # First sample only in batch
    y_ref[0],  # First sample only in batch
    metadata,
    output_dir=output_dir,
    epoch_number="ac_inclusion_example",  # Misleading parameter name, but duck typing lets it be used for naming the output. Needs upstream fix.
    field_name_overrides=used_field_names,  # Fields actually used
    size_multiplier=1.0,  #
)


# # In[10]:


# from IPython.display import Video

# Video(
#     f"{output_dir}/{metadata.dataset_name}/rollout_video/epochac_inclusion_example_{metadata.dataset_name}.mp4",
#     width=640,
#     height=360,
# )


# # Awesome! We've made predictions with our pretrained model. Now let's look into what would happen if we needed to use the model for a downstream task and didn't want to go through the effort of first moving the data into the Well format.
# #
# # ## Part 2: Non-Well data
# #
# # Now lets see what we'd do in the case where we don't have Well structured data. The key difference here is that we'd need to define our own data transformation objects to make sure that every object in the pipeline is getting the data in the format they need it.
# #
# # We also need to make sure that our field_to_index_map is lined up with the new data which may include fields we haven't seen before.
# #
# # First let's get some extra imports from the library out of the way.

# # In[11]:


# from the_well.data.datasets import WellMetadata
# from walrus.utils.experiment_utils import align_checkpoint_with_field_to_index_map

# # Let's make a hypothetical dataset. This dataset has 4 fields - velocity_x, velocity_y, density, and a new, never-before-seen field "blubber". First, we'll check out the existing `field_to_index_map` to see what we can use:

# # In[12]:


# field_to_index_map


# # Three of our fields are covered. Additionally, since Walrus is expecting dimensionally padded data, we'll also need to include a velocity_x which we can concatenate to the end since these are treated as sets and the order doesn't matter.
# #
# # So we'll be passing fields {"velocity_x": 4, "velocity_y":5, "velocity_z": 6, "density": 28, "blubber": ?????} in the order [4, 5, 28, ????. 6].
# #
# # The question we need to answer is: how do we deal with blubber? This is fortunately easy enough, we just need to add an extra field to this mapping dictionary and pass it appropriately.

# # In[13]:


# new_field_to_index_map = copy.deepcopy(field_to_index_map)
# new_field_to_index_map["blubber"] = (
#     max(field_to_index_map.values()) + 1
# )  # New index for "blubber"

# model = instantiate(
#     config.model,
#     n_states=max(new_field_to_index_map.values()) + 1,
# )


# # Use the Walrus utility to align the checkpoint
# revised_model_checkpoint = align_checkpoint_with_field_to_index_map(
#     checkpoint_state_dict=checkpoint,
#     model_state_dict=model.state_dict(),
#     checkpoint_field_to_index_map=field_to_index_map,
#     model_field_to_index_map=new_field_to_index_map,
# )

# # Now load the aligned weights
# model.load_state_dict(revised_model_checkpoint)

# model.to(device)
# model.eval()


# # Now we just need to pass data with the right signature. From before, we know the model is using the following fields:
# # - `input_fields` - float tensor [B x T_in x H x W x D x C_var]
# # - `output_fields` - float tensor [B x T_out x H x W x D x C_var]
# # - `constant_fields` - Optional, float tensor[B x H x W x D x C_con]
# # - `boundary_conditions` - int tensor [Bx3x2]
# # - `padded_field_mask` - bool tensor [C_var]
# # - `field_indices` - Int tensor [C_var + C_con]
# # - `metadata` - WellMetadata - Not strictly necessary, but our functions above use this to help with logging, so we'll make one here too

# # In[14]:


# B = 1
# T_in = 6
# T_out = 10
# H = 128
# W = 128
# D = 1
# C_var = 5  # velocity_x, velocity_y, velocity_z, density, blubber
# C_con = 0  # No constant fields in this example


# synthetic_trajectory_example = {
#     "input_fields": torch.randn(B, T_in, H, W, D, C_var, device=device),
#     "output_fields": torch.randn(B, T_out, H, W, D, C_var, device=device),
#     "constant_fields": torch.randn(B, H, W, D, C_con, device=device),
#     "boundary_conditions": torch.tensor(
#         [[[2, 2], [2, 2], [2, 2]] for _ in range(B)], device=device
#     ),  # Example BCs
#     "padded_field_mask": torch.tensor(
#         [True, True, True, True, False], device=device
#     ),  # Last field index is padded
#     "field_indices": torch.tensor(
#         [4, 5, 28, 67, 6], device=device
#     ),  # Indices for all fields
#     "metadata": WellMetadata(
#         dataset_name="synthetic_dataset",
#         n_spatial_dims=3,
#         field_names={
#             0: ["pressure", "blubber"],
#             1: ["velocity_x", "velocity_y", "velocity_z"],
#             2: [],
#         },
#         spatial_resolution=(128, 128, 1),
#         scalar_names=[],
#         constant_field_names={0: [], 1: [], 2: []},
#         constant_scalar_names=[],
#         boundary_condition_types=[],  # Doesn't matter
#         n_files=[],  # Doesn't matter
#         n_trajectories_per_file=[],  # Doesn't matter
#         n_steps_per_trajectory=[],  # Doesn't matter
#     ),
# }


# # In[15]:


# with torch.no_grad():
#     synthetic_trajectory_example["padded_field_mask"] = synthetic_trajectory_example[
#         "padded_field_mask"
#     ].to(device)  # We're going to want this out here too
#     inputs, y_ref = formatter.process_input(
#         synthetic_trajectory_example,
#         causal_in_time=model.causal_in_time,
#         predict_delta=True,
#         train=False,
#     )
#     fake_metadata = synthetic_trajectory_example["metadata"]
#     y_pred, y_ref = rollout_model(
#         model,
#         revin,
#         synthetic_trajectory_example,
#         formatter,
#         max_rollout_steps=200,
#         device=device,
#     )

#     # Lets get some extra info so we can visualize our data effectively
#     # Remove unused fields
#     y_pred, y_ref = (
#         y_pred[..., synthetic_trajectory_example["padded_field_mask"]],
#         y_ref[..., synthetic_trajectory_example["padded_field_mask"]],
#     )
#     # Collecting names to make detailed output logs
#     field_names = flatten_field_names(fake_metadata, include_constants=False)
#     used_field_names = [
#         f
#         for i, f in enumerate(field_names)
#         if synthetic_trajectory_example["padded_field_mask"][i]
#     ]


# # Congratulations! Now you've used Walrus with both data in the Well format and independent data. Generally when using our training code, it's going to be much easier to use Well formatted data as it handles most of what we've just done automatically for Well formatted data. If you need some guidance on how to do that, we have an example in a second notebook.
