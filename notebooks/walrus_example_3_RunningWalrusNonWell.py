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

import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Setup directory structure
checkpoint_base_path = "./checkpoints/"
config_base_path = "./configs/"
os.makedirs(checkpoint_base_path, exist_ok=True)
os.makedirs(config_base_path, exist_ok=True)

checkpoint_path = f"{checkpoint_base_path}/walrus.pt"
checkpoint_config_path = f"{config_base_path}/extended_config.yaml"
checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)["app"][
    "model"
]
config = OmegaConf.load(checkpoint_config_path)

# Lets start by examining our config file
print(OmegaConf.to_yaml(config))

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
field_to_index_map = data_module.train_dataset.field_to_index_map
total_input_fields = max(field_to_index_map.values()) + 1

from walrus.utils.experiment_utils import align_checkpoint_with_field_to_index_map
from the_well.data.datasets import WellMetadata
import copy

new_field_to_index_map = copy.deepcopy(field_to_index_map)
new_field_to_index_map["blubber"] = max(field_to_index_map.values()) + 1  # New index for "blubber"

model = instantiate(
    config.model,
    n_states=max(new_field_to_index_map.values()) + 1,
)

# Use the Walrus utility to align the checkpoint
revised_model_checkpoint = align_checkpoint_with_field_to_index_map(
    checkpoint_state_dict=checkpoint,
    model_state_dict=model.state_dict(),
    checkpoint_field_to_index_map=field_to_index_map,
    model_field_to_index_map=new_field_to_index_map,
)

# Now load the aligned weights
model.load_state_dict(revised_model_checkpoint)

# Move to the device we want
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Construct formatter and normalization objects
formatter = ChannelsFirstWithTimeFormatter()
revin = instantiate(config.trainer.revin)()  # This is a functools partial by default

# Pass in the new data

B = 1
T_in = 6
T_out = 10
H = 128
W = 128
D = 1
C_var = 5  # velocity_x, velocity_y, velocity_z, density, blubber
C_con = 0  # No constant fields in this example


synthetic_trajectory_example = {
    "input_fields": torch.randn(B, T_in, H, W, D, C_var, device=device),
    "output_fields": torch.randn(B, T_out, H, W, D, C_var, device=device),
    "constant_fields": torch.randn(B, H, W, D, C_con, device=device),
    "boundary_conditions": torch.tensor([[[2, 2], [2, 2], [2, 2]] for _ in range(B)], device=device),  # Example BCs
    "padded_field_mask": torch.tensor([True, True, True, True, False], device=device),  # Last field index is padded
    "field_indices": torch.tensor([4, 5, 28, 67, 6], device=device),  # Indices for all fields
    "metadata": WellMetadata(
        dataset_name="synthetic_dataset",
        n_spatial_dims=3,
        field_names={0: ['pressure', "blubber"], 1: ['velocity_x', 'velocity_y', 'velocity_z'], 2: []},
        spatial_resolution=(128, 128, 1),
        scalar_names=[], 
        constant_field_names={0: [], 1: [], 2: []},
        constant_scalar_names=[],
        boundary_condition_types=[], # Doesn't matter
        n_files =[], # Doesn't matter
        n_trajectories_per_file=[], # Doesn't matter
        n_steps_per_trajectory=[], # Doesn't matter
    ),
}


from walrus.trainer.training import expand_mask_to_match

from alive_progress import alive_it

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
    moving_batch = copy.deepcopy(batch)
    # moving_batch = batch
    y_preds = []
    # Rollout the model - Causal in time gets more predictions from the first step
    for i in alive_it(range(train_rollout_limit - 1, rollout_steps)):
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


with torch.no_grad():
    synthetic_trajectory_example["padded_field_mask"] = synthetic_trajectory_example["padded_field_mask"].to(device) # We're going to want this out here too
    inputs, y_ref = formatter.process_input(
        synthetic_trajectory_example,
        causal_in_time=model.causal_in_time,
        predict_delta=True,
        train=False,
    )
    fake_metadata = synthetic_trajectory_example["metadata"]
    y_pred, y_ref = rollout_model(
        model,
        revin,
        synthetic_trajectory_example,
        formatter,
        max_rollout_steps=200,
        device=device,
    )

    # Lets get some extra info so we can visualize our data effectively
    # Remove unused fields
    y_pred, y_ref = (
        y_pred[..., synthetic_trajectory_example["padded_field_mask"]],
        y_ref[..., synthetic_trajectory_example["padded_field_mask"]],
    )
    # Collecting names to make detailed output logs
    field_names = flatten_field_names(fake_metadata, include_constants=False)
    used_field_names = [
        f
        for i, f in enumerate(field_names)
        if synthetic_trajectory_example["padded_field_mask"][i]
    ]



import matplotlib.pyplot as plt
plt.figure()
plt.subplot(1, 2, 1)
plt.pcolormesh(y_pred[0, 0, :, :, 0].cpu().numpy())
plt.colorbar()
plt.subplot(1, 2, 2)
plt.pcolormesh(y_ref[0, 0, :, :, 0].cpu().numpy())
plt.colorbar()
plt.savefig("test0.png")

plt.figure()
plt.subplot(1, 2, 1)
plt.pcolormesh(y_pred[0, -1, :, :, 0].cpu().numpy())
plt.colorbar()
plt.subplot(1, 2, 2)
plt.pcolormesh(y_ref[0, -1, :, :, 0].cpu().numpy())
plt.colorbar()
plt.savefig("test1.png")

print("done")