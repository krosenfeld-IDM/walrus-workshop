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

# Change teh working directory
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

from walrus_workshop.rollout import rollout_model


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



# import matplotlib.pyplot as plt
# import numpy as np

# # Debug: Check shapes
# print(f"y_pred shape: {y_pred.shape}")
# print(f"y_ref shape: {y_ref.shape}")

# # Ensure we're extracting 2D arrays correctly
# pred_data = y_pred[0, 0, :, :, 0].cpu().numpy()
# ref_data = y_ref[0, 0, :, :, 0].cpu().numpy()

# # Ensure data is 2D and float
# pred_data = np.squeeze(pred_data)
# ref_data = np.squeeze(ref_data)

# print(f"pred_data shape: {pred_data.shape}, dtype: {pred_data.dtype}, min: {pred_data.min()}, max: {pred_data.max()}")
# print(f"ref_data shape: {ref_data.shape}, dtype: {ref_data.dtype}, min: {ref_data.min()}, max: {ref_data.max()}")

# def normalize_data(data):
#     return (data - data.min()) / (data.max() - data.min())

# plt.figure()
# plt.subplot(1, 2, 1)
# plt.pcolormesh(normalize_data(pred_data))
# plt.colorbar()
# plt.title('Prediction')
# plt.subplot(1, 2, 2)
# plt.pcolormesh(normalize_data(ref_data))
# plt.colorbar()
# plt.title('Reference')
# plt.savefig("figures/test0.png")
# plt.close()

# # Second figure
# pred_data_last = y_pred[0, -1, :, :, 0].cpu().numpy()
# ref_data_last = y_ref[0, -1, :, :, 0].cpu().numpy()

# pred_data_last = np.squeeze(pred_data_last)
# ref_data_last = np.squeeze(ref_data_last)

# plt.figure()
# plt.subplot(1, 2, 1)
# plt.pcolormesh(normalize_data(pred_data_last))
# plt.title('Prediction (last)')
# plt.subplot(1, 2, 2)
# plt.pcolormesh(normalize_data(ref_data_last))
# plt.colorbar()
# plt.title('Reference (last)')
# plt.savefig("figures/test1.png")
# plt.close()

from the_well.benchmark.metrics import make_video

output_dir = "./figures/"

print("Making video")
make_video(
    y_pred[0],  # First sample only in batch
    y_ref[0],  # First sample only in batch
    synthetic_trajectory_example["metadata"],
    output_dir=output_dir,
    epoch_number="_nonwell_example",  # Misleading parameter name, but duck typing lets it be used for naming the output. Needs upstream fix.
    field_name_overrides=used_field_names,  # Fields actually used
    size_multiplier=1.0,  #
)


print("done")