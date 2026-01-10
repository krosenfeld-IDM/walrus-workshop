"""
Save activations from a specific layer of the model
"""

from torch._tensor import Tensor

from typing import Any
import os
import logging
from walrus_workshop import paths
from walrus_workshop.walrus import get_trajectory, load_model
import torch
from walrus.data.well_to_multi_transformer import ChannelsFirstWithTimeFormatter
from walrus_workshop.activation import ActivationManager
from hydra.utils import instantiate
from einops import rearrange
from alive_progress import alive_it

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Change directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Set up logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG) # DEBUG

# Define the hook function
def get_activation(name, activations):
    def hook(model, input, output):
        # 'output' is usually the activation tensor you want.
        # .detach() is crucial to stop gradients from flowing back into the main model
        activations[name] = output.detach()

    return hook


def strided_formatter(data, t_start=0, t_in=6):
    x = data["output_fields"][:, t_start : t_start + t_in, ...]
    x = rearrange(x, "b t ... c -> t b c ...")
    return (x, data["field_indices"], data["boundary_conditions"])


print("Loading model...")
model, config = load_model(
    config_file=paths.configs / "well_config.yaml",
    checkpoint=paths.checkpoints / "walrus.pt",
    move_to_device=True,
)
model.eval()

# If you want to see the layer names
# layers = dict(model.named_modules())
# print(layers.keys())

# Identify the layer you want to hook.
# Print model structure to find the name: print(model)
layer_name = "blocks.20.space_mixing.activation"
target_layer = dict(model.named_modules())[layer_name]

# Manage the activations
am = ActivationManager(
    enabled=True, save_dir=os.path.abspath(f"./activations/{layer_name}"), mode="both"
)
logger.log(f"Activation manager save directory: {am.save_dir}")
activations = {}

# Register the hook
print(f"Registering hook for {layer_name}")
handle = target_layer.register_forward_hook(get_activation(layer_name, activations))

# For normalization later
formatter = ChannelsFirstWithTimeFormatter()
revin = instantiate(config.trainer.revin)()  # This is a functools partial by default

num_trajectories = 28
for trajectory_index in alive_it(range(num_trajectories)):
    print(f"Getting trajectory {trajectory_index}")
    batch, metadata = get_trajectory(
        config_file=paths.configs / "well_config.yaml",
        dataset_id="shear_flow",
        trajectory_index=trajectory_index,
        split="test",
    )  # 'val' or 'test'
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

    for t_start in range(0, batch["output_fields"].shape[1] - 6, 6):
        logger.debug(f"Processing time step {t_start} / {batch['output_fields'].shape[1] - 6}")
        with torch.no_grad():
            # inputs, y_ref = formatter.process_input(
            #     batch,
            #     causal_in_time=model.causal_in_time,
            #     predict_delta=True,
            #     train=False,
            # ) # inputs  = fields, field_indices, boundary conditions
            # Don't fill causal_in_time here since that only affects y_ref
            # This is where we pull out the inputs (vs the whole trajectory)
            # inputs, _ = formatter.process_input(batch)
            inputs = strided_formatter(batch, t_start=t_start, t_in=6)
            inputs = list[Tensor | Any](inputs)
            with torch.no_grad():
                normalization_stats = revin.compute_stats(
                    inputs[0], metadata, epsilon=1e-5
                )
            # NOTE - Currently assuming only [0] (fields) needs normalization
            normalized_inputs = inputs[:]  # Shallow copy
            normalized_inputs[0] = revin.normalize_stdmean(
                normalized_inputs[0], normalization_stats
            )
            # Inputs T B C H [W D], y_ref B T H [W D] C
            logger.debug(f"Normalized inputs shape: {normalized_inputs[0].shape}")
            logger.debug(f"Normalized inputs[0] shape: {normalized_inputs[0].shape}")  # data
            logger.debug(f"Normalized inputs[1] shape: {normalized_inputs[1].shape}")  # field indices
            logger.debug(f"Normalized inputs[2] shape: {normalized_inputs[2].shape}")  # boundary conditions
            y_pred = model(
                normalized_inputs[0],
                normalized_inputs[1],
                normalized_inputs[2].tolist(),
                metadata=metadata,
            )
            logger.debug(f"y_pred shape: {y_pred.shape}")

        # Access the captured activations
        act = activations[layer_name]
        logger.debug(f"Captured activations shape: {act.shape}")

        # Current shape: [T, 32, 32, 1, 2816]
        # Target shape:  [Total_Tokens, Hidden_Dim]

        # 1. Squeeze the singleton dimension (the '1')
        # Shape becomes: [T, 32, 32, 2816]
        act = act.squeeze(3)

        # 2. Flatten the batch and spatial dimensions
        # Shape becomes: [T * 32 * 32, 2816] -> [10240, 2816]
        sae_input = act.reshape(-1, 2816)

        # Save the activations
        logger.debug(f"Saving activations for {layer_name}")
        am.save(
            f"layer{layer_name}_traj{trajectory_index}_tstart{t_start}",
            sae_input.cpu(),
            step_idx=t_start,
            node_set=list(activations.keys())[0],
        )

print("Done")
