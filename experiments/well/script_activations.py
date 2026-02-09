"""
Save activations from a specific layer of the model for later analysis.

"""

import os
import logging

import torch
import numpy as np

from walrus.data.well_to_multi_transformer import ChannelsFirstWithTimeFormatter
from hydra.utils import instantiate
from einops import rearrange, repeat
from alive_progress import alive_it
from the_well.data import WellDataset
from pathlib import Path
from omegaconf import OmegaConf

from walrus_workshop.activation import ActivationManager
from walrus_workshop import paths
from walrus_workshop.walrus import get_trajectory, load_model


# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Change directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Set up logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # DEBUG

# Settings
data_id = "shear_flow"
split = "test"
layer_name = "blocks.20.space_mixing.activation"
checkpoint_file = paths.checkpoints / "walrus.pt"
activations_config_file = Path("./configs").resolve() / "activations.yaml"
activations_config = OmegaConf.load(activations_config_file)
walrus_config_file = paths.configs / "well_config.yaml"
walrus_config = OmegaConf.load(walrus_config_file)

# Load the dataset files so we can determine the number of trajectories
dataset = WellDataset(
    well_base_path=paths.well_base_path,
    well_dataset_name=data_id,
    well_split_name=split,
    n_steps_input=walrus_config.data.module_parameters.n_steps_input,
    n_steps_output=walrus_config.data.module_parameters.n_steps_output,
    use_normalization=False,
)
num_trajectories = sum(dataset.metadata.n_trajectories_per_file)


# Define the hook function
def get_activation(name, activations):
    def hook(model, input, output):
        # 'output' is usually the activation tensor you want.
        # .detach() is crucial to stop gradients from flowing back into the main model
        activations[name] = output.detach()

    return hook


def strided_formatter(data, t_start=0, t_in=6):
    x = data["input_fields"][:, t_start : t_start + t_in, ...]  # B T ...
    x = rearrange(x, "B T ... C -> T B C ...")
    if "constant_fields" in data:
        flat_constants = repeat(
            data["constant_fields"],
            "b ... c -> (repeat) b c ...",
            repeat=x.shape[0],
        )
        x = torch.cat([x, flat_constants], dim=2)
    return (x, data["field_indices"], data["boundary_conditions"])


logger.info("Loading model...")
model, config = load_model(
    config_file=walrus_config_file,
    checkpoint=checkpoint_file,
    move_to_device=True,
)
model.eval()

# If you want to see the layer names
# layers = dict(model.named_modules())
# print(layers.keys())

# Identify the layer you want to hook.
# Print model structure to find the name: print(model)
target_layer = dict(model.named_modules())[layer_name]

# Manage the activations
am = ActivationManager(
    enabled=True,
    save_dir=os.path.abspath(f"./activations/{split}/{layer_name}/{data_id}"),
    mode="both",
)
logger.info(f"Activation manager save directory: {am.save_dir}")
activations = {}

# Register the hook
logger.info(f"Registering hook for {layer_name}")
handle = target_layer.register_forward_hook(get_activation(layer_name, activations))

# For normalization later
formatter = ChannelsFirstWithTimeFormatter()
revin = instantiate(config.trainer.revin)()  # This is a functools partial by default

for trajectory_index in alive_it(range(num_trajectories)):
    logger.info(f"Getting trajectory {trajectory_index}")
    batch, metadata = get_trajectory(
        dataset_id=data_id,
        trajectory_id=trajectory_index,
        split=split,
    )
    batch = {
        k: v.to(device) if k not in {"metadata", "boundary_conditions", "extra_metadata"} else v
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

    rng = np.random.default_rng()
    for t_start_ in range(
        0, 101, 3 # batch["input_fields"].sahpe[1] - -3
    ):  # TODO: Read this from the config
        t_start = t_start_ + rng.integers(3)
        t_start = int(np.min([t_start, batch["input_fields"].shape[1] - 6]))
        logger.debug(
            f"Processing time step {t_start} / 101"
        )
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
            inputs = list(inputs)
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
            y_pred = model(
                normalized_inputs[0],
                normalized_inputs[1],
                normalized_inputs[2].tolist(),
                metadata=metadata,
            )

        # Access the captured activations
        act = activations[layer_name]

        # Current shape: [(T*B), H, W, D, C_mlp] = [T, 32, 32, 1, 2816] (assumes B=1)
        # Target shape:  [Total_Tokens, Hidden_Dim]

        # 1. Squeeze the singleton dimension (the '1')
        # Shape becomes: [T, 32, 32, 2816]
        act = act.squeeze(3)

        # 2. Flatten the batch and spatial dimensions
        # Shape becomes: [T * 32 * 32, 2816] -> [10240, 2816]
        sae_input = act.reshape(-1, act.shape[-1])

        # Save the activations
        file_root = f"traj_{trajectory_index}_" + "_".join(
            [
                f"{k}_{c.item():0.0e}"
                for k, c in zip(
                    batch["metadata"].constant_scalar_names,
                    batch["constant_scalars"][0],
                )
            ]
        )
        output_file_name = file_root + f"_layer{layer_name}"
        am.save(
            output_file_name,
            sae_input.cpu(),
            step_idx=t_start,
            node_set=list(activations.keys())[0],
        )

        # import json   
        # output_dir = "activations/debug"
        # os.makedirs(output_dir, exist_ok=True)
        # with open(os.path.join(output_dir, f"{file_root}.txt"), "w") as f:
        #     f.write(json.dumps(batch["extra_metadata"]))
        # print(f"Done with {file_root}")        
        # break

print("Done")
