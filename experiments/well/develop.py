from torch._tensor import Tensor


from typing import Any


from walrus_workshop.model import load_model
from walrus_workshop import paths
from walrus_workshop.trajectory import get_trajectory
import torch
from walrus.data.well_to_multi_transformer import ChannelsFirstWithTimeFormatter
from hydra.utils import instantiate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading model...")
model, config = load_model(config_file=paths.configs/"well_config.yaml", checkpoint=paths.checkpoints/"walrus.pt", move_to_device=True)
model.eval()

# Dictionary to store activations
activations = {}

# Define the hook function
def get_activation(name):
    def hook(model, input, output):
        # 'output' is usually the activation tensor you want.
        # .detach() is crucial to stop gradients from flowing back into the main model
        activations[name] = output.detach()
    return hook

layers = dict(model.named_modules())
# print(layers.keys())

# Identify the layer you want to hook. 
# Print model structure to find the name: print(model)
layer_name = "blocks.20.space_mixing.activation"
target_layer = dict(model.named_modules())[layer_name]

# Register the hook
print(f"Registering hook for {layer_name}")
handle = target_layer.register_forward_hook(get_activation(layer_name))

# For normalization
formatter = ChannelsFirstWithTimeFormatter()
revin = instantiate(config.trainer.revin)()  # This is a functools partial by default

print("Getting trajectory")
batch, metadata = get_trajectory(config_file=paths.configs/"well_config.yaml", dataset_id="shear_flow", trajectory_index=0)
batch = {
    k: v.to(device)
    if k not in {"metadata", "boundary_conditions"}
    else v
    for k, v in batch.items()
}
# Extract mask and move to device for loss eval
if (
    "mask" in batch["metadata"].constant_field_names[0] # Assuming all metadata in batch are the same
):
    mask_index = batch["metadata"].constant_field_names[0].index("mask")
    mask = batch["constant_fields"][..., mask_index : mask_index + 1]
    mask = mask.to(device, dtype=torch.bool)
else:
    mask = None

with torch.no_grad():
    inputs, y_ref = formatter.process_input(
        batch,
        causal_in_time=model.causal_in_time,
        predict_delta=True,
        train=False,
    )
    # Don't fill causal_in_time here since that only affects y_ref
    inputs, _ = formatter.process_input(batch)
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
    print(f"Normalized inputs shape: {normalized_inputs[0].shape}")
    y_pred = model(
        normalized_inputs[0],
        normalized_inputs[1],
        normalized_inputs[2].tolist(),
        metadata=metadata,
    )
# Access the captured activations
act = activations[layer_name]
print(f"Captured activations shape: {act.shape}")