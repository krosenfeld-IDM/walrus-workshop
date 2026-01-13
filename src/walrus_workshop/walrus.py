from omegaconf import OmegaConf
from hydra.utils import instantiate
import torch
from walrus_workshop import paths
from typing import Literal
from itertools import islice


def load_model(config_file, checkpoint, move_to_device=False):
    """
    Load a model from a checkpoint and config file.
    Args:
        config_file: Path to the config file
        checkpoint: Path to the checkpoint file
        move_to_device: Whether to move the model to the device. Still need to call model.eval() after loading the model.
    Returns:
        model: Loaded model
        config: Loaded config
    """

    # Load config
    config = OmegaConf.load(config_file)

    # Load model
    checkpoint = torch.load(checkpoint, map_location="cpu", weights_only=True)["app"][
        "model"
    ]

    # The dataset objects precompute a number of dataset stats on init, so this may take a little while
    data_module = instantiate(
        config.data.module_parameters,
        well_base_path=paths.data / "datasets",
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

    if move_to_device:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

    return model, config


def get_trajectory(
    config_file, dataset_id, trajectory_index=0, split: Literal["val", "test", "train"] = "val"
):
    """
    Get a trajectory from a dataset.
    Args:
        config_file: Path to the config file
        dataset_id: ID of the dataset
        trajectory_index: Index of the trajectory
    Returns:
        trajectory_example: Example trajectory
        metadata: Metadata for the trajectory

    To use:
    with torch.no_grad():
        trajectory_example = get_trajectory(config_file, dataset_id, trajectory_index)
        trajectory_example["padded_field_mask"] = trajectory_example[
            "padded_field_mask"
        ].to(device)  # We're going to want this out here too
        inputs, y_ref = formatter.process_input(
            trajectory_example,
            causal_in_time=model.causal_in_time,
            predict_delta=True,
            train=False,
        )
    """
    config = OmegaConf.load(config_file)

    # The dataset objects precompute a number of dataset stats on init, so this may take a little while
    data_module = instantiate(
        config.data.module_parameters,
        well_base_path=paths.data / "datasets",
        world_size=1,
        rank=0,
        data_workers=1,
        field_index_map_override=config.data.get(
            "field_index_map_override", {}
        ),  # Use the previous field maps to avoid cycling through the data
        prefetch_field_names=False,
    )

    # Grab the trajectory
    dataset_names = list(config.data.module_parameters.well_dataset_info.keys())
    if isinstance(dataset_id, str):
        dataset_index = dataset_names.index(dataset_id)
    else:
        dataset_index = dataset_id
    print(f"Using dataset {dataset_names[dataset_index]}")
    if split.lower() == "val":
        dataset = data_module.rollout_val_datasets[dataset_index].sub_dsets[0]
        trajectory = next(
            islice(
                data_module.rollout_val_dataloaders()[dataset_index],
                trajectory_index,
                trajectory_index + 1,
            )
        )
    elif split.lower() == "test":
        dataset = data_module.rollout_test_datasets[dataset_index].sub_dsets[0]
        trajectory = next(
            islice(
                data_module.rollout_test_dataloaders()[dataset_index],
                trajectory_index,
                trajectory_index + 1,
            )
        )
    elif split.lower() == "train":
        dataset = data_module.rollout_train_datasets[dataset_index].sub_dsets[0]
        trajectory = next(
            islice(
                data_module.rollout_train_dataloaders()[dataset_index],
                trajectory_index,
                trajectory_index + 1,
            )
        )
    else:
        raise ValueError(f"Invalid split: {split}")

    return trajectory, dataset.metadata
