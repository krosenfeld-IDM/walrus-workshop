import torch

from omegaconf import OmegaConf
from hydra.utils import instantiate
from typing import Literal
from itertools import islice
import numpy as np
from the_well.data import WellDataset

from walrus_workshop import paths

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


def default_trajectory_config():
    return {
        '_target_': 'walrus.data.MixedWellDataModule',
        'batch_size': 1,
        'n_steps_input': 199,
        'n_steps_output': 1,
        'min_dt_stride': 1,
        'max_dt_stride': 1,
        'max_samples': 2000,
        'well_dataset_info': {
            'shear_flow': {
                'include_filters': [],
                'exclude_filters': [],
            }
        }
    }

TRAJECTORY_CONFIG = default_trajectory_config()

FIELD_INDEX_MAP_OVERRIDE = OmegaConf.load(paths.configs / "field_index_map_override.yaml")

def get_trajectory(
    dataset_id: str, trajectory_id=0, config = None, split: Literal["val", "test", "train"] = "val"
):
    """
    Get a trajectory from a dataset.
    Args:
        config_file: Path to the config file
        dataset_id: ID of the dataset (e.g. "shear_flow")
        trajectory_id: ID of the trajectory (starts at 0)
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
    # config = OmegaConf.load(config_file)
    if config is None:
        config = OmegaConf.create(TRAJECTORY_CONFIG)

    # The dataset objects precompute a number of dataset stats on init, so this may take a little while
    data_module = instantiate(
        config,
        well_base_path=paths.data / "datasets",
        world_size=1,
        rank=0,
        data_workers=1,
        field_index_map_override=FIELD_INDEX_MAP_OVERRIDE, # Use the previous field maps to avoid cycling through the data
        prefetch_field_names=False,
    )

    # Grab the trajectory
    dataset_names = list(config.well_dataset_info.keys())
    if len(dataset_names) > 1 and split == 'train':
        raise ValueError("Cannot use train split with multiple datasets")

    if isinstance(dataset_id, str):
        dataset_index = dataset_names.index(dataset_id)
    else:
        raise ValueError(f"Invalid dataset ID: {dataset_id}")

    if split.lower() == "val":
        dataset = data_module.val_datasets[dataset_index]
    elif split.lower() == "test":
        dataset = data_module.test_datasets[dataset_index]
    elif split.lower() == "train":
        dataset = data_module.train_dataset
    else:
        raise ValueError(f"Invalid split: {split}")   

    # get the first window metadata
    metadata = dataset[0]["metadata"]        

    # check if trajectory_id is out of range
    assert trajectory_id < sum(metadata.n_trajectories_per_file), f"Trajectory ID {trajectory_id} is out of range (ID >= {sum(metadata.n_trajectories_per_file)})"

    # get the window size
    window_size = dataset.sub_dsets[0].n_steps_input + dataset.sub_dsets[0].n_steps_output

    # get the number of windows per trajectory per file
    num_windows_per_trajectory_per_file = [n_steps - window_size + 1 for n_steps in metadata.n_steps_per_trajectory]

    # get the file index of the trajectory
    traj_cum_counts = np.concatenate([[0], np.cumsum(metadata.n_trajectories_per_file)])
    traj_file_index = max(np.searchsorted(traj_cum_counts, trajectory_id) - 1, 0)

    # get the start index of the trajectory in the list of windows
    num_prev_traj_in_file = trajectory_id - traj_cum_counts[traj_file_index]
    traj_start_index = np.cumsum(np.concatenate([[0], num_windows_per_trajectory_per_file]))[traj_file_index]
    traj_start_index += num_windows_per_trajectory_per_file[traj_file_index] * num_prev_traj_in_file

    # Loop through windows and concatenate for full trajectory
    trajectory = None
    for i in range(traj_start_index, traj_start_index + num_windows_per_trajectory_per_file[traj_file_index], window_size):
        window = dataset[i]
        if trajectory is None:
            trajectory = window
            trajectory['input_fields'] = torch.cat([trajectory['input_fields'], window['output_fields']], dim=1)
        else:
            for k in ['input_fields', 'output_fields']:
                trajectory['input_fields'] = torch.cat([trajectory['input_fields'], window[k]], dim=1)
            for k in ['constant_scalars']:
                assert np.allclose(trajectory[k], window[k])

    return trajectory, trajectory['metadata']


def get_trajectory_slow(
    dataset_id, trajectory_index=0, config = None, split: Literal["val", "test", "train"] = "val"
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
    # config = OmegaConf.load(config_file)
    if config is None:
        config = OmegaConf.create(TRAJECTORY_CONFIG)

    config.n_steps_input = 6
    config.n_steps_output = 1

    # The dataset objects precompute a number of dataset stats on init, so this may take a little while
    data_module = instantiate(
        config,
        well_base_path=paths.data / "datasets",
        world_size=1,
        rank=0,
        data_workers=1,
        field_index_map_override=FIELD_INDEX_MAP_OVERRIDE, # Use the previous field maps to avoid cycling through the data
        prefetch_field_names=False,
    )

    # Grab the trajectory
    dataset_names = list(config.well_dataset_info.keys())
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
