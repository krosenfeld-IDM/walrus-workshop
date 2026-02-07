"""
Calculate errors for saved activations
"""

import copy
import glob
import logging
import os
import pickle
import re
from dataclasses import dataclass
from pathlib import Path

import polars as pl
import numpy as np
import torch
from alive_progress import alive_it
from hydra.utils import instantiate
from omegaconf import OmegaConf

from walrus.data.well_to_multi_transformer import ChannelsFirstWithTimeFormatter
from walrus_workshop import paths
from walrus_workshop.walrus import get_trajectory

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

os.chdir(os.path.dirname(os.path.abspath(__file__)))

@dataclass
class WalrusError:
    pred: np.ndarray
    ref: np.ndarray
    step: int
    trajectory_id: int


def search_filename(file_name, key) -> int:
    traj_match = re.search(rf"{key}_([+-]?\d+(?:e[+-]?\d+)?)", file_name, re.IGNORECASE)
    if traj_match:
        traj_number = int(traj_match.group(1))
        return traj_number
    else:
        raise ValueError(f"No number zfound in file name for {key}.")


def predict(trajectory_id: int, cfg, formatter, revin, model):
    trajectory, metadata = get_trajectory(
        "shear_flow", trajectory_id=trajectory_id, split="test"
    )

    with torch.no_grad():
        trajectory["padded_field_mask"] = trajectory["padded_field_mask"].to(
            device
        )  # We're going to want this out here too
        inputs, y_ref = formatter.process_input(
            trajectory,
            causal_in_time=model.causal_in_time,
            predict_delta=True,
            train=False,
        )

        metadata = trajectory["metadata"]
        batch = {
            k: v.to(device)
            if k not in {"metadata", "boundary_conditions", "extra_metadata"}
            else v
            for k, v in trajectory.items()
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
            trajectory,  # Note: Might need to grab only a part
            causal_in_time=model.causal_in_time,
            predict_delta=True,
            train=False,
        )

        # Inputs T B C H [W D], y_ref B T H [W D] C
        T_in = cfg.data.module_parameters.n_steps_input

        moving_batch = copy.deepcopy(trajectory)

        act_files = glob.glob(
            os.path.join(
                Path("activations"),
                "test",
                "blocks.20.space_mixing.activation",
                "shear_flow",
                f"*traj_{trajectory_id}_*",
            )
        )
        steps = [search_filename(file, "step") for file in act_files]
        steps = sorted(steps)

        errors = []
        for step in alive_it(steps):
            moving_batch["input_fields"] = trajectory["input_fields"][
                0, step : step + T_in
            ][np.newaxis, ...].to(device)  # Note: Todo, this is where we select

            # Don't fill causal_in_time here since that only affects y_ref
            inputs, _ = formatter.process_input(
                moving_batch
            )  # "b t ... c -> t b c ..."
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

            errors.append(
                WalrusError(
                    pred=y_pred.cpu().numpy(),
                    ref=trajectory["input_fields"][0, step + T_in ].numpy(),
                    step=step,
                    trajectory_id=trajectory_id,
                )
            )

        outut_dir = "errors"
        os.makedirs(outut_dir, exist_ok=True)
        with open(os.path.join(outut_dir, f"errors_{trajectory_id}.pkl"), "wb") as f:
            pickle.dump(errors, f)


def load_model(cfg):
    checkpoint = torch.load(
        paths.checkpoints / "walrus.pt", map_location="cpu", weights_only=True
    )["app"]["model"]

    # Instantiate the model
    total_input_fields = max(cfg.data.field_index_map_override.values()) + 1
    logger.info(f"Instantiating model with {total_input_fields} input fields")
    model: torch.nn.Module = instantiate(
        cfg.model,
        n_states=total_input_fields,
    )
    # Load the checkpoint
    logger.info("Loading checkpoint")
    model.load_state_dict(checkpoint)

    # Move to the device we want
    model.to(device)
    model.eval()

    return model


def main():
    cfg = OmegaConf.load(paths.configs / "well_config.yaml")
    cfg.data.module_parameters.max_rollout_steps = 200

    formatter = ChannelsFirstWithTimeFormatter()
    revin = instantiate(cfg.trainer.revin)()  # This is a functools partial by default
    model = load_model(cfg)

    from script_enstrophy import load_enstrophy_df
    df = load_enstrophy_df(data_id="shear_flow")
    group = df.group_by('id', 'filename').agg(pl.col('abs_derivative').median().alias('median_abs_derivative')).sort('median_abs_derivative', descending=True)
    top_ids = group[:20]["id"].to_list()
    for trajectory_id in top_ids:
        print(f"Processing trajectory {trajectory_id}")
        predict(trajectory_id=trajectory_id, cfg=cfg, formatter=formatter, revin=revin, model=model)
        logger.info(f"Processed trajectory {trajectory_id}")

if __name__ == "__main__":
    main()
