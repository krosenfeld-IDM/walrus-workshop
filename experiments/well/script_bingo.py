"""
Look at the top K features per token for a single simulation, keeping track of dEdT (derivative of enstrophy w.r.t. time)
"""

import os
import logging
import re
import torch
import glob
import zarr
import heapq
import pickle
import numpy as np
import polars as pl

from pathlib import Path
from dataclasses import dataclass
from omegaconf import OmegaConf
from dataclasses import dataclass
from alive_progress import alive_it

from script_enstrophy import load_enstrophy_df
from walrus_workshop.model import load_sae
from walrus_workshop.walrus import get_trajectory

# Setup logger with handler to output to terminal
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

this_dir = Path(__file__).parent

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class Feature:
    activation: float
    index: int
    dEdT: float

    def __lt__(self, other):
        # For min-heap: lower activation = lower priority
        return self.activation < other.activation


def search_filename(file_name, key) -> int:
    traj_match = re.search(rf"{key}_([+-]?\d+(?:e[+-]?\d+)?)", file_name, re.IGNORECASE)
    if traj_match:
        traj_number = int(traj_match.group(1))
        return traj_number
    else:
        raise ValueError(f"No number found in file name for {key}.")


def collect_activations(cfg, trajectory_id: int, top_k: int = 20, max_loops: int | None = None):
    # Load file list of the activations
    activations_dir = (
        this_dir
        / "activations"
        / "test"
        / "blocks.20.space_mixing.activation"
        / cfg.walrus.dataset
    )
    act_files = sorted(glob.glob(str(activations_dir / f"*_traj_{trajectory_id}*")))
    if max_loops is None:
        max_loops = len(act_files)


    # Load the trained SAE
    checkpoint_path = (
        this_dir
        / "checkpoints"
        / "sae_checkpoint_blocks.20.space_mixing.activation_source_test_k_active=32_k_aux=2048_latent=22528_beta=0.1.pt"
    )
    sae_model, sae_config = load_sae(checkpoint_path)
    sae_model = sae_model.to(device).eval()

    # # Load the trajectory
    # trajectory, trajectory_metadata = get_trajectory(
    #     cfg.walrus.dataset, trajectory_id=trajectory_id
    # )

    heaps: dict[int, list[Feature]] = {
        idx: []
        for idx in range(
            32
            * 32
            * (
                max(
                    [
                        search_filename(Path(act_file).stem, "step")
                        for act_file in act_files
                    ]
                )
                + 1
            )
        )
    }

    for loop_cnt, act_file in alive_it(enumerate(act_files)):
        # Get the step number
        file_name = Path(act_file).stem
        step_number = search_filename(file_name, "step")

        # Get the feature values
        act = zarr.open(act_file, mode="r")
        act = torch.from_numpy(np.array(act)).to(device)
        with torch.no_grad():
            _, code, _ = sae_model(act)
        
        token_indices = step_number * 32 * 32 + np.arange(code.shape[0])
        for i, token_index in enumerate(token_indices):
            heap = heaps[token_index]
            features = code[i, :].detach().cpu().numpy()  # Features for token i
            dEdT = 0  # TODO: compute dEdT
            for feature_idx, activation in enumerate(features):
                if activation > 0:  # Only consider active features
                    if len(heap) < top_k:
                        heapq.heappush(heap, Feature(activation, feature_idx, dEdT))
                    elif activation > heap[0].activation:
                        heapq.heapreplace(heap, Feature(activation, feature_idx, dEdT))
        
        if max_loops is not None and loop_cnt >= max_loops:
            break

    activations = {}
    for token_index in heaps:
        activations[token_index] = {
            "activations": np.array([feature.activation for feature in heaps[token_index]]),
            "features": np.array([feature.index for feature in heaps[token_index]]),
        }
    return activations


if __name__ == "__main__":
    os.chdir(this_dir)

    cfg = OmegaConf.load("configs/train.yaml")

    # Figure out the trajectories with large mediant abs enstrophy
    df = load_enstrophy_df(data_id=cfg.walrus.dataset)
    group = (
        df.group_by("id", "filename")
        .agg(pl.col("abs_derivative").median().alias("median_abs_derivative"))
        .sort("median_abs_derivative", descending=True)
    )
    top_ids = group[:20]["id"].to_list()

    # #$
    activations = collect_activations(cfg, top_ids[0], max_loops = 2)
    with open(f"activations_traj_{top_ids[0]}.pkl", "wb") as f:
        pickle.dump(activations, f)
