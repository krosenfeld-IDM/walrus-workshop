import zarr
import glob
import torch
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from rich import print
from pathlib import Path
from omegaconf import OmegaConf
from dataclasses import dataclass, fields
from matplotlib.colors import LinearSegmentedColormap
from alive_progress import alive_it
from sortedcontainers import SortedList
from scipy import stats
from walrus_workshop.utils import get_key_value_from_string
from walrus_workshop.walrus import get_trajectory
from walrus_workshop.model import load_sae
from walrus_workshop import metrics

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class DataChunk:
    t: np.ndarray
    enstrophy: np.ndarray
    dEdt: np.ndarray
    features: np.ndarray


@dataclass
class Feature:
    index: int
    corr: float
    p_value: float

    def __lt__(self, other):
        return self.corr < other.corr


def get_data_chunk(trajectory, steps, act_files, cfg, sae_model):
    num_steps = len(steps)
    enstrophy = np.array([])
    dEdt = np.array([])
    t = np.array([])
    features = np.array([])
    for step_index in alive_it(range(num_steps), force_tty=True):
        step = steps[step_index]
        simulation_chunk = trajectory["input_fields"][
            0, step : step + cfg.walrus.n_steps_input + 1, :, :, 0, :
        ]
        enstrophy_ = []
        for i in range(simulation_chunk.shape[0]):
            enstrophy_.append(
                metrics.compute_enstrophy(
                    simulation_chunk[i, ..., 2], simulation_chunk[i, ..., 3]
                )[0]
            )

        act = zarr.open(act_files[step_index], mode="r")
        act = torch.from_numpy(np.array(act)).to(device)
        with torch.no_grad():
            _, code, _ = sae_model(act)
        code = code.cpu().numpy().reshape(6, 32, 32, -1)

        for i in range(len(enstrophy_) - 1):
            if step + i in t:
                continue
            dEdt = np.concatenate([dEdt, [np.diff(enstrophy_)[i]]])
            enstrophy = np.concatenate([enstrophy, [enstrophy_[i]]])
            features = np.concatenate([features, code.mean(axis=(1, 2))[i].ravel()])
            t = np.concatenate([t, [step + i]])

        # dEdt = np.concatenate([dEdt, list(np.diff(enstrophy_))])
        # enstrophy = np.concatenate([enstrophy, enstrophy_[:-1]])
        # features = np.concatenate([features, code.mean(axis=(1, 2)).ravel()])
        # t = np.concatenate([t, np.arange(step, step+cfg.walrus.n_steps_input)])

    return DataChunk(
        t=t,
        enstrophy=enstrophy,
        dEdt=dEdt,
        features=features.reshape(len(enstrophy), -1),
    )


os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Load the config
cfg = OmegaConf.load("configs/train.yaml")

# Load the trained SAE
checkpoint_path = (
    Path("checkpoints")
    / "sae_checkpoint_blocks.20.space_mixing.activation_source_test_k_active=32_k_aux=2048_latent=22528_beta=0.1.pt"
)
sae_model, sae_config = load_sae(checkpoint_path)
sae_model = sae_model.to(device).eval()

# Load the trajectory
trajectory_id = 56
trajectory, trajectory_metadata = get_trajectory(cfg.walrus.dataset, trajectory_id)

# Load file list of the activations
activations_dir = (
    Path("activations")
    / "test"
    / "blocks.20.space_mixing.activation"
    / cfg.walrus.dataset
)
act_files = sorted(glob.glob(str(activations_dir / f"*_traj_{trajectory_id}_*")))
# List of steps with activations (starting step)
steps = np.array(
    [int(get_key_value_from_string(file_name, "step")) for file_name in act_files]
)

data_chunk = get_data_chunk(trajectory, steps, act_files, cfg, sae_model)

feature_list = SortedList()
for feature_idx in alive_it(range(data_chunk.features.shape[1]), force_tty=True):
    s, p = stats.spearmanr(data_chunk.features[:, feature_idx], data_chunk.enstrophy)
    if np.isfinite(s):
        feature_list.add(Feature(feature_idx, s, p))

sort_me = np.argsort(data_chunk.t)
save_features = [
    feature_list[-1].index,
    8245
]
output_dir = Path("figures/preprint/data")
os.makedirs(output_dir, exist_ok=True)
for feature_index in save_features:
    with open(output_dir / f"enstrophy_feature_{feature_index}_traj_{trajectory_id}.pkl", "wb") as f:
        pickle.dump(
            {
                "t": data_chunk.t,
                "enstrophy": data_chunk.enstrophy[sort_me],
                "features": data_chunk.features[:, feature_index][
                    sort_me
                ],
            },
            f,
        )
