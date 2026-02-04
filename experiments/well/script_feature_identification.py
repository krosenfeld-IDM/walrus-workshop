import re
import zarr
import glob
import torch

import numpy as np
import matplotlib.pyplot as plt

from scipy import stats
from sortedcontainers import SortedKeyList
from alive_progress import alive_it
from omegaconf import OmegaConf
from pathlib import Path
from scipy.ndimage import zoom
from dataclasses import dataclass, fields
from matplotlib.colors import LinearSegmentedColormap

from walrus_workshop.model import load_sae
from walrus_workshop.walrus import get_trajectory
from walrus_workshop.metrics import compute_enstrophy, compute_enstrophy_flux

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def search_filename(file_name, key) -> int:
    traj_match = re.search(rf"{key}_([+-]?\d+(?:e[+-]?\d+)?)", file_name, re.IGNORECASE)
    if traj_match:
        traj_number = int(traj_match.group(1))
        return traj_number
    else:
        raise ValueError(f"No number found in file name for {key}.")


@dataclass
class EnstrophyData:
    step: int
    simulation: np.ndarray # T x 256 x 512 x 5 array
    enstrophy: np.ndarray # 32 x 32 array
    dEdt: np.ndarray # 32 x 32
    zoom_dEdt: np.ndarray # 256 x 512

# sort the features by activation weighted dEdt for this snapshot
@dataclass
class Feature:
    index: int
    dEdt: float
    num_active: int

    def __lt__(self, other): # For min-heap: dEdt = lower priority
        return self.dEdt < other.dEdt

@dataclass
class BingoFeature:
    index: int # index into code
    coverage: float
    std_coverage: float
    corr: float

def plot_enstrophy_evolution(trajectory, start_step, cfg, substep=0):

    assert substep < cfg.walrus.n_steps_input - 1

    # step = start_step + cfg.walrus.n_steps_input - 1
    step = start_step
    simulation_chunk = trajectory['input_fields'][0, step:step+cfg.walrus.n_steps_input+1, :, :, 0, :] # NOTE: Using the entire input chunk
    nx = simulation_chunk.shape[1]
    ny = simulation_chunk.shape[2]
    dx = nx // 32
    dy = ny // 32

    enstrophy = np.zeros((simulation_chunk.shape[0], 32, 32))
    for i in range(enstrophy.shape[0]):
        for ix in range(32):
            for iy in range(32):
                token = simulation_chunk[i, ix*dx:(ix+1)*dx, iy*dy:dy*(iy+1), :]
                enstrophy[i, ix, iy] = compute_enstrophy(token[:, :, 2], token[:, :, 3])[0]
    dEdt = -1*np.diff(enstrophy, axis=0)
    zoom_dEdt = zoom(dEdt, (1, dx, dy))
    vmax = np.max(np.abs(zoom_dEdt[substep]))

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    cm = axs[0].imshow(simulation_chunk[substep, :, :, 0], cmap="gray")
    fig.colorbar(cm, ax=axs[0], shrink=0.5)
    cm = axs[1].imshow(zoom_dEdt[substep], cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    fig.colorbar(cm, ax=axs[1], shrink=0.5)
    cm = axs[2].imshow(simulation_chunk[substep, :, :, 0], cmap="gray")
    axs[2].imshow(zoom_dEdt[substep], cmap=LinearSegmentedColormap.from_list('mask', [(1, 0, 0, 0), (1, 0, 0, 0.6)]), vmin=0, vmax=vmax)
    axs[2].imshow(zoom_dEdt[substep], cmap=LinearSegmentedColormap.from_list('mask', [(0, 0, 1, 0), (0, 0, 1, 0.6)]).reversed(), vmin=-vmax, vmax=0)
    fig.colorbar(cm, shrink=0.5)

    axs[0].set_title(f"Step {step + substep}: Tracer")
    axs[1].set_title(f"Step {step + substep}: dE/dt")
    axs[2].set_title(f"Step {step + substep}: Tracer + dE/dt")
    plt.show()

    return EnstrophyData(step=step, simulation=simulation_chunk[:-1], enstrophy=enstrophy, dEdt=dEdt, zoom_dEdt=zoom_dEdt)


def plot_feature(feature: Feature, enstrophy_data: EnstrophyData, code: np.ndarray):
    activations = code[:, feature.index].reshape(-1, 32, 32)
    fig, axes = plt.subplots(2, 3, figsize=(12, 5))
    for i in range(activations.shape[0]):
        ax = axes[i // 3, i % 3]
        ax.imshow(activations[i])
        ax.contour(np.arange(512) / 16 - 0.5, np.arange(256) / 8 - 0.5, enstrophy_data.simulation[i, :, :, 0], levels=1, colors="k")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    plt.show()

def main(trajectory_id:int, step_index:int):

    # Load the config
    cfg = OmegaConf.load("configs/train.yaml")

    # Load the trajectory
    trajectory_id = 50
    trajectory, trajectory_metadata = get_trajectory(cfg.walrus.dataset, trajectory_id)

    # Load file list of the activations
    activations_dir = (
        Path("activations")
        / "test"
        / "blocks.20.space_mixing.activation"
        / cfg.walrus.dataset
    )
    act_files = sorted(glob.glob(str(activations_dir / f"*_traj_{trajectory_id}*")))
    # List of steps with activations (starting step)
    steps = np.array([search_filename(file_name, "step") for file_name in act_files])
    step = steps[step_index]

    # Load the trained SAE
    checkpoint_path = (
        Path("checkpoints")
        / "sae_checkpoint_blocks.20.space_mixing.activation_source_test_k_active=32_k_aux=2048_latent=22528_beta=0.1.pt"
    )
    sae_model, sae_config = load_sae(checkpoint_path)
    sae_model = sae_model.to(device).eval()

    enstrophy_data = plot_enstrophy_evolution(trajectory, start_step=step, cfg=cfg);

    # Get the enstrophy data and SAE features
    print(f"Opening activation file {Path(act_files[step_index]).stem}")
    assert search_filename(Path(act_files[step_index]).stem, "step") == step # make sure we are processing the same step
    act = zarr.open(act_files[step_index], mode="r")
    act = torch.from_numpy(np.array(act)).to(device)
    with torch.no_grad():
        _, code, _ = sae_model(act)
    code = code.cpu().numpy()

    dEdt_thresh = 0.01
    activations_thresh = 0.0
    enstrophy_mask = enstrophy_data.dEdt < -1*dEdt_thresh # negative
    bingo_features = SortedKeyList(key=lambda x: x.coverage)
    for i in code.shape[1]:
        activations = code[:, i].reshape(-1, 32, 32)
        coverage = np.zeros(enstrophy_data.dEdt.shape[0])
        for i in range(enstrophy_data.dEdt.shape[0]):
            coverage[i] = np.sum(activations[i][enstrophy_mask[i]] > activations_thresh) / np.sum(enstrophy_mask[i])
        corr = stats.spearmanr(activations.ravel(), enstrophy_data.dEdt.ravel())[0]
        bingo_features.add(BingoFeature(index=i, coverage=np.mean(coverage), std_coverage=np.std(coverage), corr=corr))

    for cnt, feature in enumerate(bingo_features[::-1]):
        if cnt == 20:
            break
        print(f"Feature {feature.index} has a mean coverage of {feature.coverage:.2f} and a std coverage of {feature.std_coverage:.2f} and correlation of {feature.corr:.2f}")
        plot_feature(feature, enstrophy_data, code=code)
