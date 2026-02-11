"""
Plot feature with max correlation with global enstrophy.
"""

import os
import pickle
import logging
import numpy as np
import matplotlib.pyplot as plt

from dataclasses import dataclass
from omegaconf import OmegaConf
from pathlib import Path
from walrus_workshop.metrics import compute_energy_spectrum
from walrus_workshop.walrus import get_trajectory
from script_walrus_errors import WalrusError

import zarr
import glob
import torch
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

os.chdir(os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DataChunk:
    t: np.ndarray
    enstrophy: np.ndarray
    dEdt: np.ndarray
    features: np.ndarray

def get_data_chunk(trajectory, step_offset, step, step_index, act_files, cfg, sae_model):
    simulation_chunk = trajectory['input_fields'][0, step:step+cfg.walrus.n_steps_input, :, :, 0, :]
    scale_x = int(simulation_chunk.shape[2] / 32)  # width
    scale_y = int(simulation_chunk.shape[1] / 32)  # height

    act = zarr.open(act_files[step_index], mode="r")
    act = torch.from_numpy(np.array(act)).to(device)
    with torch.no_grad():
        _, code, _ = sae_model(act)
    code = code.cpu().numpy().reshape(6, 32, 32, -1)

    enstrophy = np.zeros((32, 32))
    for ix in range(32):
        for iy in range(32):
            token = simulation_chunk[step_offset, iy*scale_y:(iy+1)*scale_y, ix*scale_x:scale_x*(ix+1), :]
            enstrophy[iy, ix] = metrics.compute_enstrophy(token[:, :, 2], token[:, :, 3])[0]

    return simulation_chunk[step_offset, ..., 0], enstrophy, code[step_offset, ..., feature_id]

# Load the config
cfg = OmegaConf.load("configs/train.yaml")

feature_id = 8245 #11253 # from explore_global_metric.ipynb
# Load the trajectory
trajectory_id = 56 # 50
# load trajectory
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
steps = np.array([int(get_key_value_from_string(file_name, "step")) for file_name in act_files])

# Load the trained SAE
checkpoint_path = (
    Path("checkpoints")
    / "sae_checkpoint_blocks.20.space_mixing.activation_source_test_k_active=32_k_aux=2048_latent=22528_beta=0.1.pt"
)
sae_model, sae_config = load_sae(checkpoint_path)
sae_model = sae_model.to(device).eval()

# Create figure
fig = plt.figure(figsize=(10, 5), constrained_layout=False, dpi=400)

# 3 rows total:
# - row 0: big axis
# - rows 1–2: 2x4 grid
gs = fig.add_gridspec(
    nrows=4, ncols=4,
    height_ratios=[1.2, 0.38, 1, 1],   # make the top row a bit taller
    hspace=0.03, wspace=0.1     # spacing like your sketch
)

# Top: spans all 4 columns
ax_top = fig.add_subplot(gs[0, :])
with open(Path("figures/preprint/data") / f"enstrophy_feature_{feature_id}_traj_{trajectory_id}.pkl", "rb") as f:
    data = pickle.load(f)
ax_top.plot(data['t'], data['enstrophy'] / np.max(data['enstrophy']), '-', color='xkcd:lavender', label=r'$\mathcal{E}$')
ax_top.plot(data['t'], data['features'] / np.max(data['features']), '-', color='xkcd:orange', label=f'Feature {feature_id}')
ax_top.set_xlabel("Timestep")
ax_top.legend(frameon=False, fontsize=10, loc='upper right')
ax_top.set_xlim(data['t'][0], data['t'][-1])
ax_top.set_ylim(0, None)

# Bottom: 2 rows x 4 columns
cmap = LinearSegmentedColormap.from_list('mask', [(1, 0, 0, 0), (1, 0, 0, 0.6)])
axs = [[fig.add_subplot(gs[r, c]) for c in range(4)] for r in (2, 3)]
# Flatten if you prefer: axs = [ax for row in axs for ax in row]

target_steps = [15, 40, 60, 80]
extent = (0.5, 512.5, 256.5, 0.5)
for i, target_step in enumerate(target_steps):
    step_index = np.argwhere(steps <= target_step)[-1][0]
    step_offset = target_step - steps[step_index]
    step = steps[step_index]
    simulation_chunk, enstrophy, feature = get_data_chunk(trajectory, step_offset, step, step_index, act_files, cfg, sae_model)

    axs[0][i].imshow(simulation_chunk, extent=extent)
    axs[1][i].imshow(enstrophy, extent=extent, cmap='Blues', vmin=0, vmax=np.max(enstrophy))
    axs[1][i].imshow(feature, extent=extent, cmap=cmap, vmin=0, vmax=feature.max())
    axs[1][i].set_xlabel(f"Step {step+step_offset}")

    bar_top = ax_top.get_ylim()[1]*0.1
    ax_top.vlines(step+step_offset, 0, bar_top, color='xkcd:gray', linestyle='-')
    ax_top.plot([step+step_offset], [bar_top], 'o', color='xkcd:gray')

    if i == 0:
        axs[1][i].text(0.02, -0.07, "Enstrophy",transform=axs[1][i].transAxes,
            ha="left",
            va="top",color='xkcd:blue')
        axs[1][i].text(0.02, -0.23, f"Feature {feature_id}",transform=axs[1][i].transAxes,
            ha="left",
            va="top",color='xkcd:red')
        axs[0][i].text(0.02, 1.02, "Tracer",transform=axs[0][i].transAxes,
            ha="left",
            va="bottom",color='xkcd:black')


# (Optional) quick styling to see boxes clearly
# for ax in [ax_top] + [ax for row in axs for ax in row]:
for ax in [ax for row in axs for ax in row]:
    ax.set_xticks([])
    ax.set_yticks([])
ax_top.set_yticks([])

fig.subplots_adjust(bottom=0.08, top=0.92, left=0.02, right=0.98)

savedir = Path("figures/preprint")
os.makedirs(savedir, exist_ok=True)
plt.savefig(savedir / f"enstrophy_feature_{feature_id}_traj_{trajectory_id}.png")
plt.close()
