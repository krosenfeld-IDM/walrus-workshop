"""
Generate a multi-page PDF showing the top 10 enstrophy-correlated features.

Each page contains 2 figures (top 10 features = 5 pages). Each figure replicates
the layout from figure_enstrophy_feature.py:
  - Top panel: time-series of normalized enstrophy + normalized feature activation
  - Bottom panels: 2x4 grid (tracer row + enstrophy/feature overlay row) at 4 target timesteps
"""

import os
import pickle
import logging
import glob

import numpy as np
import torch
import zarr
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
from omegaconf import OmegaConf
from alive_progress import alive_it

from walrus_workshop.utils import get_key_value_from_string
from walrus_workshop.walrus import get_trajectory
from walrus_workshop.model import load_sae
from walrus_workshop import metrics

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.chdir(os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────────────────────
cfg = OmegaConf.load("configs/train.yaml")
trajectory_id = 50 # 56 # 56
ref_trajectory_id = 50
target_steps = [15, 40, 60, 80]
n_top_features = 10

# ── Load ranked feature list ──────────────────────────────────────────────────
# Source /home/krosenfeld/projects/walrus-workshop/experiments/well/explore_global_metric.ipynb
with open(Path("figures/preprint/data") / f"enstrophy_feature_list_traj_{ref_trajectory_id}.pkl", "rb") as f:
    feature_list = pickle.load(f)
top_feature_ids = feature_list["feature_ids"][:n_top_features]
logger.info(f"Top {n_top_features} features: {top_feature_ids}")

# ── Load trajectory, activations, and SAE model ──────────────────────────────
trajectory, trajectory_metadata = get_trajectory(cfg.walrus.dataset, trajectory_id)

activations_dir = (
    Path("activations")
    / "test"
    / "blocks.20.space_mixing.activation"
    / cfg.walrus.dataset
)
act_files = sorted(glob.glob(str(activations_dir / f"*_traj_{trajectory_id}_*")))
steps = np.array([int(get_key_value_from_string(f, "step")) for f in act_files])

checkpoint_path = (
    Path("checkpoints")
    / "sae_checkpoint_blocks.20.space_mixing.activation_source_test_k_active=32_k_aux=2048_latent=22528_beta=0.1.pt"
)
sae_model, sae_config = load_sae(checkpoint_path)
sae_model = sae_model.to(device).eval()

# ── Time-series computation (single pass through all activation files) ────────
logger.info("Computing time-series for all top features...")
num_steps = len(steps)
t_list = []
enstrophy_list = []
# Store per-feature spatial means: dict of feature_id -> list of arrays
feature_means = {fid: [] for fid in top_feature_ids}

for step_index in alive_it(range(num_steps), force_tty=True):
    step = steps[step_index]
    simulation_chunk = trajectory['input_fields'][0, step:step+cfg.walrus.n_steps_input+1, :, :, 0, :]

    # Compute global enstrophy per sub-step
    enstrophy_per_substep = []
    for i in range(simulation_chunk.shape[0]):
        enstrophy_per_substep.append(
            metrics.compute_enstrophy(simulation_chunk[i, ..., 2], simulation_chunk[i, ..., 3])[0]
        )

    # SAE inference
    act = zarr.open(act_files[step_index], mode="r")
    act = torch.from_numpy(np.array(act)).to(device)
    with torch.no_grad():
        _, code, _ = sae_model(act)
    code = code.cpu().numpy().reshape(6, 32, 32, -1)

    # Accumulate per-timestep data (skip duplicates)
    for i in range(len(enstrophy_per_substep) - 1):
        t_val = step + i
        if len(t_list) > 0 and t_val <= t_list[-1]:
            continue
        t_list.append(t_val)
        enstrophy_list.append(enstrophy_per_substep[i])
        spatial_mean = code[i].mean(axis=(0, 1))  # shape: (n_latent,)
        for fid in top_feature_ids:
            feature_means[fid].append(spatial_mean[fid])

# Sort by time
t_arr = np.array(t_list)
sort_idx = np.argsort(t_arr)
t_arr = t_arr[sort_idx]
enstrophy_arr = np.array(enstrophy_list)[sort_idx]
for fid in top_feature_ids:
    feature_means[fid] = np.array(feature_means[fid])[sort_idx]


# ── Spatial data at target timesteps (for all features) ──────────────────────
def get_spatial_data(trajectory, step_offset, step, step_index, act_files, cfg, sae_model, feature_id):
    """Compute spatial panels for a single feature at a single timestep."""
    simulation_chunk = trajectory['input_fields'][0, step:step+cfg.walrus.n_steps_input, :, :, 0, :]
    scale_x = int(simulation_chunk.shape[2] / 32)
    scale_y = int(simulation_chunk.shape[1] / 32)

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


# Pre-compute step indices for target steps
target_info = []
for target_step in target_steps:
    step_index = np.argwhere(steps <= target_step)[-1][0]
    step_offset = target_step - steps[step_index]
    step = steps[step_index]
    target_info.append((step_index, step_offset, step))

# Cache spatial data at target timesteps: {(step_index, step_offset): (tracer, enstrophy, code_full)}
# We cache the SAE code and spatial enstrophy per target step, then extract per-feature later
logger.info("Computing spatial data at target timesteps...")
spatial_cache = {}
for step_index, step_offset, step in target_info:
    simulation_chunk = trajectory['input_fields'][0, step:step+cfg.walrus.n_steps_input, :, :, 0, :]
    scale_x = int(simulation_chunk.shape[2] / 32)
    scale_y = int(simulation_chunk.shape[1] / 32)

    act = zarr.open(act_files[step_index], mode="r")
    act = torch.from_numpy(np.array(act)).to(device)
    with torch.no_grad():
        _, code, _ = sae_model(act)
    code = code.cpu().numpy().reshape(6, 32, 32, -1)

    enstrophy_spatial = np.zeros((32, 32))
    for ix in range(32):
        for iy in range(32):
            token = simulation_chunk[step_offset, iy*scale_y:(iy+1)*scale_y, ix*scale_x:scale_x*(ix+1), :]
            enstrophy_spatial[iy, ix] = metrics.compute_enstrophy(token[:, :, 2], token[:, :, 3])[0]

    tracer = simulation_chunk[step_offset, ..., 0]
    spatial_cache[(step_index, step_offset)] = (tracer, enstrophy_spatial, code[step_offset])

logger.info("Spatial data computed.")

# ── Generate PDF ─────────────────────────────────────────────────────────────
savedir = Path("figures/preprint")
os.makedirs(savedir, exist_ok=True)
pdf_path = savedir / f"enstrophy_top_features_traj_{trajectory_id}_ref_{ref_trajectory_id}.pdf"

cmap = LinearSegmentedColormap.from_list('mask', [(1, 0, 0, 0), (1, 0, 0, 0.6)])
extent = (0.5, 512.5, 256.5, 0.5)

logger.info(f"Generating PDF: {pdf_path}")
with PdfPages(pdf_path) as pdf:
    for page_idx in range(5):
        fig = plt.figure(figsize=(8.5, 11), dpi=200)

        # Two features per page
        for slot in range(2):
            feat_idx = page_idx * 2 + slot
            feature_id = top_feature_ids[feat_idx]
            feat_arr = feature_means[feature_id]

            # Each slot occupies half the page height
            # slot 0: top half, slot 1: bottom half
            # Within each slot: 4 rows (timeseries, spacer, tracer row, overlay row)
            row_offset = slot * 4
            gs = fig.add_gridspec(
                nrows=8, ncols=4,
                height_ratios=[1.2, 0.38, 1, 1, 1.2, 0.38, 1, 1],
                hspace=0.03, wspace=0.1,
                top=0.95, bottom=0.3,
                left=0.02, right=0.98,
            )

            # Time-series panel
            ax_top = fig.add_subplot(gs[row_offset, :])
            ax_top.plot(t_arr, enstrophy_arr / np.max(enstrophy_arr), '-', color='xkcd:lavender', label=r'$\mathcal{E}$')
            ax_top.plot(t_arr, feat_arr / np.max(feat_arr), '-', color='xkcd:orange', label=f'Feature {feature_id}')
            ax_top.set_xlabel("Timestep")
            ax_top.legend(frameon=False, fontsize=8, loc='upper right')
            ax_top.set_xlim(t_arr[0], t_arr[-1])
            ax_top.set_ylim(0, None)
            ax_top.set_yticks([])
            ax_top.set_title(f"Feature {feature_id}", fontsize=10, fontweight='bold')

            # Spatial panels: 2 rows x 4 columns
            axs = [[fig.add_subplot(gs[row_offset + 2, c]) for c in range(4)],
                    [fig.add_subplot(gs[row_offset + 3, c]) for c in range(4)]]

            for i, (step_index, step_offset, step) in enumerate(target_info):
                tracer, enstrophy_spatial, code_full = spatial_cache[(step_index, step_offset)]
                feature_spatial = code_full[..., feature_id]

                axs[0][i].imshow(tracer, extent=extent)
                axs[1][i].imshow(enstrophy_spatial, extent=extent, cmap='Blues', vmin=0, vmax=np.max(enstrophy_spatial))
                axs[1][i].imshow(feature_spatial, extent=extent, cmap=cmap, vmin=0, vmax=feature_spatial.max())
                axs[1][i].set_xlabel(f"Step {step + step_offset}", fontsize=7)

                bar_top = ax_top.get_ylim()[1] * 0.1
                ax_top.vlines(step + step_offset, 0, bar_top, color='xkcd:gray', linestyle='-')
                ax_top.plot([step + step_offset], [bar_top], 'o', color='xkcd:gray', markersize=3)

                if i == 0:
                    axs[1][i].text(0.02, -0.07, "Enstrophy", transform=axs[1][i].transAxes,
                                   ha="left", va="top", color='xkcd:blue', fontsize=7)
                    axs[1][i].text(0.02, -0.23, f"Feature {feature_id}", transform=axs[1][i].transAxes,
                                   ha="left", va="top", color='xkcd:red', fontsize=7)
                    axs[0][i].text(0.02, 1.02, "Tracer", transform=axs[0][i].transAxes,
                                   ha="left", va="bottom", color='xkcd:black', fontsize=7)

            for ax in [a for row in axs for a in row]:
                ax.set_xticks([])
                ax.set_yticks([])

        pdf.savefig(fig)
        plt.close(fig)
        logger.info(f"  Page {page_idx + 1}/5 saved.")

logger.info(f"PDF saved to {pdf_path}")
