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

os.chdir(os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

fig = plt.figure(figsize=(10, 3.5), constrained_layout=False, dpi=300)

gs = fig.add_gridspec(
    nrows=4,
    ncols=4,
    height_ratios=[1.0, 0.10, 1.0, 0.10],
    width_ratios=[1.0, 1.0, 0.30, 2.6],
    wspace=0.05,
    hspace=0.5,
)

# left column
ax11 = fig.add_subplot(gs[0, 0])
cax11 = fig.add_subplot(gs[1, 0])
ax21 = fig.add_subplot(gs[2, 0])
cax21 = fig.add_subplot(gs[3, 0])

# middle column (skip the gap column at index 1)
ax12 = fig.add_subplot(gs[0, 1])
cax12 = fig.add_subplot(gs[1, 1])
ax22 = fig.add_subplot(gs[2, 1])
cax22 = fig.add_subplot(gs[3, 1])

# Big axis on the right spanning all rows
ax_big = fig.add_subplot(gs[:, 3])

# Load the config
cfg = OmegaConf.load("configs/train.yaml")

# Load the trajectory
trajectory_id = 50  # reference trajectory (highest median enstrophy)
trajectory, trajectory_metadata = get_trajectory(cfg.walrus.dataset, trajectory_id)

# Load the errors
with open(f"errors/errors_{trajectory_id}.pkl", "rb") as f:
    errors = pickle.load(f)


@dataclass
class EnergySpectrum:
    k: np.ndarray
    E: np.ndarray


# Plot tracer density field
field_id = 0

# Top-left: tracer field
error = errors[0]
logging.info(f"Plotting tracer field for error {error.step}")
simulation = np.squeeze(
    trajectory["input_fields"][
        0, error.step : error.step + cfg.walrus.n_steps_input + 1, ..., field_id
    ][-1]
).numpy()
vmax = np.max(np.abs(simulation))
im11 = ax11.imshow(simulation, cmap="gray", vmin=-vmax, vmax=vmax)
ax11.set_title("Tracer")
ax11.text(
    0.01,
    -0.08,
    f"Timestep: {error.step + cfg.walrus.n_steps_input}",
    transform=ax11.transAxes,
    ha="left",
    va="top",
)
# ax11.set_title(f"Tracer: {error.step+cfg.walrus.n_steps_input}")
fig.colorbar(im11, cax=cax11, orientation="horizontal")

# Top-middle: bias
bias = np.squeeze(error.pred[..., field_id]) - np.squeeze(error.ref[..., field_id])
vmax = np.max(np.abs(bias))
im12 = ax12.imshow(bias, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
ax12.set_title("Bias")
fig.colorbar(im12, cax=cax12, orientation="horizontal")

# Big right
energy_spectrum_ref = EnergySpectrum(
    *compute_energy_spectrum(
        np.squeeze(error.ref[..., 1]), np.squeeze(error.ref[..., 2])
    )
)
energy_spectrum_pred = EnergySpectrum(
    *compute_energy_spectrum(
        np.squeeze(error.pred[..., 1]), np.squeeze(error.pred[..., 2])
    )
)
ax_big.plot(
    energy_spectrum_ref.k[1:],
    energy_spectrum_pred.E[1:] / energy_spectrum_ref.E[1:],
    "-",
    color="xkcd:blue",
    label=f"Timestep: {error.step + cfg.walrus.n_steps_input}",
)

# Bottom-left: tracer field
error = errors[-5]
logging.info(f"Plotting tracer field for error {error.step}")
simulation = np.squeeze(
    trajectory["input_fields"][
        0, error.step : error.step + cfg.walrus.n_steps_input + 1, ..., field_id
    ][-1]
).numpy()
vmax = np.max(np.abs(simulation))
im21 = ax21.imshow(simulation, cmap="gray", vmin=-vmax, vmax=vmax)
# ax21.set_title(f"Tracer: {error.step+cfg.walrus.n_steps_input}")
ax21.text(
    0.01,
    -0.08,
    f"Timestep: {error.step + cfg.walrus.n_steps_input}",
    transform=ax21.transAxes,
    ha="left",
    va="top",
)
fig.colorbar(im21, cax=cax21, orientation="horizontal")

# Botto-middle: bias
bias = np.squeeze(error.pred[..., field_id]) - np.squeeze(error.ref[..., field_id])
vmax = np.max(np.abs(bias))
im22 = ax22.imshow(bias, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
fig.colorbar(im22, cax=cax22, orientation="horizontal")

# Big right
energy_spectrum_ref = EnergySpectrum(
    *compute_energy_spectrum(
        np.squeeze(error.ref[..., 1]), np.squeeze(error.ref[..., 2])
    )
)
energy_spectrum_pred = EnergySpectrum(
    *compute_energy_spectrum(
        np.squeeze(error.pred[..., 1]), np.squeeze(error.pred[..., 2])
    )
)
ax_big.plot(
    energy_spectrum_ref.k[1:],
    energy_spectrum_pred.E[1:] / energy_spectrum_ref.E[1:],
    "-",
    color="xkcd:lightblue",
    label=f"Timestep: {error.step + cfg.walrus.n_steps_input}",
)

# # Horizontal colorbars in those dedicated axes
# fig.colorbar(im00, cax=cax_left, orientation="horizontal")
# fig.colorbar(im01,  cax=cax_mid,  orientation="horizontal")

logger.info("Cleaning up axes")
ax_big.legend(frameon=False, loc="upper left")
ax_big.set_xlabel("Wavenumber")
ax_big.set_ylabel("Walrus / Reference")
ax_big.set_title("Energy spectral ratio")
ax_big.set_yscale("log")
ax_big.set_xscale("log")
ax_big.set_xlim(0.48, 6)
ax_big.set_ylim(None, 1e4)
ax_big.axhline(1, color="gray", linestyle="--")


ax11.set_xticks([])
ax11.set_yticks([])
ax12.set_xticks([])
ax12.set_yticks([])
ax21.set_xticks([])
ax21.set_yticks([])
ax22.set_xticks([])
ax22.set_yticks([])

for cax in (cax11, cax12, cax21, cax22):
    cax.tick_params(labelsize=8)  # tick labels on the cbar axis
    cax.xaxis.label.set_size(9)  # if the label was set on the axis

fig.subplots_adjust(bottom=0.16, top=0.92, left=0.02, right=0.98)
plt.show()

savedir = Path("figures/preprint")
os.makedirs(savedir, exist_ok=True)
plt.savefig(savedir / "energy_spectra.png")
plt.close()
