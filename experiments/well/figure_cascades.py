import os
import pickle

import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from dataclasses import dataclass

from walrus_workshop.metrics import compute_energy_spectrum
from script_walrus_errors import WalrusError

@dataclass
class EnergySpectrum:
    k: np.ndarray
    E: np.ndarray

cmap = {'ref': 'xkcd:blue', 'pred': 'xkcd:lightblue'}

os.chdir(os.path.dirname(os.path.abspath(__file__)))

data_dir = Path("errors")

fig, axs = plt.subplots(1, 2, figsize=(10, 3.5), dpi=300, sharey=True, sharex=True)

for i, traj_id in enumerate([50, 3]):
    ax = axs[i]
    errors = pickle.load(open(data_dir / f"errors_{traj_id}.pkl", "rb"))
    steps = [error.step for error in errors]

    error = errors[0]
    energy_spectrum_ref = EnergySpectrum(*compute_energy_spectrum(np.squeeze(error.ref[..., 1]), np.squeeze(error.ref[..., 2])))
    energy_spectrum_pred = EnergySpectrum(*compute_energy_spectrum(np.squeeze(error.pred[..., 1]), np.squeeze(error.pred[..., 2])))
    ax.plot(energy_spectrum_ref.k, energy_spectrum_ref.E, color=cmap['ref'], ls='-')
    ax.plot(energy_spectrum_pred.k, energy_spectrum_pred.E, color=cmap['pred'], ls='-')
    ax.plot([],[],'k-', label=f'Step {error.step}')


    error = errors[-1]
    energy_spectrum_ref = EnergySpectrum(*compute_energy_spectrum(np.squeeze(error.ref[..., 1]), np.squeeze(error.ref[..., 2])))
    energy_spectrum_pred = EnergySpectrum(*compute_energy_spectrum(np.squeeze(error.pred[..., 1]), np.squeeze(error.pred[..., 2])))
    ax.plot(energy_spectrum_ref.k, energy_spectrum_ref.E, color=cmap['ref'], ls='--')
    ax.plot(energy_spectrum_pred.k, energy_spectrum_pred.E, color=cmap['pred'], ls='--')
    ax.plot([],[],'k--', label=f'Step {error.step}')

    ax.set_xlabel("Wavenumber")
    ax.set_ylabel("Energy Spectrum")
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xlim(0.05, None)
    ax.set_ylim(1e-15, None)
    ax.legend(frameon=False, fontsize=10, loc='upper right')
    ax.set_title(f"Simulation {traj_id}")

    if i == 0:
        ax.text(0.02, 0.02, 'Numerical simulation', transform=ax.transAxes, ha='left', va='bottom', fontsize=12, color=cmap['ref'])
        ax.text(0.02, 0.1, 'Walrus', transform=ax.transAxes, ha='left', va='bottom', fontsize=12, color=cmap['pred'])



plt.tight_layout()
plt.savefig("figures/preprint/cascades.png")
plt.close()

