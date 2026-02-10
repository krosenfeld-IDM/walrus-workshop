"""
Iterate over all trajectories and calculate as a function of time corresponding to the present
activations:
1. total enstrophy
2. directionality of enstrophy flux
"""

import os
import pickle
import argparse
import zarr
import glob
import torch
import logging
import numpy as np
import matplotlib.pyplot as plt
from rich import print
from pathlib import Path
from omegaconf import OmegaConf
from dataclasses import dataclass
from alive_progress import alive_it
from sortedcontainers import SortedList
from scipy import stats
from walrus_workshop.utils import get_key_value_from_string
from walrus_workshop.walrus import get_trajectory
from walrus_workshop.model import load_sae
from walrus_workshop import metrics

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class EnstrophyFluxFeatures:
    k: np.array
    Pi: np.array
    T: np.array


def calculate_metrics(Pi_omega, T_omega, k_centers, k_forcing=None):
    dk = k_centers[1] - k_centers[0]
    metrics = {}
    
    # 1. Maximum flux magnitude and location
    metrics['Pi_max'] = np.max(Pi_omega)
    metrics['Pi_min'] = np.min(Pi_omega)
    metrics['k_at_Pi_max'] = k_centers[np.argmax(Pi_omega)]
    
    # 2. Mean flux in inertial range (if k_forcing provided)
    if k_forcing is not None:
        inertial_mask = k_centers > k_forcing
        metrics['Pi_mean_inertial'] = np.mean(Pi_omega[inertial_mask])
    else:
        # Use middle third of wavenumber range as proxy
        n = len(k_centers)
        mid_mask = (np.arange(n) > n//3) & (np.arange(n) < 2*n//3)
        metrics['Pi_mean_middle'] = np.mean(Pi_omega[mid_mask])
    
    # 3. Net enstrophy transfer (should be ~0 for conservative nonlinearity)
    metrics['net_transfer'] = np.sum(T_omega) * dk
    
    # 4. Forward vs inverse fraction
    forward_flux = np.sum(np.maximum(Pi_omega, 0))
    inverse_flux = np.sum(np.maximum(-Pi_omega, 0))
    total = forward_flux + inverse_flux
    if total > 0:
        metrics['forward_fraction'] = forward_flux / total
        metrics['inverse_fraction'] = inverse_flux / total
    else:
        metrics['forward_fraction'] = 0.5
        metrics['inverse_fraction'] = 0.5
    
    # 5. Cascade directionality index: ranges from -1 (inverse) to +1 (forward)
    metrics['directionality'] = (forward_flux - inverse_flux) / (total + 1e-16)
    
    # 6. Flux-weighted mean wavenumber (where is transfer happening?)
    weights = np.abs(Pi_omega)
    if np.sum(weights) > 0:
        metrics['k_flux_weighted'] = np.sum(k_centers * weights) / np.sum(weights)
    else:
        metrics['k_flux_weighted'] = 0
    
    # 7. Transfer spectrum moments
    T_abs = np.abs(T_omega)
    if np.sum(T_abs) > 0:
        metrics['k_transfer_centroid'] = np.sum(k_centers * T_abs) / np.sum(T_abs)
        metrics['transfer_bandwidth'] = np.sqrt(
            np.sum((k_centers - metrics['k_transfer_centroid'])**2 * T_abs) / np.sum(T_abs)
        )
    
    # 8. Plateau flux estimate (median of positive flux region)
    positive_flux = Pi_omega[Pi_omega > 0]
    if len(positive_flux) > 0:
        metrics['Pi_plateau'] = np.median(positive_flux)
    else:
        metrics['Pi_plateau'] = 0
    
    # 9. Enstrophy dissipation estimate (flux at high k)
    high_k_mask = k_centers > 0.7 * k_centers.max()
    metrics['Pi_high_k'] = np.mean(Pi_omega[high_k_mask])
    return metrics

def run_enstrophy_calculation(trajectory_id):
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

    directionality = np.array([])
    plateau = np.array([])
    t = np.array([])
    for step_index in range(len(steps)):
        step = steps[step_index]
        simulation_chunk = trajectory['input_fields'][0, step:step+cfg.walrus.n_steps_input, :, :, 0, :]
        for i in range(cfg.walrus.n_steps_input):
            t_ = step + i
            if t_ in t:
                continue
            enstrophy_flux = EnstrophyFluxFeatures(*metrics.compute_enstrophy_flux(simulation_chunk[i, ..., 2], simulation_chunk[i, ..., 3]))
            m = calculate_metrics(enstrophy_flux.Pi, enstrophy_flux.T, enstrophy_flux.k)
            directionality = np.concatenate([directionality, [m['directionality']]])
            plateau = np.concatenate([plateau, [m['Pi_plateau']]])
            t = np.concatenate([t, [step+i]])

    output_dir = Path("metrics") / "enstrophy_flux" 
    os.makedirs(output_dir, exist_ok=True)
    with open(output_dir / f"trajectory_{trajectory_id}.pkl", "wb") as f:
        pickle.dump({"directionality": directionality, "plateau": plateau, "t": t}, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--calculate_enstrophy_flux", type=int, default=0)
    args = parser.parse_args()

    os.chdir(Path(__file__).parent)
    # Load the config
    cfg = OmegaConf.load("configs/train.yaml")

    if args.calculate_enstrophy_flux:
        for trajectory_id in alive_it(range(112)):
            run_enstrophy_calculation(trajectory_id)