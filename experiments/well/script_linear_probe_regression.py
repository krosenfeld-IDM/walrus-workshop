"""
Train a linear probe using SAE features as input and the deformation as output.

Top 5 features by R² for trajectory 50 for deformation:
  Feature 13376 | R²=0.0031 | ρ=0.1105 (p=1.24e-113)
  Feature  8952 | R²=0.0028 | ρ=0.0835 (p=1.65e-65)
  Feature 22206 | R²=0.0026 | ρ=0.0746 (p=1.41e-52)
  Feature 14734 | R²=0.0024 | ρ=0.1030 (p=5.63e-99)
  Feature  9713 | R²=0.0023 | ρ=0.1174 (p=5.24e-128)

Top 5 features by R² for dEdt:
  Feature 13441 | R²=0.0011 | ρ=0.0762 (p=7.51e-55)
  Feature 14734 | R²=0.0011 | ρ=0.0637 (p=7.07e-39)
  Feature 16056 | R²=0.0010 | ρ=0.0765 (p=2.74e-55)
  Feature 19092 | R²=0.0010 | ρ=0.0633 (p=2.52e-38)
  Feature 17801 | R²=0.0009 | ρ=0.0699 (p=1.98e-46)
"""

import gc
import os
import pickle
import zarr
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt
from rich import print
from pathlib import Path
from omegaconf import OmegaConf
from dataclasses import dataclass
from alive_progress import alive_it
from sortedcontainers import SortedList
from walrus_workshop.utils import get_key_value_from_string
from walrus_workshop.walrus import get_trajectory
from walrus_workshop.model import load_sae
from walrus_workshop.metrics import subgrid_stress

import torch.nn as nn
import numpy as np
from dataclasses import dataclass
from scipy.stats import spearmanr

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



@dataclass
class Feature:
    index: int

@dataclass
class ProbeResult:
    feature_idx: int
    r_squared: float
    spearman_rho: float
    spearman_p: float
    weight: float
    bias: float

@dataclass
class DataChunk:
    step: int
    n_features: int
    n_timesteps: int
    simulation: np.ndarray
    code: np.ndarray
    target: np.ndarray

def get_data_chunk(step, step_index, act_files, trajectory, cfg, sae_model, device, verbose=False, target='tke'):

    # Get SAE features
    if verbose:
        print(f"Opening activation file {Path(act_files[step_index]).stem}")
    assert get_key_value_from_string(Path(act_files[step_index]).stem, "step") == step # make sure we are processing the same step
    act = zarr.open(act_files[step_index], mode="r")
    act = torch.from_numpy(np.array(act)).to(device)
    with torch.no_grad():
        _, code, _ = sae_model(act)
    code = code.cpu().numpy()

    # Get simulation chunk
    simulation_chunk = trajectory['input_fields'][0, step:step+cfg.walrus.n_steps_input+1, :, :, 0, :]
    if verbose:
        print(f"Simulation chunk shape: {simulation_chunk.shape}")

    scale_x = int(simulation_chunk.shape[2] / 32)  # width
    scale_y = int(simulation_chunk.shape[1] / 32)  # height

    target_index_dict = {'tau_xx': 0, 'tau_yy': 1, 'tau_xy': 2, 'tke': 3}
    target_field = np.zeros((simulation_chunk.shape[0], 32, 32)) # 32 x 32 
    for i in range(simulation_chunk.shape[0]):
        target_field[i] = subgrid_stress(simulation_chunk[i, ..., 1], simulation_chunk[i, ..., 2], (32, 32))[target_index_dict[target]]

    data_chunk = DataChunk(step=step, n_features=code.shape[1], n_timesteps=simulation_chunk.shape[0]-1, simulation=simulation_chunk[:-1], code=code, target=target_field[:-1])
    return data_chunk


class LinearProbe(nn.Module):
    """Single-feature linear probe: y_hat = a * alpha_j + b"""
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.zeros(1))
        self.b = nn.Parameter(torch.zeros(1))

    def forward(self, alpha_j):
        return self.a * alpha_j + self.b


def train_probe(alpha_j: torch.Tensor, target: torch.Tensor,
                lr=1e-3, n_steps=1000, device="cpu"):
    """
    Train a single linear probe for one SAE feature.

    Args:
        alpha_j: (N,) feature activations for feature j
        target:  (N,) continuous target field (e.g. total deformation)
        lr: learning rate
        n_steps: training iterations
        device: "cpu" or "cuda"

    Returns:
        Trained probe and final training loss
    """
    probe = LinearProbe().to(device)
    alpha_j = alpha_j.float().to(device)
    target = target.float().to(device)

    # Standardize target for stable training
    t_mean, t_std = target.mean(), target.std()
    target_norm = (target - t_mean) / (t_std + 1e-8)

    opt = torch.optim.Adam(probe.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for step in range(n_steps):
        opt.zero_grad()
        pred = probe(alpha_j).squeeze()
        loss = loss_fn(pred, target_norm)
        loss.backward()
        opt.step()

    # Rescale parameters back to original target scale
    with torch.no_grad():
        probe.a.data *= t_std
        probe.b.data = probe.b.data * t_std + t_mean

    return probe, loss.item()


def evaluate_probe(probe, alpha_j, target, device="cpu"):
    """Compute R² and Spearman correlation for a trained probe."""
    alpha_j = alpha_j.float().to(device)
    target = target.float().to(device)

    with torch.no_grad():
        pred = probe(alpha_j).squeeze()

    pred_np = pred.cpu().numpy()
    target_np = target.cpu().numpy()

    # R²
    ss_res = np.sum((target_np - pred_np) ** 2)
    ss_tot = np.sum((target_np - target_np.mean()) ** 2)
    r2 = 1 - ss_res / (ss_tot + 1e-8)

    # Spearman rank correlation (handle constant input)
    if np.std(pred_np) < 1e-10 or np.std(target_np) < 1e-10:
        rho, p_val = np.nan, np.nan
    else:
        rho, p_val = spearmanr(pred_np, target_np)

    return r2, rho, p_val


def plot_feature(feature, data_chunk, with_simulation=False, verbose=False, simulation_field=0):
    """ Plot feature """
    if verbose:
        print(f"Plotting feature {feature.index}")
    activations = data_chunk.code[:, feature.index].reshape(-1, 32, 32)
    vmax = np.max(np.abs(activations))
    fig, axs = plt.subplots(2, 3, figsize=(10, 5))
    for i in range(data_chunk.n_timesteps):
        ax = axs[i // 3, i % 3]
        cb = ax.imshow(activations[i], cmap='hot', vmin=0, vmax=vmax)
        fig.colorbar(cb, ax=ax, shrink=0.5)
        if with_simulation:
            ax.contour(np.linspace(-0.5, activations.shape[1]+0.5, data_chunk.simulation.shape[2]), 
            np.linspace(-0.5, activations.shape[2]+0.5, data_chunk.simulation.shape[1]), 
            data_chunk.simulation[i, ..., simulation_field], levels=1, colors='lavender')
        # ax.set_title(f"Step {data_chunk.step + i}"); ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    plt.show()

def probe_neurons_baseline(neuron_activations: torch.Tensor, target: torch.Tensor,
                           train_frac=0.8, lr=1e-3, n_steps=1000,
                           device="cpu", verbose=True):
    """
    Same analysis but on raw neuron activations (no SAE).
    Use this as a baseline to compare against SAE features.

    Args:
        neuron_activations: (N, n_d) raw node embeddings
        target: (N,) continuous target values

    Returns:
        List of ProbeResult sorted by R²
    """
    if verbose:
        print("Running neuron baseline probes...")
    return probe_all_features(neuron_activations, target,
                              train_frac=train_frac, lr=lr,
                              n_steps=n_steps, device=device,
                              verbose=verbose)

def probe_all_features(activations: torch.Tensor, target: torch.Tensor,
                       train_frac=0.8, lr=1e-3, n_steps=1000,
                       device="cpu", verbose=True):
    """
    Train and evaluate a linear probe for every SAE feature.

    Args:
        activations: (N, n_l) sparse feature activations across all nodes/times
        target:      (N,) continuous target values
        train_frac:  fraction of data for training
        lr, n_steps: training hyperparameters
        device: "cpu" or "cuda"

    Returns:
        List of ProbeResult sorted by R² (descending)
    """
    N, n_l = activations.shape
    assert target.shape[0] == N

    # Train/val split
    perm = torch.randperm(N)
    split = int(N * train_frac)
    train_idx, val_idx = perm[:split], perm[split:]

    act_train = activations[train_idx]
    act_val = activations[val_idx]
    t_train = target[train_idx]
    t_val = target[val_idx]

    results = []
    for j in alive_it(range(n_l)):
        # Skip dead features (all zeros)
        col = act_train[:, j]
        if col.abs().max() < 1e-10:
            continue

        probe, _ = train_probe(col, t_train, lr=lr, n_steps=n_steps, device=device)
        r2, rho, p_val = evaluate_probe(probe, act_val[:, j], t_val, device=device)

        results.append(ProbeResult(
            feature_idx=j,
            r_squared=r2,
            spearman_rho=rho,
            spearman_p=p_val,
            weight=probe.a.item(),
            bias=probe.b.item(),
        ))

        if verbose and (j + 1) % 500 == 0:
            print(f"  Probed {j+1}/{n_l} features")

    results.sort(key=lambda r: r.r_squared, reverse=True)

    if verbose:
        print(f"\nTop 5 features by R²:")
        for r in results[:5]:
            print(f"  Feature {r.feature_idx:5d} | R²={r.r_squared:.4f} "
                  f"| ρ={r.spearman_rho:.4f} (p={r.spearman_p:.2e})")

    return results

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    trajectory_id = 50
    trajectory, trajectory_metadata = get_trajectory(dataset_id="shear_flow", trajectory_id=trajectory_id)

    # Load the config
    cfg = OmegaConf.load("configs/train.yaml")

    # Load file list of the activations
    activations_dir = (
        Path("activations")
        / "test"
        / "blocks.20.space_mixing.activation"
        / cfg.walrus.dataset
    )
    act_files = sorted(glob.glob(str(activations_dir / f"*_traj_{trajectory_id}*")))
    # List of steps with activations (starting step)
    steps = np.array([int(get_key_value_from_string(file_name, "step")) for file_name in act_files])

    # Load the trained SAE
    checkpoint_path = (
        Path("checkpoints")
        / "sae_checkpoint_blocks.20.space_mixing.activation_source_test_k_active=32_k_aux=2048_latent=22528_beta=0.1.pt"
    )
    sae_model, sae_config = load_sae(checkpoint_path)
    sae_model = sae_model.to(device).eval()

    for target_name in ['tau_xy', 'tke', 'tau_xx', 'tau_yy']:

        # Load the data
        activations = []
        target = []
        for step_index, step in enumerate(alive_it(steps)):
            data_chunk = get_data_chunk(step, step_index, act_files, trajectory, cfg, sae_model, device, verbose=False, target=target_name)
            activations.append(data_chunk.code)
            target.append(data_chunk.target)
        activations = torch.from_numpy(np.array(activations)).to(device) # [N, B, F]
        activations = activations.flatten(0, -2)  # flatten dims 0 through second-to-last
        target = torch.from_numpy(np.array(target)).to(device)
        target = target.flatten() # flatten all dims

        # Train the probe on the SAE features
        probe_results = probe_all_features(activations, target, device=device, verbose=True)

        # Save the probe results
        output_dir = Path("probes")
        os.makedirs(output_dir, exist_ok=True)
        with open(output_dir / f"probe_results_traj_{trajectory_id}_{target}.pkl", "wb") as f:
            pickle.dump(probe_results, f)

        # clean up memory
        del activations
        gc.collect()
        torch.cuda.empty_cache()

        activations = []
        for act_file in act_files:
            act = np.array(zarr.open(act_file, mode="r"))
            activations.append(act)
        activations = torch.from_numpy(np.array(activations)).to(device) # [N B F]
        activations = activations.flatten(0, -2)

        probe_results = probe_neurons_baseline(activations, target, device=device, verbose=True)

        # Save the probe results
        output_dir = Path("probes")
        os.makedirs(output_dir, exist_ok=True)
        with open(output_dir / f"probe_baseline_results_traj_{trajectory_id}_{target}.pkl", "wb") as f:
            pickle.dump(probe_results, f)

        # clean up memory
        del activations
        gc.collect()
        torch.cuda.empty_cache()