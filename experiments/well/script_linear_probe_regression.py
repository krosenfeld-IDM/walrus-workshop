"""
Train a linear probe using SAE features as input and the deformation as output.
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
from walrus_workshop.metrics import subgrid_stress, coarsen_field, compute_deformation

import torch.nn as nn
import numpy as np
from dataclasses import dataclass
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

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
    r_squared_conditional: float = np.nan
    frac_active: float = 0.0

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
    n_t = cfg.walrus.n_steps_input
    spatial_size = 32
    assert act.shape[0] == n_t * spatial_size * spatial_size, \
        f"Expected {n_t * spatial_size * spatial_size} tokens, got {act.shape[0]}"
    with torch.no_grad():
        _, code, _ = sae_model(act)
    code = code.cpu().numpy()

    # Get simulation chunk
    simulation_chunk = trajectory['input_fields'][0, step:step+cfg.walrus.n_steps_input, :, :, 0, :]
    if verbose:
        print(f"Simulation chunk shape: {simulation_chunk.shape}")

    target_index_dicts = [{'u':2, 'v':3}, {'tau_xx': 0, 'tau_yy': 1, 'tau_xy': 2, 'tke': 3}, {'deformation': 0, 'shear_deformation': 1, 'stretch_deformation': 2}]
    target_index_dict = [target_index_dict for target_index_dict in target_index_dicts if target in target_index_dict][0]
    target_field = np.zeros((simulation_chunk.shape[0], 32, 32)) # 32 x 32 
    if target in ['deformation', 'shear_deformation', 'stretch_deformation']:
        for i in range(simulation_chunk.shape[0]):
            target_field[i] = coarsen_field(compute_deformation(simulation_chunk[i, ..., 1], simulation_chunk[i, ..., 2])[target_index_dict[target]], (32, 32))
    elif target in ['u', 'v']:
        for i in range(simulation_chunk.shape[0]):
            target_field[i]  = coarsen_field(simulation_chunk[i, ..., target_index_dict[target]], (32, 32), method='mean')
    elif target in ['tau_xx', 'tau_yy', 'tau_xy', 'tke']:
        for i in range(simulation_chunk.shape[0]):
            target_field[i] = subgrid_stress(simulation_chunk[i, ..., 1], simulation_chunk[i, ..., 2], (32, 32))[target_index_dict[target]]

    data_chunk = DataChunk(step=step, n_features=code.shape[1], n_timesteps=1, simulation=simulation_chunk[-1], code=code.reshape(n_t, spatial_size, spatial_size, -1)[-1], target=target_field[-1])
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
                lr=1e-3, n_steps=1000, device="cpu",
                balance_threshold=0.0):
    """
    Train a single linear probe for one SAE feature using weighted OLS.

    Args:
        alpha_j: (N,) feature activations for feature j
        target:  (N,) continuous target field (e.g. total deformation)
        lr: learning rate (unused, kept for API compatibility)
        n_steps: training iterations (unused, kept for API compatibility)
        device: "cpu" or "cuda"
        balance_threshold: target values above this are "positive" class

    Returns:
        Trained probe and final training loss
    """
    probe = LinearProbe().to(device)
    alpha_j = alpha_j.float().to(device)
    target = target.float().to(device)

    # Build per-sample weights based on class membership
    pos_mask = target > balance_threshold
    n_pos = pos_mask.sum().float()
    n_neg = (~pos_mask).sum().float()

    weights = torch.ones_like(target)
    if n_pos > 0 and n_neg > 0:
        weights[pos_mask] = 1.0 / (2.0 * n_pos)
        weights[~pos_mask] = 1.0 / (2.0 * n_neg)
    else:
        weights /= weights.sum()

    # Weighted OLS closed-form: minimize sum_i w_i (a * x_i + b - y_i)^2
    w_sum = weights.sum()
    x_mean = (weights * alpha_j).sum() / w_sum
    y_mean = (weights * target).sum() / w_sum

    x_centered = alpha_j - x_mean
    y_centered = target - y_mean

    a = (weights * x_centered * y_centered).sum() / \
        ((weights * x_centered * x_centered).sum() + 1e-8)
    b = y_mean - a * x_mean

    with torch.no_grad():
        probe.a.data.fill_(a.item())
        probe.b.data.fill_(b.item())

    # Compute weighted MSE for the returned loss
    with torch.no_grad():
        residuals = probe(alpha_j).squeeze() - target
        loss = (weights * residuals ** 2).sum().item()

    return probe, loss

def _train_probe(alpha_j: torch.Tensor, target: torch.Tensor,
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

    # Closed-form OLS for single-feature linear regression
    cov = ((alpha_j - alpha_j.mean()) * (target - target.mean())).mean()
    a = cov / (alpha_j.var() + 1e-8)
    b = target.mean() - a * alpha_j.mean()

    with torch.no_grad():
        probe.a.data.fill_(a.item())
        probe.b.data.fill_(b.item())

    # Compute final MSE loss for consistency with original return signature
    with torch.no_grad():
        loss = nn.MSELoss()(probe(alpha_j).squeeze(), target).item()

    return probe, loss


def evaluate_probe(probe, alpha_j, target, device="cpu"):
    """Compute R², conditional R², and Spearman correlation for a trained probe."""
    alpha_j = alpha_j.float().to(device)
    target = target.float().to(device)

    with torch.no_grad():
        pred = probe(alpha_j).squeeze()

    pred_np = pred.cpu().numpy()
    target_np = target.cpu().numpy()
    alpha_np = alpha_j.cpu().numpy()

    # R² (unconditional — over all tokens)
    ss_res = np.sum((target_np - pred_np) ** 2)
    ss_tot = np.sum((target_np - target_np.mean()) ** 2)
    r2 = 1 - ss_res / (ss_tot + 1e-8)

    # Conditional R² (only on tokens where feature fires)
    active_mask = alpha_np > 0
    if active_mask.sum() > 1:
        ss_res_cond = np.sum((target_np[active_mask] - pred_np[active_mask]) ** 2)
        ss_tot_cond = np.sum((target_np[active_mask] - target_np[active_mask].mean()) ** 2)
        r2_conditional = 1 - ss_res_cond / (ss_tot_cond + 1e-8)
        frac_active = active_mask.mean()
    else:
        r2_conditional = np.nan
        frac_active = 0.0

    # Spearman rank correlation (handle constant input)
    if np.std(pred_np) < 1e-10 or np.std(target_np) < 1e-10:
        rho, p_val = np.nan, np.nan
    else:
        rho, p_val = spearmanr(pred_np, target_np)

    return r2, rho, p_val, r2_conditional, frac_active


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

def probe_top_features_jointly(activations: torch.Tensor, target: torch.Tensor,
                               top_k=50, alpha=1.0, test_size=0.2,
                               random_state=42, verbose=True):
    """
    Multivariate ridge regression using top-k most-active SAE features.

    Fits a joint linear model y = X @ w + b using the top-k features
    (by activation frequency) to show whether features collectively
    explain the target even if individual R² values are low.

    Args:
        activations: (N, n_l) sparse feature activations
        target: (N,) continuous target values
        top_k: number of top features to select by activation frequency
        alpha: Ridge regularization strength
        test_size: fraction of data for validation
        random_state: random seed for train/test split
        verbose: print results

    Returns:
        dict with keys: r2_train, r2_val, top_indices, model
    """
    # Select features with highest activation frequency
    activity = (activations > 0).float().mean(dim=0)
    top_indices = activity.topk(top_k).indices

    X = activations[:, top_indices].cpu().numpy()
    y = target.cpu().numpy()

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)

    r2_train = model.score(X_train, y_train)
    r2_val = model.score(X_val, y_val)

    if verbose:
        print(f"\nJoint ridge probe (top {top_k} features, alpha={alpha}):")
        print(f"  R² train = {r2_train:.4f}")
        print(f"  R² val   = {r2_val:.4f}")

    return {
        "r2_train": r2_train,
        "r2_val": r2_val,
        "top_indices": top_indices,
        "model": model,
    }


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
    for j in range(n_l):
        # Skip dead features (all zeros)
        col = act_train[:, j]
        if col.abs().max() < 1e-10:
            continue

        probe, _ = train_probe(col, t_train, lr=lr, n_steps=n_steps, device=device)
        r2, rho, p_val, r2_cond, frac_active = evaluate_probe(probe, act_val[:, j], t_val, device=device)

        results.append(ProbeResult(
            feature_idx=j,
            r_squared=r2,
            spearman_rho=rho,
            spearman_p=p_val,
            weight=probe.a.item(),
            bias=probe.b.item(),
            r_squared_conditional=r2_cond,
            frac_active=frac_active,
        ))

        if verbose and (j + 1) % 500 == 0:
            print(f"  Probed {j+1}/{n_l} features")

    results.sort(key=lambda r: r.r_squared, reverse=True)

    if verbose:
        print("\nTop 5 features by R²:")
        for r in results[:5]:
            print(f"  Feature {r.feature_idx:5d} | R²={r.r_squared:.4f} "
                  f"| R²_cond={r.r_squared_conditional:.4f} "
                  f"| active={r.frac_active:.3f} "
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

    for target_name in alive_it(['deformation', 'shear_deformation', 'stretch_deformation', 'u', 'v', 'tau_xx', 'tau_yy', 'tau_xy', 'tke']):

        # Load the data
        activations = []
        target = []
        for step_index, step in enumerate(steps):
            data_chunk = get_data_chunk(step, step_index, act_files, trajectory, cfg, sae_model, device, verbose=False, target=target_name)
            activations.append(data_chunk.code)
            target.append(data_chunk.target)
        activations = torch.from_numpy(np.array(activations)).to(device) # [N, B, F]
        activations = activations.flatten(0, -2)  # flatten dims 0 through second-to-last
        target = torch.from_numpy(np.array(target)).to(device)
        target = target.flatten() # flatten all dims

        # Train the probe on the SAE features
        probe_results = probe_all_features(activations, target, device=device, verbose=True)

        # Multi-feature ridge regression probe
        ridge_results = probe_top_features_jointly(activations, target, top_k=50, alpha=1.0, verbose=True)

        # Save the probe results
        output_dir = Path("probes")
        os.makedirs(output_dir, exist_ok=True)
        with open(output_dir / f"probe_results_traj_{trajectory_id}_{target_name}.pkl", "wb") as f:
            pickle.dump(probe_results, f)
        with open(output_dir / f"probe_ridge_results_traj_{trajectory_id}_{target_name}.pkl", "wb") as f:
            pickle.dump(ridge_results, f)

        # clean up memory
        del activations
        gc.collect()
        torch.cuda.empty_cache()

        activations = []
        for act_file in act_files:
            act = np.array(zarr.open(act_file, mode="r"))
            activations.append(act)
        activations = torch.from_numpy(np.array(activations).reshape(len(steps), 6, 32, 32, -1))[:, -1, :, :, :].to(device) # [N B F]
        activations = activations.flatten(0, -2)

        probe_results = probe_neurons_baseline(activations, target, device=device, verbose=True)

        # Save the probe results
        output_dir = Path("probes")
        os.makedirs(output_dir, exist_ok=True)
        with open(output_dir / f"probe_baseline_results_traj_{trajectory_id}_{target_name}.pkl", "wb") as f:
            pickle.dump(probe_results, f)

        # clean up memory
        del activations
        gc.collect()
        torch.cuda.empty_cache()