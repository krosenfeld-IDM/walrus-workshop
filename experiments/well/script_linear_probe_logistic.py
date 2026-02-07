"""
Train a linear probe using SAE features as input and the sign of Okubo-Weiss as classification target.

"""

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
from walrus_workshop.metrics import compute_okubo_weiss

import torch.nn as nn
import numpy as np
from dataclasses import dataclass

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



@dataclass
class Feature:
    index: int

@dataclass
class ProbeResult:
    feature_idx: int
    accuracy: float
    f1: float
    auroc: float
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

def get_data_chunk(step, step_index, act_files, trajectory, cfg, sae_model, device, verbose=False):

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

    Q_sign = np.zeros((simulation_chunk.shape[0], 32, 32)) # 32 x 32 
    for i in range(simulation_chunk.shape[0]):
        for ix in range(32):
            for iy in range(32):
                token = simulation_chunk[i, iy*scale_y:(iy+1)*scale_y, ix*scale_x:scale_x*(ix+1), :]
                Q_sign[i, iy, ix] = np.sign(compute_okubo_weiss(token[:, :, 2], token[:, :, 3])[0].mean())

    data_chunk = DataChunk(step=step, n_features=code.shape[1], n_timesteps=simulation_chunk.shape[0]-1, simulation=simulation_chunk[:-1], code=code, target=Q_sign)
    return data_chunk


class LogisticProbe(nn.Module):
    """Single-feature logistic probe: logit = a * alpha_j + b"""
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.zeros(1))
        self.b = nn.Parameter(torch.zeros(1))

    def forward(self, alpha_j):
        return self.a * alpha_j + self.b


def train_probe(alpha_j: torch.Tensor, target: torch.Tensor,
                lr=1e-3, n_steps=1000, device="cpu"):
    """
    Train a single logistic probe for one SAE feature.

    Args:
        alpha_j: (N,) feature activations for feature j
        target:  (N,) binary target in {-1, +1}
        lr: learning rate
        n_steps: training iterations
        device: "cpu" or "cuda"

    Returns:
        Trained probe and final training loss
    """
    probe = LogisticProbe().to(device)
    alpha_j = alpha_j.float().to(device)
    target = (target.float().to(device) + 1) / 2  # {-1,+1} -> {0,1}

    opt = torch.optim.Adam(probe.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    for step in range(n_steps):
        opt.zero_grad()
        logits = probe(alpha_j).squeeze()
        loss = loss_fn(logits, target)
        loss.backward()
        opt.step()

    return probe, loss.item()


def evaluate_probe(probe, alpha_j, target, device="cpu"):
    """Compute accuracy, F1, and AUROC for a trained logistic probe."""
    alpha_j = alpha_j.float().to(device)
    target = (target.float().to(device) + 1) / 2  # {-1,+1} -> {0,1}

    with torch.no_grad():
        logits = probe(alpha_j).squeeze()
        probs = torch.sigmoid(logits)

    pred_labels = (probs >= 0.5).float()

    # Accuracy
    accuracy = (pred_labels == target).float().mean().item()

    # F1 (positive class = 1)
    tp = ((pred_labels == 1) & (target == 1)).sum().float()
    fp = ((pred_labels == 1) & (target == 0)).sum().float()
    fn = ((pred_labels == 0) & (target == 1)).sum().float()
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = (2 * precision * recall / (precision + recall + 1e-8)).item()

    # AUROC via trapezoidal rule
    sorted_indices = torch.argsort(probs, descending=True)
    sorted_target = target[sorted_indices]
    n_pos = target.sum()
    n_neg = (1 - target).sum()
    if n_pos < 1 or n_neg < 1:
        auroc = float('nan')
    else:
        tpr = torch.cumsum(sorted_target, dim=0) / n_pos
        fpr = torch.cumsum(1 - sorted_target, dim=0) / n_neg
        tpr = torch.cat([torch.zeros(1, device=device), tpr])
        fpr = torch.cat([torch.zeros(1, device=device), fpr])
        auroc = torch.trapezoid(tpr, fpr).item()

    return accuracy, f1, auroc


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

def probe_all_features(activations: torch.Tensor, target: torch.Tensor,
                       train_frac=0.8, lr=1e-3, n_steps=1000,
                       device="cpu", verbose=True):
    """
    Train and evaluate a logistic probe for every SAE feature.

    Args:
        activations: (N, n_l) sparse feature activations across all nodes/times
        target:      (N,) binary target in {-1, +1}
        train_frac:  fraction of data for training
        lr, n_steps: training hyperparameters
        device: "cpu" or "cuda"

    Returns:
        List of ProbeResult sorted by AUROC (descending)
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
        accuracy, f1, auroc = evaluate_probe(probe, act_val[:, j], t_val, device=device)

        results.append(ProbeResult(
            feature_idx=j,
            accuracy=accuracy,
            f1=f1,
            auroc=auroc,
            weight=probe.a.item(),
            bias=probe.b.item(),
        ))

        if verbose and (j + 1) % 500 == 0:
            print(f"  Probed {j+1}/{n_l} features")

    results.sort(key=lambda r: r.auroc, reverse=True)

    if verbose:
        print("\nTop 5 features by AUROC:")
        for r in results[:5]:
            print(f"  Feature {r.feature_idx:5d} | AUROC={r.auroc:.4f} "
                  f"| F1={r.f1:.4f} | Acc={r.accuracy:.4f}")

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

    # Load the data
    activations = []
    target = []
    for step_index, step in enumerate(alive_it(steps)):
        data_chunk = get_data_chunk(step, step_index, act_files, trajectory, cfg, sae_model, device, verbose=False)
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
    with open(output_dir / f"probe_results_traj_{trajectory_id}_dEdt.pkl", "wb") as f:
        pickle.dump(probe_results, f)

    #