# (walrus-workshop) krosenfeld@internal.idm.ctr@ipapvwks23:~/projects/walrus-workshop/experiments/well$ /home/krosenfeld/projects/walrus-workshop/.venv/bin/python /home/krosenfeld/projects/walrus-workshop/experiments/well/script_enstrophy_rho.py
# Using device: cuda
# on 5: /home/krosenfeld/projects/walrus-workshop/experiments/well/script_enstrophy_rho.py:130: ConstantInputWarning: An input array is constant; the correlation coefficient is not defined.
#         s,p = stats.spearmanr(data["features"][:, feature_idx], data["enstrophy"])
#       /home/krosenfeld/projects/walrus-workshop/experiments/well/script_enstrophy_rho.py:138: ConstantInputWarning: An input array is constant; the correlation coefficient is not defined.
#         s,p = stats.spearmanr(data["features"][:, feature_idx], target)
# |████████████████████████████████████████| 22528/22528 [100%] in 8:03.6 (46.59/s) 
# Max rho: 0.8851957616766555, p-value: 0.0 (index: 7090)
# Null max rho: 0.0392585340591056, p-value: 2.0628250904777538e-05 (index: 2551)
# Significance threshold: 0.7904258524952378
# on 5: /home/krosenfeld/projects/walrus-workshop/experiments/well/script_enstrophy_rho.py:159: ConstantInputWarning: An input array is constant; the correlation coefficient is not defined.
#         s,p = stats.spearmanr(data["features"][:, feature_idx], data["dEdt"])
#       /home/krosenfeld/projects/walrus-workshop/experiments/well/script_enstrophy_rho.py:167: ConstantInputWarning: An input array is constant; the correlation coefficient is not defined.
#         s,p = stats.spearmanr(data["features"][:, feature_idx], target)
# |████████████████████████████████████████| 22528/22528 [100%] in 8:04.1 (46.53/s) 
# Max rho: 0.5181077509184498, p-value: 0.0 (index: 16914)
# Null max rho: 0.042540354867597094, p-value: 3.945435329866102e-06 (index: 18313)
# Significance threshold: 0.30940763006072963

import os
import logging
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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Feature:
    index: int
    corr: float
    p_value: float

    def __lt__(self, other):
        return self.corr < other.corr

def process_trajectory(trajectory_id):
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

    enstrophy = np.array([])
    dEdt = np.array([])
    features = np.array([])
    t = np.array([])
    for step_index in range(len(steps)):
        step = steps[step_index]

        # Calculate enstrophy
        simulation_chunk = trajectory['input_fields'][0, step:step+cfg.walrus.n_steps_input+1, :, :, 0, :]
        enstrophy_ = []
        for i in range(simulation_chunk.shape[0]):
            enstrophy_.append(metrics.compute_enstrophy(simulation_chunk[i, ..., 2], simulation_chunk[i, ..., 3])[0])


        # Process activations
        act = zarr.open(act_files[step_index], mode="r")
        act = torch.from_numpy(np.array(act)).to(device)
        with torch.no_grad():
            _, code, _ = sae_model(act)
        code = code.cpu().numpy().reshape(6, 32, 32, -1)

        # Save actuvatuibs and features
        for i in range(len(enstrophy_)-1):
            if step+i in t:
                continue
            dEdt = np.concatenate([dEdt, [np.diff(enstrophy_)[i]]])
            enstrophy = np.concatenate([enstrophy, [enstrophy_[i]]])
            features = np.concatenate([features, code.mean(axis=(1, 2))[i].ravel()])
            t = np.concatenate([t, [step+i]])
                
    return t, enstrophy, dEdt, features.reshape(len(enstrophy), -1)


def step_1_generate_data():

    # Load the config
    cfg = OmegaConf.load("configs/train.yaml")

    # Load the trained SAE
    checkpoint_path = (
        Path("checkpoints")
        / "sae_checkpoint_blocks.20.space_mixing.activation_source_test_k_active=32_k_aux=2048_latent=22528_beta=0.1.pt"
    )
    sae_model, sae_config = load_sae(checkpoint_path)
    sae_model = sae_model.to(device).eval()    

    t = []
    enstrophy = []
    dEdt = []
    features = []
    traj = []
    for trajectory_id in alive_it(range(0, 112)):
        t_, enstrophy_, dEdt_, features_ = process_trajectory(trajectory_id)


        t = np.concatenate([t, t_])
        enstrophy = np.concatenate([enstrophy, enstrophy_])
        dEdt = np.concatenate([dEdt, dEdt_])
        features = np.concatenate([features, features_.flatten()])
        traj = np.concatenate([traj, [trajectory_id] * len(t_)])

    output_dir = Path("metrics") / "enstrophy_rho" 
    os.makedirs(output_dir, exist_ok=True)
    with open(output_dir / "all_data_enstrphy.pkl", "wb") as f:
        pickle.dump({"t": t, "traj": traj, "enstrophy": enstrophy, "dEdt": dEdt, "features": features.reshape(len(enstrophy), -1)}, f)


def step_2_calculate_rho():

    # Load the data
    with open(Path("metrics") / "enstrophy_rho" / "all_data_enstrphy.pkl", "rb") as f:
        data = pickle.load(f)

    # Calculate the rho over all features
    feature_list = SortedList()
    num_features = data["features"].shape[1]
    null_feature_list = SortedList()
    num_null_samples = 10
    for feature_idx in alive_it(range(num_features)):
        s,p = stats.spearmanr(data["features"][:, feature_idx], data["enstrophy"])
        if np.isfinite(s):
            feature_list.add(Feature(feature_idx, s, p))
        s_max = -np.inf
        p_max = 0
        for i in range(num_null_samples):
            target = data["enstrophy"].copy()
            np.random.shuffle(target)
            s,p = stats.spearmanr(data["features"][:, feature_idx], target)
            if np.isfinite(s) and (s > s_max):
                s_max = s
                p_max = p
            null_feature_list.add(Feature(feature_idx, s_max, p_max))

    print(f"Max rho: {feature_list[-1].corr}, p-value: {feature_list[-1].p_value} (index: {feature_list[-1].index})")
    print(f"Null max rho: {null_feature_list[-1].corr}, p-value: {null_feature_list[-1].p_value} (index: {null_feature_list[-1].index})")
    thresh = np.quantile([feature.corr for feature in feature_list], 0.99)
    print(f"Significance threshold: {thresh}")

    output_dir = Path("metrics") / "enstrophy_rho" 
    os.makedirs(output_dir, exist_ok=True)
    with open(output_dir / "feature_list_enstrophy.pkl", "wb") as f:
        pickle.dump({'feature_ids': [feature.index for feature in feature_list[::-1]], 'significance_threshold': thresh, 'note': 'sorted in descenting order of rho'}, f)

    # Calculate the rho over all features
    feature_list = SortedList()
    num_features = data["features"].shape[1]
    null_feature_list = SortedList()
    for feature_idx in alive_it(range(num_features)):
        s,p = stats.spearmanr(data["features"][:, feature_idx], data["dEdt"])
        if np.isfinite(s):
            feature_list.add(Feature(feature_idx, s, p))
        s_max = -np.inf
        p_max = 0
        target = data["dEdt"].copy()
        for i in range(num_null_samples):
            np.random.shuffle(target)
            s,p = stats.spearmanr(data["features"][:, feature_idx], target)
            if np.isfinite(s) and (s > s_max):
                s_max = s
                p_max = p
            null_feature_list.add(Feature(feature_idx, s_max, p_max))

    print(f"Max rho: {feature_list[-1].corr}, p-value: {feature_list[-1].p_value} (index: {feature_list[-1].index})")
    print(f"Null max rho: {null_feature_list[-1].corr}, p-value: {null_feature_list[-1].p_value} (index: {null_feature_list[-1].index})")
    thresh = np.quantile([feature.corr for feature in feature_list], 0.99)
    print(f"Significance threshold: {thresh}")

    output_dir = Path("metrics") / "enstrophy_rho" 
    os.makedirs(output_dir, exist_ok=True)
    with open(output_dir / "feature_list_dEdt.pkl", "wb") as f:
        pickle.dump({'feature_ids': [feature.index for feature in feature_list[::-1]], 'significance_threshold': thresh, 'note': 'sorted in descenting order of rho'}, f)

if __name__ == "__main__":

    os.chdir(Path(__file__).parent)

    # step_1_generate_data()

    step_2_calculate_rho()