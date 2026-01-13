"""
Compute statistics over the SAEs
"""

import os
import glob
import logging
import numpy as np
from alive_progress import alive_it
import torch
from walrus_workshop.utils import split_test_train
from walrus_workshop.model import load_sae

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.chdir(os.path.dirname(os.path.abspath(__file__)))

layer_name = "blocks.30.space_mixing.activation"
num_arrays = 150

save_dir = os.path.abspath(f"./activations/{layer_name}")
act_files = sorted(glob.glob(os.path.join(save_dir, "*.npy")))

train_files, test_files = split_test_train(act_files, random_state=42, test_size=0.2)
act_files = test_files[:10]

# Load the trained SAE
sae, config = load_sae(f"./checkpoints/sae_checkpoint_{layer_name}_num{num_arrays}.pt")
sae = sae.to(device).eval()

stats_by_file = {}
for file_idx, file in enumerate(alive_it(act_files)):
    # Load the activations
    act = np.load(file)

    # Move to device
    xb = torch.from_numpy(act).to(device)

    # Forward pass
    with torch.no_grad():
        recon, code, _ = sae(xb)   # [n_nodes, n_features]

    code_np = code.cpu().numpy()
    
    stats_by_file[file] = {
        "n_nodes": code_np.shape[0],
        "n_features": code_np.shape[1],
        "sparsity": np.count_nonzero(code_np) / code_np.size,
        "l0_norm": np.count_nonzero(code_np), 
        "l1_sum": np.sum(np.abs(code_np)),
        "l2_norm": ((recon - xb).pow(2).sum(dim=-1).mean()).cpu().numpy(),
    }

    print(
        f"Code shape={code_np.shape}, "
        f"sparsity={stats_by_file[file]['sparsity']:.4f}"
    )

print(f"Mean L2 norm: {np.mean([stats_by_file[file]['l2_norm'] for file in stats_by_file])}, std: {np.std([stats_by_file[file]['l2_norm'] for file in stats_by_file])}")
print(f"Mean L0 norm: {np.mean([stats_by_file[file]['l0_norm'] for file in stats_by_file])}, std: {np.std([stats_by_file[file]['l0_norm'] for file in stats_by_file])}")
print(f"Mean L1 sum: {np.mean([stats_by_file[file]['l1_sum'] for file in stats_by_file])}, std: {np.std([stats_by_file[file]['l1_sum'] for file in stats_by_file])}")
print(f"Mean sparsity: {np.mean([stats_by_file[file]['sparsity'] for file in stats_by_file])}, std: {np.std([stats_by_file[file]['sparsity'] for file in stats_by_file])}")