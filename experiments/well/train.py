"""
Train TopK SAE on activations
"""
import os
import glob
import inspect
import logging
import sys
from walrus_workshop.model import SAE
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.data import Dataset
from alive_progress import alive_it

# Setup logger with handler to output to terminal
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Create console handler if one doesn't exist
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

class NumpyListDataset(Dataset):
    def __init__(self, numpy_arrays, device="cpu"):
        """
        Args:
            numpy_arrays: List of np.ndarray, each shape [A, d_in]
        """
        # Concatenate all arrays into one large tensor
        # If dataset is too huge for RAM, you would map index -> specific array
        logger.info("Concatenating arrays...")
        self.data = torch.from_numpy(np.concatenate(numpy_arrays, axis=0)).float()
        
        # Optional: Move to GPU immediately if VRAM allows for speed
        # self.data = self.data.to(device)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def train_sae(
    sae_model, 
    numpy_data_list, 
    batch_size=4096, 
    lr=3e-4, 
    epochs=10, 
    device="cuda"
):
    # 1. Setup Data
    dataset = NumpyListDataset(numpy_data_list)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 2. Setup Optimizer
    sae_model = sae_model.to(device)
    optimizer = optim.Adam(sae_model.parameters(), lr=lr, betas=(0.9, 0.999))
    
    # Optional: Learning rate warmup/decay is often used for SAEs
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * len(dataloader))

    print(f"Starting training on {len(dataset)} samples...")
    
    for epoch in range(epochs):
        sae_model.train()
        total_loss = 0
        total_mse = 0
        total_aux = 0
        
        for batch_idx, x in alive_it(enumerate(dataloader)):
            x = x.to(device)
            
            # --- Forward Pass ---
            # recon: Main reconstruction
            # code: Latent activations
            # aux_recon: Reconstruction from dead neurons
            recon, code, aux_recon = sae_model(x)
            
            # --- Loss Calculation ---
            # Note: The model forward pass normalizes x internally.
            # To compute loss correctly, we need that normalized version of x.
            # Let's manually replicate the normalization for the target
            # OR trust that the reconstruction should match the input distribution provided.
            
            # Re-normalize x locally to match model's internal normalization logic
            # so MSE is valid.
            x_centered = x - x.mean(dim=1, keepdim=True)
            x_normed = x_centered / x_centered.norm(dim=1, keepdim=True).clamp_min(1e-6)
            
            # 1. Main Reconstruction Loss (MSE)
            mse_loss = (recon - x_normed).pow(2).sum(dim=-1).mean()
            
            # 2. AuxK Loss
            # We want dead neurons (aux_recon) to predict the RESIDUAL (x - recon).
            # We detach the residual target so aux loss doesn't affect the main decoder.
            residual = (x_normed - recon).detach()
            aux_loss = (aux_recon - residual).pow(2).sum(dim=-1).mean()
            
            # Combine
            loss = mse_loss + aux_loss
            
            # --- Backward ---
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            # --- Post-Optimization Constraints ---
            with torch.no_grad():
                # 1. Enforce Unit Norm Decoder columns
                sae_model._renorm_decoder_columns_()
                
                # 2. Update Dead Neuron Statistics
                sae_model.update_dead_mask(code, batch_size=x.shape[0])

            # Logging
            total_loss += loss.item()
            total_mse += mse_loss.item()
            total_aux += aux_loss.item()
            
            if batch_idx % 100 == 0:
                logger.info(f"Epoch {epoch+1} | Batch {batch_idx} | "
                      f"MSE: {mse_loss.item():.4f} | Aux: {aux_loss.item():.4f} | "
                      f"Dead: {sae_model.dead_mask.sum().item()}")

        avg_loss = total_loss / len(dataloader)
        logger.info(f"=== Epoch {epoch+1} Finished. Avg Loss: {avg_loss:.5f} ===")

    return sae_model

def save_sae(save_path, cfg=None, model=None):
    # Assuming 'trained_model' is your model instance
    # and you have your config variables available

    checkpoint = {
        # 1. The Model Architecture Arguments
        "config": cfg,
        # 2. The Model Weights (including dead_mask and miss_counts buffers)
        "model_state_dict": model.state_dict()
    }

    torch.save(checkpoint, save_path)
    logger.info(f"Model saved to {save_path}")    

def load_sae(save_path):
    logger.info(f"Loading model from {save_path}")

    # 1. Load
    checkpoint = torch.load(save_path)
    config = checkpoint["config"]

    # 2. Instantiate (The Pythonic Way)
    loaded_model = SAE(**config) 

    # 3. Load weights
    loaded_model.load_state_dict(checkpoint["model_state_dict"])

    return loaded_model, config

def train_demo():

    # Hyperparameters
    cfg = {
        "d_in": 768,
        "latent": 768*4,
        "k_active": 32,
        "k_aux": 512,
        "dead_window": 50_000,
    }
    
    # 1. Generate Dummy Data (N numpy arrays)
    # Simulating 5 arrays, each with 10k samples of dim 768
    num_arrays = 5
    samples_per_array = 10_000
    numpy_arrays = [np.random.randn(samples_per_array, cfg.get("d_in", 768)).astype(np.float32) 
                    for _ in range(num_arrays)]

    # 2. Initialize Model
    model = SAE(
        d_in=cfg.get("d_in", 768),
        latent=cfg.get("latent", 768*4),
        k_active=cfg.get("k_active", 32),
        k_aux=cfg.get("k_aux", 512),
        dead_window=cfg.get("dead_window", 50_000), # Set smaller for this demo to see updates
    )

    # 3. Train
    device = "cuda" if torch.cuda.is_available() else "cpu"
    trained_model = train_sae(
        model, 
        numpy_arrays, 
        batch_size=1024, 
        lr=3e-4, 
        epochs=3, 
        device=device
    )
    return trained_model, cfg  

def train_walrus():  
    layer_name = "blocks.20.space_mixing.activation"
    save_dir = os.path.abspath(f"./activations/{layer_name}")
    act_files = glob.glob(os.path.join(save_dir, "*.npy"))
    act_shape = np.load(act_files[0]).shape
    cfg = {
        "d_in": act_shape[1],
        "latent": 4096, # np.prod(act_shape),
        "k": 32,
        "k_aux": 512,
    }

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    trained_model, cfg = train_demo()
    save_sae(save_path="sae_checkpoint.pt", cfg=cfg, model=trained_model)

    # new_model, new_cfg = load_sae(save_path="sae_checkpoint.pt")


# os.chdir(os.path.dirname(os.path.abspath(__file__)))

# layer_name = "blocks.20.space_mixing.activation"
# save_dir = os.path.abspath(f"./activations/{layer_name}")
# act_files = glob.glob(os.path.join(save_dir, "*.npy"))

# act_shape = np.load(act_files[0]).shape
# cfg = {
#     "d_in": act_shape[1],
#     "latent": 4096, # np.prod(act_shape),
#     "k": 32,
#     "k_aux": 512,
# }

# device = "cuda" if torch.cuda.is_available() else "cpu"

# model = SAE(
#     d_in=cfg.get("d_in", 512),
#     latent=cfg.get("latent", 4096),
#     k_active=cfg.get("k", 32),
#     k_aux=cfg.get("k_aux", 512),
# )

# model = model.to(device).eval()
