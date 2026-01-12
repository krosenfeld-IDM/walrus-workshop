"""
Train TopK SAE on activations
"""

import os
import glob
import logging
import sys
from walrus_workshop.model import SAE
from walrus_workshop.utils import split_test_train
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset
import torch.optim as optim
from alive_progress import alive_it
import wandb

# Setup logger with handler to output to terminal
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Create console handler if one doesn't exist
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
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


class LazyNumpyDataset(IterableDataset):
    """
    Memory-efficient dataset that streams from .npy files without loading all into RAM.

    Uses numpy's mmap_mode="r" to memory-map files, allowing the OS to page data
    in/out as needed. This keeps RAM usage bounded regardless of total dataset size.
    """

    def __init__(self, file_paths, d_in, batch_size=4096, seed=0):
        """
        Args:
            file_paths: List of paths to .npy files
            d_in: Expected feature dimension
            batch_size: Number of samples per batch
            seed: Random seed for shuffling
        """
        super().__init__()
        self.files = file_paths
        self.d_in = d_in
        self.batch = batch_size
        self.seed = seed

        # Pre-compute file metadata using mmap (doesn't load data into RAM)
        self.file_meta = []
        for f in self.files:
            arr = np.load(f, mmap_mode="r")
            assert arr.ndim == 2 and arr.shape[1] == d_in, f"{f} has shape {arr.shape}"
            self.file_meta.append({"path": f, "n_samples": arr.shape[0]})

        self.total_samples = sum(m["n_samples"] for m in self.file_meta)
        self.total_batches = (self.total_samples + batch_size - 1) // batch_size

    def __iter__(self):
        # Handle multi-worker sharding
        worker = torch.utils.data.get_worker_info()
        nw = worker.num_workers if worker else 1
        wid = worker.id if worker else 0
        rng = np.random.default_rng(self.seed + 997 * wid)

        # Each worker gets a shard of files
        file_shard = self.file_meta[wid::nw]
        rng.shuffle(file_shard)

        for md in file_shard:
            X = np.load(md["path"], mmap_mode="r")  # Memory-mapped, not loaded
            n = X.shape[0]
            perm = rng.permutation(n)
            for start in range(0, n, self.batch):
                sel = perm[start : start + self.batch]
                if len(sel) == 0:
                    break
                # np.asarray forces a copy from mmap to contiguous array
                yield torch.from_numpy(np.asarray(X[sel, :])).float()


def train_sae(
    sae_model,
    dataloader,
    total_samples,
    batches_per_epoch,
    lr=3e-4,
    epochs=10,
    device="cuda",
    wandb_cfg=None,
    sae_cfg=None,
):
    """
    Train SAE model.

    Args:
        sae_model: The SAE model to train
        dataloader: PyTorch DataLoader yielding batches
        total_samples: Total number of samples (for logging)
        batches_per_epoch: Number of batches per epoch (for scheduler)
        lr: Learning rate
        epochs: Number of epochs
        device: Device to train on
        wandb_cfg: Wandb configuration dict
        sae_cfg: SAE configuration dict for logging
    """
    # Initialize wandb if requested
    use_wandb = wandb_cfg is not None and wandb_cfg.get("use_wandb", False)
    if use_wandb:
        # Merge SAE config with wandb config for metadata
        wandb_config = wandb_cfg.copy()
        if sae_cfg is not None:
            wandb_config.update(sae_cfg)
        # Also add training hyperparameters
        wandb_config.update(
            {
                "learning_rate": lr,
                "epochs": epochs,
                "device": device,
            }
        )

        wandb.init(
            project=wandb_cfg.get("wandb_project", "walrus-workshop"),
            name=wandb_cfg.get("wandb_run_name", None),
            config=wandb_config,
        )
        logger.info("Wandb logging enabled")

    # Setup Optimizer
    sae_model = sae_model.to(device)
    optimizer = optim.Adam(sae_model.parameters(), lr=lr, betas=(0.9, 0.999))

    # Learning rate warmup/decay
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs * batches_per_epoch
    )

    print(f"Starting training on {total_samples} samples...")

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

            # Wandb logging (every 100 batches to avoid too much logging)
            if use_wandb and batch_idx % 100 == 0:
                current_lr = scheduler.get_last_lr()[0]
                wandb.log(
                    {
                        "train/loss": loss.item(),
                        "train/mse_loss": mse_loss.item(),
                        "train/aux_loss": aux_loss.item(),
                        "train/dead_neurons": sae_model.dead_mask.sum().item(),
                        "train/learning_rate": current_lr,
                        "epoch": epoch,
                        "batch": batch_idx,
                        "global_step": epoch * batches_per_epoch + batch_idx,
                    }
                )

            if batch_idx % 100 == 0:
                logger.info(
                    f"Epoch {epoch + 1} | Batch {batch_idx} | "
                    f"MSE: {mse_loss.item():.4f} | Aux: {aux_loss.item():.4f} | "
                    f"Dead: {sae_model.dead_mask.sum().item()}"
                )

        # Use actual batch count for averaging (batch_idx is 0-indexed, so add 1)
        actual_batches = batch_idx + 1
        avg_loss = total_loss / actual_batches
        avg_mse = total_mse / actual_batches
        avg_aux = total_aux / actual_batches

        # Log epoch-level metrics to wandb
        if use_wandb:
            wandb.log(
                {
                    "epoch/avg_loss": avg_loss,
                    "epoch/avg_mse_loss": avg_mse,
                    "epoch/avg_aux_loss": avg_aux,
                    "epoch/dead_neurons": sae_model.dead_mask.sum().item(),
                    "epoch/learning_rate": scheduler.get_last_lr()[0],
                    "epoch": epoch + 1,
                }
            )

        logger.info(f"=== Epoch {epoch + 1} Finished. Avg Loss: {avg_loss:.5f} ===")

    # Finish wandb run
    if use_wandb:
        wandb.finish()

    return sae_model


def save_sae(save_path, cfg=None, model=None):
    # Assuming 'trained_model' is your model instance
    # and you have your config variables available

    checkpoint = {
        # 1. The Model Architecture Arguments
        "config": cfg,
        # 2. The Model Weights (including dead_mask and miss_counts buffers)
        "model_state_dict": model.state_dict(),
    }

    torch.save(checkpoint, save_path)
    logger.info(f"Model saved to {save_path}")


def train_demo():
    # Hyperparameters
    batch_size = 1024
    cfg = {
        "d_in": 768,
        "latent": 768 * 4,
        "k_active": 32,
        "k_aux": 512,
        "dead_window": 50_000,
        "batch_size": batch_size,
    }
    wandb_cfg = {
        "use_wandb": True,  # Set to True to enable wandb logging
        "wandb_project": "walrus-workshop-demo",
        "wandb_run_name": None,  # None will auto-generate a name
    }

    # 1. Generate Dummy Data (N numpy arrays)
    # Simulating 5 arrays, each with 10k samples of dim 768
    num_arrays = 5
    samples_per_array = 10_000
    numpy_arrays = [
        np.random.randn(samples_per_array, cfg.get("d_in", 768)).astype(np.float32)
        for _ in range(num_arrays)
    ]

    # Create dataset and dataloader (small data, OK to load in memory)
    dataset = NumpyListDataset(numpy_arrays)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 2. Initialize Model
    model = SAE(
        d_in=cfg.get("d_in", 768),
        latent=cfg.get("latent", 768 * 4),
        k_active=cfg.get("k_active", 32),
        k_aux=cfg.get("k_aux", 512),
        dead_window=cfg.get("dead_window", 50_000),
    )

    # 3. Train
    device = "cuda" if torch.cuda.is_available() else "cpu"
    trained_model = train_sae(
        model,
        dataloader,
        total_samples=len(dataset),
        batches_per_epoch=len(dataloader),
        lr=3e-4,
        epochs=3,
        device=device,
        wandb_cfg=wandb_cfg,
        sae_cfg=cfg,
    )
    return trained_model, cfg


def train_walrus(layer_name: str, num_arrays: int | None = 10, num_workers: int = 4):
    save_dir = os.path.abspath(f"./activations/{layer_name}")
    act_files = sorted(glob.glob(os.path.join(save_dir, "*.npy")))
    act_shape = np.load(act_files[0], mmap_mode="r").shape

    # Split into train/test using reproducible split
    train_files, _ = split_test_train(act_files, random_state=42, test_size=0.2)

    # Limit to num_arrays if specified
    if num_arrays is not None:
        train_files = train_files[:num_arrays]
    else:
        num_arrays = len(train_files)

    batch_size = 1024

    # Setup lazy-loading dataset (memory-efficient)
    logger.info(f"Setting up lazy loading for {len(train_files)} activation files")
    dataset = LazyNumpyDataset(
        train_files, d_in=act_shape[1], batch_size=batch_size, seed=42
    )
    # batch_size=None because LazyNumpyDataset already yields batches
    dataloader = DataLoader(dataset, num_workers=num_workers, batch_size=None)

    cfg = {
        "d_in": act_shape[1],
        "latent": act_shape[1] * 32,  # d_in x expansion factor
        "k_active": 32,
        "k_aux": 512,
        "batch_size": batch_size,
    }

    wandb_cfg = {
        "use_wandb": True,  # Set to True to enable wandb logging
        "wandb_project": f"walrus-workshop-{layer_name}",
        "wandb_run_name": f"num_arrays={num_arrays}, k_active={cfg.get('k_active', 32)}, k_aux={cfg.get('k_aux', 512)}, latent={cfg.get('latent', 768 * 4)}",
    }

    # Initialize Model
    model = SAE(
        d_in=cfg.get("d_in", 768),
        latent=cfg.get("latent", 768 * 4),
        k_active=cfg.get("k_active", 32),
        k_aux=cfg.get("k_aux", 512),
        dead_window=cfg.get("dead_window", 50_000),
    )

    # Train
    device = "cuda" if torch.cuda.is_available() else "cpu"
    trained_model = train_sae(
        model,
        dataloader,
        total_samples=dataset.total_samples,
        batches_per_epoch=dataset.total_batches,
        lr=3e-4,
        epochs=5,
        device=device,
        wandb_cfg=wandb_cfg,
        sae_cfg=cfg,
    )
    return trained_model, cfg


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    num_arrays = 150

    layer_name = "blocks.20.space_mixing.activation"
    trained_model, cfg = train_walrus(layer_name, num_arrays=None)
    save_sae(
        save_path=f"./checkpoints/sae_checkpoint_{layer_name}_num{num_arrays}.pt",
        cfg=cfg,
        model=trained_model,
    )
