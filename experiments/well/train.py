"""
Train TopK SAE on activations
"""

import os
import glob
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from alive_progress import alive_it
import wandb
import yaml

from walrus_workshop.model import SAE
from walrus_workshop.data import split_test_train, NumpyListDataset, LazyNumpyDataset

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


def load_config(config_path: str | Path | None = None) -> dict:
    """
    Load training configuration from YAML file.
    
    Args:
        config_path: Path to config file. If None, looks for train_config.yml
                     in the same directory as this script.
    
    Returns:
        Dictionary containing configuration
    """
    if config_path is None:
        script_dir = Path(__file__).parent
        config_path = script_dir / "train_config.yml"
    else:
        config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info(f"Loaded config from {config_path}")
    return config


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


def train_demo(config_path: str | Path | None = None):
    # Load configuration from YAML
    config = load_config(config_path)
    
    # Extract configuration sections
    model_cfg = config.get("model", {})
    training_cfg = config.get("training", {})
    wandb_cfg_dict = config.get("wandb", {})
    demo_cfg = config.get("demo", {})
    
    # Build cfg dictionary for model
    batch_size = training_cfg.get("batch_size", 1024)
    d_in = model_cfg.get("d_in", 768)
    latent = model_cfg.get("latent", d_in * 4)
    
    cfg = {
        "d_in": d_in,
        "latent": latent,
        "k_active": model_cfg.get("k_active", 32),
        "k_aux": model_cfg.get("k_aux", 512),
        "dead_window": model_cfg.get("dead_window", 50_000),
        "batch_size": batch_size,
    }
    
    # Build wandb_cfg dictionary
    wandb_cfg = {
        "use_wandb": wandb_cfg_dict.get("use_wandb", True),
        "wandb_project": wandb_cfg_dict.get("wandb_project", "walrus-workshop-demo"),
        "wandb_run_name": wandb_cfg_dict.get("wandb_run_name", None),
    }

    # 1. Generate Dummy Data (N numpy arrays)
    num_arrays = demo_cfg.get("num_arrays", 5)
    samples_per_array = demo_cfg.get("samples_per_array", 10_000)
    numpy_arrays = [
        np.random.randn(samples_per_array, d_in).astype(np.float32)
        for _ in range(num_arrays)
    ]

    # Create dataset and dataloader (small data, OK to load in memory)
    dataset = NumpyListDataset(numpy_arrays)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 2. Initialize Model
    model = SAE(
        d_in=cfg["d_in"],
        latent=cfg["latent"],
        k_active=cfg["k_active"],
        k_aux=cfg["k_aux"],
        dead_window=cfg["dead_window"],
    )

    # 3. Train
    device_str = training_cfg.get("device", "cuda")
    device = device_str if device_str == "cpu" else ("cuda" if torch.cuda.is_available() else "cpu")
    lr = training_cfg.get("learning_rate", 3e-4)
    epochs = demo_cfg.get("epochs", training_cfg.get("epochs", 3))
    
    trained_model = train_sae(
        model,
        dataloader,
        total_samples=len(dataset),
        batches_per_epoch=len(dataloader),
        lr=lr,
        epochs=epochs,
        device=device,
        wandb_cfg=wandb_cfg,
        sae_cfg=cfg,
    )
    return trained_model, cfg


def train_walrus(
    layer_name: str,
    num_arrays: int | None = None,
    num_workers: int | None = None,
    save: bool = False,
    config_path: str | Path | None = None,
):
    # Load configuration from YAML
    config = load_config(config_path)
    
    # Extract configuration sections
    model_cfg = config.get("model", {})
    training_cfg = config.get("training", {})
    wandb_cfg_dict = config.get("wandb", {})
    walrus_cfg = config.get("walrus", {})
    
    # Override num_arrays and num_workers from config if not provided
    if num_arrays is None:
        num_arrays = walrus_cfg.get("num_arrays", None)
    if num_workers is None:
        num_workers = walrus_cfg.get("num_workers", 4)
    
    save_dir = os.path.abspath(f"./activations/{layer_name}")
    act_files = sorted(glob.glob(os.path.join(save_dir, "*.npy")))
    act_shape = np.load(act_files[0], mmap_mode="r").shape

    # Split into train/test using reproducible split
    random_state = walrus_cfg.get("random_state", 42)
    train_files, _ = split_test_train(act_files, random_state=random_state)

    # Limit to num_arrays if specified
    if num_arrays is not None:
        train_files = train_files[:num_arrays]
    else:
        num_arrays = len(train_files)

    batch_size = training_cfg.get("batch_size", 1024)
    d_in = act_shape[1]  # Dynamic: determined from data
    expansion_factor = model_cfg.get("expansion_factor", 32)
    latent = d_in * expansion_factor  # Dynamic: d_in * expansion_factor

    # Setup lazy-loading dataset (memory-efficient)
    logger.info(f"Setting up lazy loading for {len(train_files)} activation files")
    dataset = LazyNumpyDataset(
        train_files, d_in=d_in, batch_size=batch_size, seed=random_state
    )
    # batch_size=None because LazyNumpyDataset already yields batches
    dataloader = DataLoader(dataset, num_workers=num_workers, batch_size=None)

    cfg = {
        "d_in": d_in,
        "latent": latent,
        "k_active": model_cfg.get("k_active", 32),
        "k_aux": model_cfg.get("k_aux", 512),
        "dead_window": model_cfg.get("dead_window", 50_000),
        "batch_size": batch_size,
    }

    # Build wandb_cfg with dynamic layer_name
    wandb_project_base = wandb_cfg_dict.get("wandb_project", "walrus-workshop")
    wandb_project = f"{wandb_project_base}-{layer_name}"
    wandb_run_name = wandb_cfg_dict.get("wandb_run_name", None)
    if wandb_run_name is None:
        wandb_run_name = (
            f"num_arrays={num_arrays}, k_active={cfg['k_active']}, "
            f"k_aux={cfg['k_aux']}, latent={cfg['latent']}"
        )
    
    wandb_cfg = {
        "use_wandb": wandb_cfg_dict.get("use_wandb", True),
        "wandb_project": wandb_project,
        "wandb_run_name": wandb_run_name,
    }

    # Initialize Model
    model = SAE(
        d_in=cfg["d_in"],
        latent=cfg["latent"],
        k_active=cfg["k_active"],
        k_aux=cfg["k_aux"],
        dead_window=cfg["dead_window"],
    )

    # Train
    device_str = training_cfg.get("device", "cuda")
    device = device_str if device_str == "cpu" else ("cuda" if torch.cuda.is_available() else "cpu")
    lr = training_cfg.get("learning_rate", 3e-4)
    epochs = walrus_cfg.get("epochs", training_cfg.get("epochs", 5))
    
    trained_model = train_sae(
        model,
        dataloader,
        total_samples=dataset.total_samples,
        batches_per_epoch=dataset.total_batches,
        lr=lr,
        epochs=epochs,
        device=device,
        wandb_cfg=wandb_cfg,
        sae_cfg=cfg,
    )

    if save:
        save_sae(
            save_path=f"./checkpoints/sae_checkpoint_{layer_name}_num{num_arrays}.pt",
            cfg=cfg,
            model=trained_model,
        )
    return trained_model, cfg


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    for layer_number in [21, 30]:
        layer_name = f"blocks.{layer_number}.space_mixing.activation"
        trained_model, cfg = train_walrus(layer_name, num_arrays=None, save=True)
