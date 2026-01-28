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

from walrus_workshop.utils import load_config
from walrus_workshop.model import SAE
from walrus_workshop.data import LazyZarrDataset
from walrus_workshop.activation import ActivationsDataSet

# Setup logger with handler to output to terminal
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create console handler if one doesn't exist
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def resample_dead_features(sae_model, residuals, resample_fraction=0.01):
    """
    Reinitialize a fraction of dead features using residual directions.

    Args:
        sae_model: The SAE model
        residuals: Recent residuals tensor [N, d_in] to sample from
        resample_fraction: Fraction of dead features to resample (default 1%)

    Returns:
        Number of features resampled
    """
    dead_mask = sae_model.dead_mask
    if not dead_mask.any() or residuals.shape[0] == 0:
        return 0

    dead_indices = torch.where(dead_mask)[0]
    num_dead = len(dead_indices)
    num_to_resample = max(1, int(resample_fraction * num_dead))

    # Randomly select which dead features to reinit
    perm = torch.randperm(num_dead, device=dead_indices.device)
    resample_indices = dead_indices[perm[:num_to_resample]]

    # Sample residual directions
    n_residuals = residuals.shape[0]
    sampled_idx = torch.randint(0, n_residuals, (num_to_resample,), device=residuals.device)
    new_directions = residuals[sampled_idx].clone()

    # Normalize to unit norm
    norms = new_directions.norm(dim=1, keepdim=True).clamp_min(1e-8)
    new_directions = new_directions / norms

    # Update decoder columns (dec.weight shape: [d_in, latent])
    with torch.no_grad():
        sae_model.dec.weight.data[:, resample_indices] = new_directions.T
        # Reset encoder rows to match (enc.weight shape: [latent, d_in])
        sae_model.enc.weight.data[resample_indices, :] = new_directions
        # Reset miss_counts for resampled features
        sae_model.miss_counts[resample_indices] = 0
        sae_model.dead_mask[resample_indices] = False

    return num_to_resample


def train_sae(
    sae_model,
    dataloader,
    total_samples,
    batches_per_epoch,
    lr=3e-4,
    epochs=10,
    device="cuda",
    wandb_cfg=None,
    save_every=None,
    checkpoint_dir=None,
    checkpoint_prefix=None,
    usage_histogram_window=1000,
    resample_dead_interval_factor=0.5,
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
        save_every: Save checkpoint every N batches (None to disable)
        checkpoint_dir: Directory for checkpoint files
        checkpoint_prefix: Prefix for checkpoint filenames
        usage_histogram_window: Number of steps to accumulate feature usage before logging
        resample_dead_interval_factor: Resample dead features every (factor * dead_window) samples
    """
    sae_cfg = sae_model.get_config()

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

    # Learning rate warmup + cosine decay
    total_steps = epochs * batches_per_epoch
    warmup_steps = int(0.05 * total_steps)  # 5% warmup

    warmup_scheduler = optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda step: min(1.0, (step + 1) / warmup_steps) if warmup_steps > 0 else 1.0
    )
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, total_steps - warmup_steps)
    )
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer, [warmup_scheduler, cosine_scheduler], milestones=[warmup_steps]
    )

    # --- Feature usage histogram tracking ---
    feature_usage_counts = torch.zeros(sae_model.latent, dtype=torch.long, device=device)
    usage_steps_accumulated = 0

    # --- Dead feature resampling tracking ---
    dead_window_samples = sae_cfg.get("dead_window", 500_000)
    resample_interval = int(resample_dead_interval_factor * dead_window_samples)
    samples_since_resample = 0
    # Buffer for recent residuals (for resampling initialization)
    max_residual_buffer = 4096  # Store up to 4096 recent residuals
    residual_buffer = None
    residual_buffer_idx = 0

    print(f"Starting training on {total_samples} samples...")
    print(f"Dead feature resampling every {resample_interval} samples")

    for epoch in range(epochs):
        sae_model.train()
        total_loss = 0
        total_mse = 0
        total_aux = 0

        for batch_idx, x in enumerate(alive_it(dataloader, total=batches_per_epoch)):
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
            torch.nn.utils.clip_grad_norm_(sae_model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            # --- Post-Optimization Constraints ---
            with torch.no_grad():
                # 1. Enforce Unit Norm Decoder columns
                sae_model._renorm_decoder_columns_()

                # 2. Update Dead Neuron Statistics
                sae_model.update_dead_mask(code, batch_size=x.shape[0])

                # 3. Track feature usage (which features are in top-k)
                active_features = (code > 0)  # [batch, latent]
                feature_usage_counts += active_features.sum(dim=0).long()
                usage_steps_accumulated += 1

                # 4. Store residuals for dead feature resampling
                if residual_buffer is None:
                    residual_buffer = torch.zeros(max_residual_buffer, x.shape[1], device=device)
                # Store a subset of residuals from this batch (vectorized)
                n_to_store = min(x.shape[0], max_residual_buffer // 4)
                store_indices = torch.randperm(x.shape[0], device=device)[:n_to_store]
                sampled_residuals = residual[store_indices]
                # Compute buffer indices for circular buffer
                buffer_indices = torch.arange(n_to_store, device=device)
                buffer_indices = (residual_buffer_idx + buffer_indices) % max_residual_buffer
                residual_buffer[buffer_indices] = sampled_residuals
                residual_buffer_idx = (residual_buffer_idx + n_to_store) % max_residual_buffer

            # --- Track samples for dead feature resampling ---
            samples_since_resample += x.shape[0]

            # --- Periodic dead feature resampling ---
            if samples_since_resample >= resample_interval:
                num_resampled = resample_dead_features(
                    sae_model,
                    residual_buffer if residual_buffer is not None else torch.zeros(1, sae_model.d_in, device=device),
                    resample_fraction=0.01
                )
                if num_resampled > 0:
                    logger.info(f"Resampled {num_resampled} dead features at step {epoch * batches_per_epoch + batch_idx}")
                    if use_wandb:
                        wandb.log({
                            "train/features_resampled": num_resampled,
                            "global_step": epoch * batches_per_epoch + batch_idx,
                        })
                samples_since_resample = 0

            # Logging
            total_loss += loss.item()
            total_mse += mse_loss.item()
            total_aux += aux_loss.item()

            # Wandb logging (every 100 batches to avoid too much logging)
            if use_wandb and batch_idx % 100 == 0:
                current_lr = scheduler.get_last_lr()[0]
                log_dict = {
                    "train/loss": loss.item(),
                    "train/mse_loss": mse_loss.item(),
                    "train/aux_loss": aux_loss.item(),
                    "train/dead_neurons": sae_model.dead_mask.sum().item(),
                    "train/learning_rate": current_lr,
                    "epoch": epoch,
                    "batch": batch_idx,
                    "global_step": epoch * batches_per_epoch + batch_idx,
                }

                # Log usage histogram statistics when window is reached
                if usage_steps_accumulated >= usage_histogram_window:
                    usage_cpu = feature_usage_counts.cpu().numpy()
                    # Log histogram to wandb
                    log_dict["feature_usage/histogram"] = wandb.Histogram(usage_cpu)
                    # Log summary statistics
                    log_dict["feature_usage/mean"] = usage_cpu.mean()
                    log_dict["feature_usage/std"] = usage_cpu.std()
                    log_dict["feature_usage/max"] = usage_cpu.max()
                    log_dict["feature_usage/min"] = usage_cpu.min()
                    log_dict["feature_usage/zero_count"] = (usage_cpu == 0).sum()
                    # Percentiles for distribution understanding
                    log_dict["feature_usage/p50"] = np.percentile(usage_cpu, 50)
                    log_dict["feature_usage/p90"] = np.percentile(usage_cpu, 90)
                    log_dict["feature_usage/p99"] = np.percentile(usage_cpu, 99)
                    # Reset counters
                    feature_usage_counts.zero_()
                    usage_steps_accumulated = 0

                wandb.log(log_dict)

            if batch_idx % 100 == 0:
                logger.info(
                    f"Epoch {epoch + 1} | Batch {batch_idx} | "
                    f"MSE: {mse_loss.item():.4f} | Aux: {aux_loss.item():.4f} | "
                    f"Dead: {sae_model.dead_mask.sum().item()}"
                )

            # Periodic checkpoint saving
            if save_every is not None and checkpoint_dir is not None:
                if save_every == "epoch":
                    if epoch > 0 and batch_idx == 0:
                        os.makedirs(checkpoint_dir, exist_ok=True)
                        save_path = f"{checkpoint_dir}/{checkpoint_prefix}_epoch_{epoch}.pt"
                        save_sae(save_path=save_path, cfg=sae_cfg, model=sae_model)
                else:
                    global_step = epoch * batches_per_epoch + batch_idx
                    if (global_step + 1) % save_every == 0:
                        os.makedirs(checkpoint_dir, exist_ok=True)
                        save_path = f"{checkpoint_dir}/{checkpoint_prefix}_step_{global_step + 1}.pt"
                        save_sae(save_path=save_path, cfg=sae_cfg, model=sae_model)

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


    # Split into train/test using reproducible split=
    datasets = ActivationsDataSet(
        name=walrus_cfg.get("dataset", "shear_flow"),
        layer_name=layer_name,
        split=training_cfg.get("split", "train"),
        seed=walrus_cfg.get("random_state", 42),
        source_split=training_cfg.get("source_split", "test"),
    )
    train_files = datasets.data

    # Override num_arrays and num_workers from config if not provided
    if num_arrays is None:
        num_arrays = walrus_cfg.get("num_arrays", len(train_files))
    if num_workers is None:
        num_workers = walrus_cfg.get("num_workers", 4)    

    # Limit to num_arrays (already set from config or defaults to len(train_files))
    train_files = train_files[:num_arrays]

    batch_size = training_cfg.get("batch_size", 1024)

    # Setup lazy-loading dataset (memory-efficient)
    logger.info(f"Setting up lazy loading for {len(train_files)} activation files")
    dataloader = datasets.to_dataloader(batch_size=batch_size, num_workers=num_workers)

    # Build wandb_cfg with dynamic layer_name
    wandb_project_base = wandb_cfg_dict.get("wandb_project", "walrus-workshop")
    wandb_project = f"{wandb_project_base}-sae-{layer_name}"
    wandb_run_name = wandb_cfg_dict.get("wandb_run_name", None)
    if wandb_run_name is None:
        wandb_run_name = (
            f"num_arrays={num_arrays}, k_active={model_cfg.get('k_active', 128)}, "
            f"k_aux={model_cfg.get('k_aux', 512)}, latent={model_cfg.get('latent', datasets.d_in) * model_cfg.get('expansion_factor', 32)}"
        )

    wandb_cfg = {
        "use_wandb": wandb_cfg_dict.get("use_wandb", True),
        "wandb_project": wandb_project,
        "wandb_run_name": wandb_run_name,
    }

    sae_cfg = dict(
        d_in=datasets.d_in,
        latent=datasets.d_in * model_cfg.get("expansion_factor", 32),
        k_active=model_cfg.get("k_active", 128),
        k_aux=model_cfg.get("k_aux", 512),
        dead_window=model_cfg.get("dead_window", 500_000),
    )
    # Initialize Model
    model = SAE(
        **sae_cfg,
    )

    # Train
    lr = training_cfg.get("learning_rate", 3e-4)
    epochs = walrus_cfg.get("epochs", training_cfg.get("epochs", 5))

    # Periodic checkpoint saving configuration
    save_every = training_cfg.get("save_every", "epoch")
    if save_every == 0:
        save_every = None  # Disable periodic saving
    checkpoint_prefix = f"sae_checkpoint_{layer_name}_source_{training_cfg.get('source_split', 'train')}"

    trained_model = train_sae(
        model,
        dataloader,
        total_samples=dataloader.dataset.total_samples,
        batches_per_epoch=dataloader.dataset.total_batches,
        lr=lr,
        epochs=epochs,
        device=device,
        wandb_cfg=wandb_cfg,
        save_every=save_every,
        checkpoint_dir="./checkpoints",
        checkpoint_prefix=checkpoint_prefix,
    )

    if save:
        os.makedirs("./checkpoints", exist_ok=True)
        save_sae(
            save_path=f"./checkpoints/sae_checkpoint_{layer_name}_source_{training_cfg.get('source_split', 'train')}.pt",
            cfg=trained_model.get_config(),
            model=trained_model,
        )
    return trained_model, trained_model.get_config()


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    for layer_number in [20]:
        layer_name = f"blocks.{layer_number}.space_mixing.activation"
        trained_model, cfg = train_walrus(layer_name, num_arrays=None, save=True)
