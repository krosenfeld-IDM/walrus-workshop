"""
Look at the top features during training and their most activating examples
"""
import pickle
import logging
import torch
import os
import heapq
import hydra
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from alive_progress import alive_it
from dataclasses import dataclass

from walrus_workshop.model import load_sae
from walrus_workshop.activation import ActivationsDataSet


# Setup logger with handler to output to terminal
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

this_dir = Path(__file__).parent
config_path = this_dir / "configs" / "top_features.yaml"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class Exemplar:
    """Single exemplar with its activation, position in batch, and batch ID."""
    activation: float
    index: int
    batch_id: int
    
    def __lt__(self, other):
        # For min-heap: lower activation = lower priority
        return self.activation < other.activation


def collect_exemplars(
    sae_model,
    dataloader,
    top_activations,
    num_exemplars: int,
    device: str
) -> dict[int, dict[str, np.ndarray]]:
    """
    Collect top-k exemplars for each feature based on activation strength.
    
    Uses a min-heap per feature to efficiently track the top activations
    across all batches without storing everything in memory.
    
    Args:
        sae_model: The sparse autoencoder model
        dataloader: DataLoader yielding batches of tokens
        top_activations: Object with .indices attribute containing feature indices
        num_exemplars: Number of top exemplars to keep per feature
        device: Device to run computations on
        
    Returns:
        Dictionary mapping feature_idx -> {activations, indices, batch_ids}
    """
    feature_indices = top_activations.indices.cpu().numpy()
    
    # Min-heap per feature (stores Exemplar objects)
    heaps: dict[int, list[Exemplar]] = {idx: [] for idx in feature_indices}
    
    for batch_idx, x in enumerate(
        alive_it(dataloader, total=dataloader.dataset.total_batches)
    ):
        x = x.to(device)
        
        # Forward pass
        recon, code, aux_recon = sae_model(x)
        
        # Get activations for all tracked features at once: (B, num_features)
        batch_activations = code[:, feature_indices].cpu().numpy()
        
        for i, feature_idx in enumerate(feature_indices):
            feature_acts = batch_activations[:, i]
            
            # Find top-k candidates in this batch using argpartition (O(n) vs O(n log n))
            if len(feature_acts) <= num_exemplars:
                top_k_batch_indices = np.arange(len(feature_acts))
            else:
                top_k_batch_indices = np.argpartition(
                    feature_acts, -num_exemplars
                )[-num_exemplars:]
            
            heap = heaps[feature_idx]
            
            for batch_pos in top_k_batch_indices:
                activation = feature_acts[batch_pos]
                exemplar = Exemplar(activation, int(batch_pos), batch_idx)
                
                if len(heap) < num_exemplars:
                    heapq.heappush(heap, exemplar)
                elif activation > heap[0].activation:
                    heapq.heapreplace(heap, exemplar)
    
    # Convert heaps to final format (sorted descending by activation)
    exemplars = {}
    for feature_idx in feature_indices:
        sorted_exemplars = sorted(heaps[feature_idx], reverse=True)
        exemplars[feature_idx] = {
            "activations": np.array([e.activation for e in sorted_exemplars]),
            "indices": np.array([e.index for e in sorted_exemplars]),
            "batch_ids": np.array([e.batch_id for e in sorted_exemplars]),
        }
    
    return exemplars

@hydra.main(config_path="configs", config_name="train.yaml")
def main(cfg):
    # Load the trained SAE
    checkpoint_path = (
        this_dir
        / "checkpoints"
        / "sae_checkpoint_blocks.20.space_mixing.activation_source_test_k_active=32_k_aux=2048_latent=22528_beta=0.1.pt"
    )
    # "this_dir", "checkpoints", "sae_checkpoint_blocks.20.space_mixing.activation_source_test.pt"
    # )
    sae_model, sae_config = load_sae(checkpoint_path)
    sae_model = sae_model.to(device).eval()

    # Get the number of features from the SAE sae_config
    top_activations = torch.topk(
        sae_model.activation_counts, k=cfg.analysis.features.top_k
    )  # values and indices

    # setup dataset
    # Split into train/test using reproducible split=
    datasets = ActivationsDataSet(
        name=cfg.walrus.dataset,
        layer_name=cfg.walrus.layer_name,
        split=cfg.training.split,
        seed=cfg.walrus.random_state,
        source_split=cfg.training.source_split,
        activations_base_path=this_dir / "activations",
    )
    dataloader = datasets.to_dataloader(
        batch_size=cfg.training.batch_size, num_workers=cfg.training.num_workers
    )

    num_exemplars = cfg.analysis.features.num_exemplars_per_feature
    exemplars = collect_exemplars(sae_model, dataloader, top_activations, num_exemplars, "cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(this_dir / "exemplars", exist_ok=True)
    with open(this_dir / "exemplars" / "exemplars.pkl", "wb") as f:
        pickle.dump(exemplars, f)

    print("pause")

if __name__ == "__main__":
    main()
