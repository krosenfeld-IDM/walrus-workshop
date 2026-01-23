"""
Calculate the Maximum Activating Exemplars (MAEs)
"""

import torch
from walrus_workshop.model import load_sae
from alive_progress import alive_it
import numpy as np
import os
import zarr
import pickle
import gzip
import logging

from walrus_workshop.utils import load_config
from walrus_workshop.activation import ActivationsDataSet

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Set up logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def run_maes(
    layer_name: str,
    num_exemplars_per_feature: int,
    activation_threshold: float,
    data_id: str = "shear_flow",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load activation files
    # save_dir = os.path.abspath(f"./activations/{layer_name}")
    # act_files = sorted(glob.glob(os.path.join(save_dir, "*.npy")))

    datasets = ActivationsDataSet(
        name=data_id,
        layer_name=layer_name,
        split=None,
        source_split="test",
    )
    files = datasets.data

    # Load the trained SAE
    checkpoint_path = os.path.join(
        "checkpoints", "sae_checkpoint_blocks.20.space_mixing.activation_source_test.pt"
    )
    sae_model, sae_config = load_sae(checkpoint_path)
    sae_model = sae_model.to(device).eval()

    # Get the number of features from the SAE sae_config
    n_features = sae_config["latent"]

    # Store the top features
    top_features = {}

    logger.info(f"Processing {len(files)} test files to find MAEs...")
    for file_idx, file in enumerate(alive_it(files)):
        # Load the activations
        act =  zarr.open(file, mode="r")

        # Move to device
        xb = torch.from_numpy(np.array(act)).to(device)

        # Forward pass
        with torch.no_grad():
            _, code, _ = sae_model(xb)

        # Loop over features in the SAE nework
        for feature_idx in range(n_features):
            # Get activations for this feature across all nodes
            feature = code[:, feature_idx].cpu().numpy()  # [B]

            mask = feature > activation_threshold
            if not mask.any():
                # next_idx += feature.shape[0]
                continue

            if feature_idx not in top_features:
                top_features[feature_idx] = {
                    "top_scores": [],
                    "top_indices": [],
                    "file": [],
                }

            candidate_scores = feature[mask]

            # Map the masked position back to the activation indices
            batch_positions = np.nonzero(mask)[0].astype(np.int64)

            # Merge candidates into running top-k
            for score, idx in zip(candidate_scores.tolist(), batch_positions.tolist()):
                if (
                    len(top_features[feature_idx]["top_scores"])
                    < num_exemplars_per_feature
                ):
                    top_features[feature_idx]["top_scores"].append(score)
                    top_features[feature_idx]["top_indices"].append(idx)
                    top_features[feature_idx]["file"].append(file)
                else:
                    # Replace current min if this is larger
                    min_i = int(np.argmin(top_features[feature_idx]["top_scores"]))
                    if score > top_features[feature_idx]["top_scores"][min_i]:
                        top_features[feature_idx]["top_scores"][min_i] = score
                        top_features[feature_idx]["top_indices"][min_i] = idx
                        top_features[feature_idx]["file"][min_i] = file

        # Free GPU memory
        del xb, code
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Save top_features to disk efficiently
    output_dir = os.path.abspath(f"./maes/{layer_name}")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(
        output_dir,
        f"top_features_{layer_name}_num{num_arrays}_k{num_exemplars_per_feature}.pkl.gz",
    )

    logger.info(f"Saving top_features to {output_path}...")
    with gzip.open(output_path, "wb") as f:
        pickle.dump(top_features, f, protocol=pickle.HIGHEST_PROTOCOL)

    logger.info(f"Saved {len(top_features)} features to {output_path}")


if __name__ == "__main__":
    layer_name = "blocks.20.space_mixing.activation"
    num_arrays = 180
    num_exemplars_per_feature = 20  # Number of top exemplars to save per feature
    activation_threshold = 0.0  # Threshold for SAE feature activation

    run_maes(layer_name, num_exemplars_per_feature, activation_threshold)  # $$
