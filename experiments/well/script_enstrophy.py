import glob
import os
import re
import zarr
import argparse
import logging
import torch
import pickle
import polars as pl
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from alive_progress import alive_it
from omegaconf import OmegaConf

from walrus_workshop import paths
from walrus_workshop.model import load_sae
from walrus_workshop.metrics import compute_enstrophy
from walrus_workshop.walrus import get_trajectory, TRAJECTORY_CONFIG
from walrus_workshop.utils import get_keyvalue_from_string, load_config
from walrus_workshop.activation import ActivationsDataSet

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Set up logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def plot_entrophy():
    # Plot and compare the enstrophies for the differente well simulations
    file_list = glob.glob(os.path.join("metrics", "enstrophy", "*.csv"))

    save_dir = os.path.join("figures", "shear_flow", "enstrophy")
    os.makedirs(save_dir, exist_ok=True)

    for file in alive_it(file_list):
        df = pl.read_csv(file)
        # Parse the filename for the Reynolds and Schmidt number (shear_flow_N_ReynoldsA_SchmidtB.csv)
        filename = os.path.basename(file)
        filename_without_ext = os.path.splitext(filename)[0]
        match = re.search(
            r"Reynolds([\d.e+-]+)_Schmidt([\d.e+-]+)", filename_without_ext
        )
        if match:
            reynolds = float(match.group(1))
            schmidt = float(match.group(2))
            title = f"Reynolds={reynolds:.2e}, Schmidt={schmidt:.2e}"
        else:
            title = filename

        plt.figure(figsize=(10, 5))
        plt.plot(df["enstrophy_ref"], label="Reference")
        plt.plot(df["enstrophy_pred"], label="Prediction")
        plt.title(title)

        plt.legend()
        plt.savefig(os.path.join(save_dir, f"{os.path.splitext(filename)[0]}.png"))
        plt.close()


def calc_enstrophy(dataset_id: str = "shear_flow", trajectory_id: int = 0):
    """Calculate enstrophy for each time step of a trajectory"""

    config = OmegaConf.create(TRAJECTORY_CONFIG)
    trajectory, metadata = get_trajectory(
        dataset_id=dataset_id, trajectory_id=trajectory_id, config=config, split="test"
    )
    y_ref = trajectory["input_fields"][0][..., trajectory["padded_field_mask"]]
    enstrophy = []
    for i in range(y_ref.shape[0]):
        enstrophy.append(
            compute_enstrophy(
                y_ref[i, :, :, 0, 2].cpu().numpy(), y_ref[i, :, :, 0, 3].cpu().numpy()
            )[0]
        )

    output_dir = os.path.join("metrics", "enstrophy", dataset_id)
    os.makedirs(output_dir, exist_ok=True)

    trajectory_name = (
        metadata.dataset_name
        + f"_{trajectory_id}_"
        + "_".join(
            [
                f"{k}_{v.item():0.0e}"
                for k, v in zip(
                    metadata.constant_scalar_names,
                    trajectory["constant_scalars"][0],
                )
            ]
        )
    )
    # Save the two lists to a csv file using polars
    pl.DataFrame({"enstrophy": enstrophy}).write_csv(
        os.path.join(output_dir, f"{trajectory_name}.csv")
    )


def load_enstrophy_df(data_id: str = "shear_flow"):
    """Load the enstrophy dataframe"""

    file_list = glob.glob(os.path.join("metrics", "enstrophy", data_id, "*.csv"))
    file_list = sorted(file_list)

    df = None
    for file in file_list:
        df_ = pl.read_csv(file)
        df_ = df_.with_columns(pl.col("enstrophy").diff().alias("derivative"))
        df_ = df_.with_columns(pl.col("derivative").abs().alias("abs_derivative"))

        file_stem = Path(file).stem
        file_parts = file_stem.split("_")
        for i in range(2, len(file_parts) - 1):
            part = file_parts[i] + file_parts[i + 1]
            # for part in file_parts:
            # if match := re.match(r'([a-zA-Z]+)([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)', part):
            if match := get_keyvalue_from_string(part):
                df_ = df_.with_columns(
                    pl.lit(match.groups()[1], dtype=float).alias(match.groups()[0])
                )

        # Extract the integer between "shear_flow_" and "_Reynolds"
        match = re.search(rf"{data_id}_(\d+)_", file)
        if match:
            df_ = df_.with_columns(pl.lit(int(match.groups()[0])).alias("id"))

        # add a row index (used as the time step)
        df_ = df_.with_row_index(name="step")

        # add filename
        df_ = df_.with_columns(pl.lit(file).alias("filename"))

        # add an index column
        if df is None:
            df = df_
        else:
            df = pl.concat([df, df_])
    return df


def run_sae_top_enstrophy(data_id: str = "shear_flow", epoch: int = 4, num_top_enstrophy: int = 20, num_top_features: int = 20, split="test"):
    """Identify the trajectories / numerical simulations that contain the largest enstrophy values and run the SAE on them"""

    layer_name = "blocks.20.space_mixing.activation"
    constant_scalar_names = ['Reynolds', 'Schmidt']
    activations_dir = Path("activations") / split / layer_name / data_id

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the trained SAE
    checkpoint_path = os.path.join("checkpoints", "sae_checkpoint_blocks.20.space_mixing.activation_source_test.pt")
    logger.debug(f"Loading SAE model from {checkpoint_path}")
    sae_model, config = load_sae(
        checkpoint_path
    )
    sae_model = sae_model.to(device).eval()
    
    # Get the number of features from the SAE config
    n_features = config["latent"]

    logger.debug(f"Loading enstrophy dataframe for {data_id}")
    df = load_enstrophy_df(data_id="shear_flow")
    df = df.group_by(["id", "filename"] + constant_scalar_names).agg(pl.col("abs_derivative").median().alias("median_abs_derivative")).sort("median_abs_derivative", descending=True)[:num_top_enstrophy]
    logger.debug(f"Running SAE on {len(df)} trajectories ({df['id'].sort().to_list()})")

    # loop over trajctories
    for row in alive_it(df.iter_rows(named=True), total=len(df)):
        act_files = sorted(glob.glob(str(activations_dir / f"*_traj_{row["id"]}*")))
        # results = {}
        for file_idx, file in enumerate(act_files):
            act = zarr.open(file, mode="r")

            # Move to device
            xb = torch.from_numpy(np.array(act)).to(device)

            # Forward pass
            logger.debug(f"forward pass for file {file}")
            with torch.no_grad():
                _, code, _ = sae_model(xb)
            logger.debug(f"processed with {code.shape[0]} nodes and {code.shape[1]} features")

            # Loop over features in the SAE nework
            features = [0] * n_features
            for feature_idx in range(n_features):
                # Get activations for this feature across all nodes
                # top_features = code[:, feature_idx].sort(descending=True)[:top_features].cpu().numpy()  
                (values, top_features) = torch.sort(code[:, feature_idx], descending=True)                
                
                # save
                features[feature_idx] = (values.cpu().numpy()[:num_top_features], top_features.cpu().numpy()[:num_top_features])

            # results[file] = features
            output_path = os.path.join("sae", "top_enstrophy", f"top_enstrophy_traj_{row['id']}_file_{file_idx}.pkl")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            logger.debug(f"Saving results to {output_path}")
            with open(output_path, "wb") as f:
                pickle.dump({'features':features, 'file':file}, f)

            # Free GPU memory
            del xb, code, features
            if device.type == "cuda":
                torch.cuda.empty_cache()






if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--calculate_enstrophy", action="store_true", default=False)
    parser.add_argument("--plot_enstrophy", action="store_true", default=False)
    parser.add_argument("--run_sae_top_enstrophy", action="store_true", default=True)
    args = parser.parse_args()

    if args.calculate_enstrophy:
        logger.info("Calculating enstrophy")
        for i in alive_it(range(112)):
            calc_enstrophy(dataset_id="shear_flow", trajectory_id=i)

    if args.plot_enstrophy:
        logger.info("Plotting enstrophy")
        plot_entrophy()

    if args.run_sae_top_enstrophy:
        logger.info("Running SAE on top enstrophy trajectories")
        run_sae_top_enstrophy(data_id="shear_flow", epoch=4)
