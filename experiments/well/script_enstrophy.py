import matplotlib.pyplot as plt
import glob
import os
import polars as pl
import re
from alive_progress import alive_it
from omegaconf import OmegaConf
from walrus_workshop.metrics import compute_enstrophy
from walrus_workshop.walrus import get_trajectory, TRAJECTORY_CONFIG

os.chdir(os.path.dirname(os.path.abspath(__file__)))


def plot_entrophy():
    # Plot and compare the enstrophies for the differente well simulations
    file_list = glob.glob(os.path.join("metrics", "enstrophy", "*.csv"))

    save_dir = os.path.join("figures", "enstrophy", "shear_flow")
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


if __name__ == "__main__":
    for i in alive_it(range(112)):
        calc_enstrophy(dataset_id="shear_flow", trajectory_id=i)
    # plot_entrophy()