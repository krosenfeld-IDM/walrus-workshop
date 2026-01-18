import matplotlib.pyplot as plt
import glob
import os
import polars as pl
import re
from alive_progress import alive_it

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Plot and compare the enstrophies for the differente well simulations
file_list = glob.glob(os.path.join("metrics", "enstrophy", "*.csv"))

save_dir = os.path.join("figures", "enstrophy", "shear_flow")
os.makedirs(save_dir, exist_ok=True)

for file in alive_it(file_list):
    df = pl.read_csv(file)
    # Parse the filename for the Reynolds and Schmidt number (shear_flow_N_ReynoldsA_SchmidtB.csv)
    filename = os.path.basename(file)
    filename_without_ext = os.path.splitext(filename)[0]
    match = re.search(r'Reynolds([\d.e+-]+)_Schmidt([\d.e+-]+)', filename_without_ext)
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