from walrus_workshop.walrus import get_trajectory
import time
import matplotlib.pyplot as plt
import numpy as np
from einops import rearrange
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))


trajectory_id = 4 # 700 does not work
B = 0 # But not actually the batch size...
split = "test"

start_time = time.time()
window, metadata = get_trajectory("shear_flow", trajectory_id=trajectory_id, split="test")
stop_time = time.time()
print(f"Data loaded in {stop_time - start_time} seconds")

F = metadata.n_fields # Note that this has "extra" fields like velocity_z. Use the padded_field_mask to remove.

fig, axs = plt.subplots(sum(window['padded_field_mask']).item(), 4, figsize=(4 * 2.4, sum(window['padded_field_mask']) * 1.2))

x = window["input_fields"] # B T Lx Ly Lz F
x = rearrange(x, "B T ... Lz F -> B F Lz T ...") # B F Lz T Lx Ly
for t, T in enumerate([0,1, 198, 199]):
    for field in range(sum(window['padded_field_mask'])):
        vmin = np.nanmin(x[B, field])
        vmax = np.nanmax(x[B, field])
        axs[field, t].imshow(x[B, field, 0, T], cmap="RdBu_r", interpolation="none", vmin=vmin, vmax=vmax)
        axs[field, t].set_xticks([])
        axs[field, t].set_yticks([])

        axs[0, t].set_title(f"${T}$")
plt.tight_layout()
plt.savefig("figures/trajectory.png")


