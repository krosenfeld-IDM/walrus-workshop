import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
import numpy as np
import os
from the_well.data.utils import flatten_field_names
from the_well.data import WellMetadata
from typing import List
import torch

def add_colorbar(im, aspect=20, pad_fraction=0.5, **kwargs):
    """Add a vertical color bar to an image plot.

    Taken from user Matthias at:
    https://stackoverflow.com/questions/18195758/set-matplotlib-colorbar-size-to-match-graph
    """
    from mpl_toolkits import axes_grid1

    divider = axes_grid1.make_axes_locatable(im.axes)
    width = axes_grid1.axes_size.AxesY(im.axes, aspect=1.0 / aspect)
    pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
    current_ax = plt.gca()
    cax = divider.append_axes("right", size=width, pad=pad)
    plt.sca(current_ax)
    return im.axes.figure.colorbar(im, cax=cax, **kwargs)



def make_video(
    # predicted_images: torch.Tensor,
    true_images: torch.Tensor,
    metadata: WellMetadata,
    output_dir: str,
    epoch_number: int = 0,
    field_name_overrides: List[str] = None,
    size_multiplier: float = 1.0,
):
    """Make a video of the rollout comparison with improved aesthetics and speed.

    Predicted/true are 2/3D channels last tensors.

    Based on the_well.benchmark.metrics.make_video, but with the predicted and error images removed.
    """
    if field_name_overrides is not None:
        field_names = field_name_overrides
    else:
        field_names = flatten_field_names(metadata, include_constants=False)
    dset_name = metadata.dataset_name
    ndims = metadata.n_spatial_dims
    if ndims == 3:
        # Slice the data along the middle of the last axis
        true_images = true_images[..., true_images.shape[-2] // 2, :]
        # predicted_images = predicted_images[..., predicted_images.shape[-2] // 2, :]

    # Grid coordinate handling
    grid_type = metadata.grid_type
    if grid_type == "cartesian":
        coords = ["x", "y", "z"][:ndims]
    elif "spher" in grid_type and ndims == 2:
        coords = ["theta", "phi"]
    elif "spher" in grid_type and ndims == 3:
        coords = ["r", "theta", "phi"]
    else:
        coords = ["x", "y", "z"][:ndims]

    # Calculate the error
    # error_images = (true_images - predicted_images).abs()

    # if isinstance(predicted_images, torch.Tensor):
    #     predicted_images = predicted_images.cpu().numpy()
    if isinstance(true_images, torch.Tensor):
        true_images = true_images.cpu().numpy()
    # if isinstance(error_images, torch.Tensor):
    #     error_images = error_images.cpu().numpy()

    # Calculate percentiles for normalization (vectorized)
    n_fields = len(field_names)
    vmaxes = np.nanpercentile(true_images.reshape(-1, n_fields), 99.9, axis=0)
    vmins = np.nanpercentile(true_images.reshape(-1, n_fields), 0.1, axis=0)
    # emaxes = (vmaxes - vmins) * 0.6666  # Set error max to 2/3 of data range

    h, w = metadata.spatial_resolution[:2]
    aspect_ratio = w / h

    # Improved sizing logic
    base_width_per_field = 3.5
    base_height_per_row = 2.5

    if aspect_ratio > 1.5:  # Wide images
        width_per_field = base_width_per_field * min(aspect_ratio / 1.5, 4.0)
        height_per_row = base_height_per_row
    elif aspect_ratio < 0.67:  # Tall images
        width_per_field = base_width_per_field
        height_per_row = base_height_per_row * min(1.5 / aspect_ratio, 4.0)
    else:  # Roughly square
        width_per_field = base_width_per_field
        height_per_row = base_height_per_row

    fig_width = size_multiplier * (1.5 + width_per_field * n_fields)
    fig_height = size_multiplier * (1.2 + height_per_row * 3)

    # Scale quality settings based on size_multiplier
    # Lower multiplier = faster rendering, lower quality
    # Higher multiplier = slower rendering, higher quality
    if size_multiplier < 0.5:
        dpi = 75
        bitrate = 2000
        preset = "ultrafast"
        interpolation = "nearest"
    elif size_multiplier < 1.0:
        dpi = 100
        bitrate = 5000
        preset = "fast"
        interpolation = "bilinear"
    else:
        dpi = 200
        bitrate = 10000
        preset = "medium"
        interpolation = "lanczos"

    # Setup output path
    write_path = f"{output_dir}/{metadata.dataset_name}/rollout_video"
    os.makedirs(write_path, exist_ok=True)
    output_file = f"{write_path}/epoch{epoch_number}_{dset_name}.mp4"

    n_frames = true_images.shape[0]
    fps = max(5, min(16, int(n_frames / 8)))

    # Better colormap choices
    data_cmap = "viridis"
    # error_cmap = "inferno"

    plt.style.use("dark_background")

    # Pre-create figure once
    fig, axes = plt.subplots(
        1,
        n_fields,
        figsize=(fig_width, fig_height),
        dpi=dpi,
        sharex=True,
        sharey=True,
    )

    # Set layout with proper check for layout engine
    layout_engine = fig.get_layout_engine()
    if layout_engine is not None:
        layout_engine.set(w_pad=0.08, h_pad=0.08, hspace=0.02, wspace=0.25)
    else:
        fig.subplots_adjust(wspace=0.25, hspace=0.02)

    if n_fields == 1:
        axes = axes[:, np.newaxis]

    suptitle = fig.suptitle(
        f"{dset_name}", fontsize=14, fontweight="bold", y=0.98
    )

    # Create all image objects once
    ims = []
    for j, field_name in enumerate(field_names):
        axes[0, j].set_title(field_name, fontsize=11, pad=8, fontweight="semibold")

        # True data
        im = axes[0, j].imshow(
            true_images[0, ..., j],
            cmap=data_cmap,
            vmax=vmaxes[j],
            vmin=vmins[j],
            origin="lower",
            interpolation=interpolation,
            aspect="auto",
        )
        add_colorbar(im, ax=axes[0, j])
        ims.append((im, 0, j))  # Store reference, row, and column

        # # Predicted data
        # im = axes[1, j].imshow(
        #     predicted_images[0, ..., j],
        #     cmap=data_cmap,
        #     vmax=vmaxes[j],
        #     vmin=vmins[j],
        #     origin="lower",
        #     interpolation=interpolation,
        #     aspect="auto",
        # )
        # add_colorbar(im, ax=axes[1, j])
        # ims.append((im, 1, j))

        # # Error data
        # im = axes[2, j].imshow(
        #     error_images[0, ..., j],
        #     cmap=error_cmap,
        #     vmax=emaxes[j],
        #     vmin=0,
        #     origin="lower",
        #     interpolation=interpolation,
        #     aspect="auto",
        # )
        # add_colorbar(im, ax=axes[2, j])
        # ims.append((im, 2, j))

        # axes[2, j].set_xlabel(coords[1], fontsize=10)

        for i in range(1):
            axes[i, j].tick_params(
                axis="both",
                which="both",
                labelsize=8,
                length=2,
                width=0.5,
                colors="gray",
            )
            axes[i, j].set_xticks([axes[i, j].get_xlim()[0], axes[i, j].get_xlim()[1]])
            axes[i, j].set_yticks([axes[i, j].get_ylim()[0], axes[i, j].get_ylim()[1]])
            plt.setp(axes[i, j].get_xticklabels(), visible=False)
            plt.setp(axes[i, j].get_yticklabels(), visible=False)

    axes[0, 0].set_ylabel(f"True\n{coords[0]}", fontsize=10, fontweight="semibold")
    # axes[1, 0].set_ylabel(f"Predicted\n{coords[0]}", fontsize=10, fontweight="semibold")
    # axes[2, 0].set_ylabel(f"Error\n{coords[0]}", fontsize=10, fontweight="semibold")

    # Setup FFMpeg writer with quality settings based on size_multiplier
    # Use vf filter to pad to even dimensions without distorting aspect ratio
    Writer = FFMpegWriter(
        fps=fps,
        bitrate=bitrate,
        codec="libx264",
        extra_args=[
            "-pix_fmt",
            "yuv420p",
            "-preset",
            preset,
            "-vf",
            "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        ],
    )

    # Render frames efficiently using FFMpeg writer
    data_arrays = [true_images] #, predicted_images, error_images]

    with Writer.saving(fig, output_file, dpi):
        for frame_idx in range(n_frames):
            # Update all images
            for im, row, col in ims:
                im.set_array(data_arrays[row][frame_idx, ..., col])

            suptitle.set_text(f"{dset_name} - Frame {frame_idx}/{n_frames-1}")

            # Grab and save frame
            Writer.grab_frame()

    plt.close(fig)
    plt.style.use("default")
    return dict()
