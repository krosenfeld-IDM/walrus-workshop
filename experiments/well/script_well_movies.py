"""
Make movies of the well simulations.
"""

import os
from alive_progress import alive_it, alive_bar
from omegaconf import OmegaConf
from walrus_workshop.walrus import get_trajectory, TRAJECTORY_CONFIG
from walrus_workshop.viz import make_video
from the_well.data.utils import flatten_field_names
import logging
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Set up logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def make_well_movie(dataset_id: str = "shear_flow", trajectory_id: int = 0, split: str = "test"):
    config = OmegaConf.create(TRAJECTORY_CONFIG)
    trajectory, metadata = get_trajectory(
        dataset_id=dataset_id, trajectory_id=trajectory_id, config=config, split=split
    )
    y_ref = trajectory["input_fields"][..., trajectory["padded_field_mask"]]

    field_names = flatten_field_names(metadata, include_constants=False)
    used_field_names = [
        f
        for i, f in enumerate(field_names)
        if trajectory["padded_field_mask"][i]
    ]

    trajectory_name = (
        metadata.dataset_name
        + f"{trajectory_id}_"
        + "_".join(
            [
                f"{k}_{v.item():0.0e}"
                for k, v in zip(
                    trajectory["metadata"].constant_scalar_names,
                    trajectory["constant_scalars"][0],
                )
            ]
        )
    )

    make_video(y_ref[0][:100], metadata, output_dir="movies", prefix=trajectory_name, field_name_overrides=used_field_names)

    # import json   
    # output_dir = "movies/test"
    # os.makedirs(output_dir, exist_ok=True)
    # with open(os.path.join(output_dir, f"{trajectory_name}.txt"), "w") as f:
    #     f.write(json.dumps(trajectory["extra_metadata"]))
        
    # print(f"Done with {trajectory_name}")


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_parallel", "-p", type=int, default=1)
    parser.add_argument("--max_workers", "-w", type=int, default=4)
    parser.add_argument("--split", "-s", type=str, default="test")

    args = parser.parse_args()

    if args.run_parallel:
        from concurrent.futures import ThreadPoolExecutor,as_completed 

        max_workers = args.max_workers  # Limit to 8 concurrent threads (takes about 8GB RAM per thread for 100 frames)

        trajectory_ids = list(range(0,112))

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(make_well_movie, "shear_flow", trajectory_id, args.split): trajectory_id
                for trajectory_id in trajectory_ids
            }

            with alive_bar(len(futures), title="Rendering movies") as bar:
                for future in as_completed(futures):
                    trajectory_id = futures[future]
                    # Will re-raise any exception from the worker thread
                    future.result()
                    bar()  # advance progress by 1 completed task
    else:
        for trajectory_id in alive_it(range(4,112)):    
            make_well_movie("shear_flow", trajectory_id, args.split)

    print("Done")