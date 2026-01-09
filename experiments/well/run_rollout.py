from omegaconf import OmegaConf
import torch
from hydra.utils import instantiate
from walrus_workshop import paths
from walrus.data.well_to_multi_transformer import ChannelsFirstWithTimeFormatter
from walrus_workshop.rollout import rollout_model
from the_well.data.utils import flatten_field_names
from the_well.benchmark.metrics import make_video

def main(dataset_id: [str, int] = 0, trajectory_index: int = 0):
    # checkpoint_config_path = os.path.join(".", "configs", "bubbleml_poolboil_subcool.yaml")
    checkpoint = torch.load(paths.checkpoints/"walrus.pt", map_location="cpu", weights_only=True)["app"][
        "model"
    ]    
    config = OmegaConf.load(paths.configs/"well_config.yaml")

    # The dataset objects precompute a number of dataset stats on init, so this may take a little while
    data_module = instantiate(
        config.data.module_parameters,
        well_base_path=paths.data/"datasets",
        world_size=1,
        rank=0,
        data_workers=1,
        field_index_map_override=config.data.get(
            "field_index_map_override", {}
        ),  # Use the previous field maps to avoid cycling through the data
        prefetch_field_names=False,
    )

    # Retrieve the number of fields used in training from the mapping of field to index and increment by 1
    field_to_index_map = data_module.train_dataset.field_to_index_map
    total_input_fields = max(field_to_index_map.values()) + 1

    # Instantiate the model
    model: torch.nn.Module = instantiate(
        config.model,
        n_states=total_input_fields,
    )
    # Load the checkpoint
    model.load_state_dict(checkpoint)

    # Move to the device we want
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    formatter = ChannelsFirstWithTimeFormatter()
    revin = instantiate(config.trainer.revin)()  # This is a functools partial by default

    # Grab the trajectory
    dataset_names = list(config.data.module_parameters.well_dataset_info.keys())
    if isinstance(dataset_id, str):
        dataset_index = dataset_names.index(dataset_id)
    else:
        dataset_index = dataset_id
    print(f"Using dataset {dataset_names[dataset_index]}")
    dataset = data_module.rollout_val_datasets[dataset_index].sub_dsets[trajectory_index]
    metadata = dataset.metadata

    trajectory_example = next(iter(data_module.rollout_val_dataloaders()[dataset_index]))

    with torch.no_grad():
        trajectory_example["padded_field_mask"] = trajectory_example[
            "padded_field_mask"
        ].to(device)  # We're going to want this out here too
        inputs, y_ref = formatter.process_input(
            trajectory_example,
            causal_in_time=model.causal_in_time,
            predict_delta=True,
            train=False,
        )
        y_pred, y_ref = rollout_model(
            model,
            revin,
            trajectory_example,
            formatter,
            max_rollout_steps=200,
            device=device,
        )

        # Lets get some extra info so we can visualize our data effectively
        # Remove unused fields
        y_pred, y_ref = (
            y_pred[..., trajectory_example["padded_field_mask"]],
            y_ref[..., trajectory_example["padded_field_mask"]],
        )
        # Collecting names to make detailed output logs
        field_names = flatten_field_names(metadata, include_constants=False)
        used_field_names = [
            f
            for i, f in enumerate(field_names)
            if trajectory_example["padded_field_mask"][i]
        ]


    output_dir = "./figures/"

    make_video(
        y_pred[0],  # First sample only in batch
        y_ref[0],  # First sample only in batch
        metadata,
        output_dir=output_dir,
        epoch_number=f"rollout_{trajectory_index}_example",  # Misleading parameter name, but duck typing lets it be used for naming the output. Needs upstream fix.
        field_name_overrides=used_field_names,  # Fields actually used
        size_multiplier=1.0,  #
    )


if __name__ == "__main__":
    import os
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main(dataset_id='shear_flow')