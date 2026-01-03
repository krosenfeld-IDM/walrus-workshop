import torch
import copy
from alive_progress import alive_it
from walrus.trainer.training import expand_mask_to_match

def rollout_model(
    model,
    revin,
    batch,
    formatter,
    max_rollout_steps=200,
    model_epsilon=1e-5,
    device=torch.device("cpu"),
):
    """Rollout the model for as many steps as we have data for.

    Simplified version of the trainer method for demo purposes.
    """
    metadata = batch["metadata"]
    batch = {
        k: v.to(device) if k not in {"metadata", "boundary_conditions"} else v
        for k, v in batch.items()
    }
    # Extract mask and move to device for loss eval
    if (
        "mask"
        in batch["metadata"].constant_field_names[
            0
        ]  # Assuming all metadata in batch are the same
    ):
        mask_index = batch["metadata"].constant_field_names[0].index("mask")
        mask = batch["constant_fields"][..., mask_index : mask_index + 1]
        mask = mask.to(device, dtype=torch.bool)
    else:
        mask = None

    inputs, y_ref = formatter.process_input(
        batch,
        causal_in_time=model.causal_in_time,
        predict_delta=True,
        train=False,
    )

    # Inputs T B C H [W D], y_ref B T H [W D] C
    T_in = batch["input_fields"].shape[1]
    max_rollout_steps = max_rollout_steps + (T_in - 1)
    rollout_steps = min(
        y_ref.shape[1], max_rollout_steps
    )  # Number of timesteps in target
    train_rollout_limit = 1

    y_ref = y_ref[
        :, :rollout_steps
    ]  # If we set a maximum number of rollout steps, just cut it off now to save memory
    # Create a moving batch of one step at a time
    moving_batch = copy.deepcopy(batch)
    # moving_batch = batch
    y_preds = []
    # Rollout the model - Causal in time gets more predictions from the first step
    for i in alive_it(range(train_rollout_limit - 1, rollout_steps)):
        # Don't fill causal_in_time here since that only affects y_ref
        inputs, _ = formatter.process_input(moving_batch)
        inputs = list(inputs)
        with torch.no_grad():
            normalization_stats = revin.compute_stats(
                inputs[0], metadata, epsilon=model_epsilon
            )
        # NOTE - Currently assuming only [0] (fields) needs normalization
        normalized_inputs = inputs[:]  # Shallow copy
        normalized_inputs[0] = revin.normalize_stdmean(
            normalized_inputs[0], normalization_stats
        )
        y_pred = model(
            normalized_inputs[0],
            normalized_inputs[1],
            normalized_inputs[2].tolist(),
            metadata=metadata,
        )
        # During validation, don't maintain full inner predictions
        if model.causal_in_time:
            y_pred = y_pred[-1:]  # y_pred is T first, y_ref is not
        # In validation, we want to reconstruct predictions on original scale
        y_pred = inputs[0][-y_pred.shape[0] :].float() + revin.denormalize_delta(
            y_pred, normalization_stats
        )  # Unnormalize delta and add to input
        y_pred = formatter.process_output(y_pred, metadata)[
            ..., : y_ref.shape[-1]
        ]  # Cut off constant channels

        # If we have masked fields, just move them back to zeros
        if mask is not None:
            mask_pred = expand_mask_to_match(mask, y_pred)
            y_pred.masked_fill_(mask_pred, 0)

        y_pred = y_pred.masked_fill(~batch["padded_field_mask"], 0.0)

        # If not last step, update moving batch for autoregressive prediction
        if i != rollout_steps - 1:
            moving_batch["input_fields"] = torch.cat(
                [moving_batch["input_fields"][:, 1:], y_pred[:, -1:]], dim=1
            )
        # For causal models, we get use full predictions for the first batch and
        # incremental predictions for subsequent batches - concat 1:T to y_ref for loss eval
        if model.causal_in_time and i == train_rollout_limit - 1:
            y_preds.append(y_pred)
        else:
            y_preds.append(y_pred[:, -1:])
    y_pred_out = torch.cat(y_preds, dim=1)
    if mask is not None:
        mask_ref = expand_mask_to_match(mask, y_ref)
        y_ref.masked_fill_(mask_ref, 0)
    return y_pred_out, y_ref