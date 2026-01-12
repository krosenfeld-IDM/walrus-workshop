# README

TODO:
- Add a config for: expansion factor, T

# Shear flow
Data chunks are Tx256x512. At each layer the sizes are Tx32x32. T=6 by default.
Using an expansion factor of 32.

`batch` is a dict:
```
dict_keys(['input_fields', 'output_fields', 'constant_scalars', 'boundary_conditions', 'space_grid', 'input_time_grid', 'output_time_grid', 'padded_field_mask', 'field_indices', 'metadata'])
```
`input_fields` and `output_fields` have shape [B, T, W, H, D, C]. 

## blobfuse
[Notes](https://gatesfoundation.atlassian.net/wiki/spaces/~krosenfeld@idmod.org/blog/2025/02/28/3709468680/Blob+Storage+w+Azure) on Atlassian for creating the storage.