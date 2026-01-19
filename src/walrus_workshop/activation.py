import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence
from torch.utils.data import DataLoader
import numpy as np
import zarr

from walrus_workshop import paths


class ActivationManager:
    """
    Handles selective saving of activations from walrus.

    Source: https://github.com/theodoremacmillan/graphcast/blob/sae-hooks/graphcast/deep_typed_graph_net.py
    """

    def __init__(self,
                 enabled: bool = False,
                 save_dir: Optional[str] = None,
                 save_steps: Optional[Sequence[int]] = None,
                 save_node_sets: Optional[Sequence[str]] = None,
                 mode: str = "post_res"):
        self.enabled = enabled
        self.save_dir = save_dir
        self.save_steps = save_steps
        self.save_node_sets = save_node_sets
        self.mode = mode
        self.current_time_str: Optional[str] = None   # <--- NEW
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

    # --- add these small helpers ---
    def set_time(self, time_str: Optional[str]):
        """Set a global time string (e.g. '2021-09-28T06Z') for subsequent saves."""
        self.current_time_str = time_str

    def clear_time(self):
        """Unset the current global time label."""
        self.current_time_str = None

    def _should_save(self, tag: str, step_idx: Optional[int], node_set: str) -> bool:
        """Determine if this activation should be saved."""
        if not self.enabled:
            return False
        if self.mode not in tag and self.mode != "both":
            return False
        if self.save_steps is not None and step_idx not in self.save_steps:
            return False
        if self.save_node_sets is not None and node_set not in self.save_node_sets:
            return False
        return True

    def save(self, tag: str, x, *,
             step_idx: Optional[int] = None,
             node_set: Optional[str] = None,
             time_str: Optional[str] = None):
        """Save activation array x for given step / node set."""
        if not self._should_save(tag, step_idx, node_set):
            return

        # Prefer explicit time_str if provided, else use the global one
        ts = time_str or self.current_time_str
        arr = np.asarray(x).copy()

        safe_tag = tag.replace("/", "_")
        step_prefix = f"step_{step_idx:04d}_" if step_idx is not None else ""
        time_suffix = f"_t{ts}" if ts else ""
        node_suffix = f"_{node_set}" if node_set else ""

        fname = f"{step_prefix}{safe_tag}{time_suffix}.zarr"
        zarr.save(os.path.join(self.save_dir, fname), arr.astype(np.float16))

    def get_cache(self):
        """Return in-memory cache (if using memory mode)."""
        return self._cache if self._cache is not None else {}

    def clear(self):
        """Clear in-memory cache."""
        if self._cache is not None:
            self._cache.clear()


# Global instance (imported across modules)
_ACT_MANAGER = ActivationManager()

def get_activation_manager():
    """Return global ActivationManager instance."""
    return _ACT_MANAGER


@dataclass
class ActivationsDatasetItem:
    """Represents a single activation file with extracted metadata."""
    filename: str
    full_path: Path
    layer_idx: Optional[int] = None
    trajectory_idx: Optional[int] = None
    parameters: Dict[str, float] = field(default_factory=dict)
    tag: Optional[str] = None
    time_str: Optional[str] = None
    
    def __str__(self) -> str:
        """Return the full path of the file."""
        return str(self.full_path)
    
    def __fspath__(self) -> str:
        """Return the full path for use with path-like operations (e.g., np.load)."""
        return str(self.full_path.as_posix())

    @property
    def resolve(self) -> Path:
        """Resolve the full path of the file."""
        return self.full_path.resolve


@dataclass
class ActivationsDataSet:
    """Dataset class for loading activation files saved by ActivationManager."""
    
    name: str
    layer_name: Optional[str] = None
    split: Optional[str] = None
    activations_base_path: Optional[str] = None
    test_size: float = 0.2
    seed: int = 42
    source_split: str = "train"
    
    def __post_init__(self):

        if not self.activations_base_path:
            self.activations_base_path = Path(os.path.join(os.getcwd(), "activations"))

        if self.layer_name:
            self.path = Path(self.activations_base_path) / self.source_split / self.layer_name / self.name
        else:
            self.path = Path(self.activations_base_path) / self.source_split / self.name

        # Initialize data list
        self.data: List[ActivationsDatasetItem] = []
        
        # Load and split the dataset
        self._load_and_split()
    
    def _extract_metadata(self, filename: str) -> Dict:
        """
        Extract metadata from activation filename.
        
        Activation files are saved with the pattern:
        - layer{step_idx:04d}_{trajectory_idx}_{param1}_{value1}_{param2}_{value2}_layer{tag}[_tstart{time}].npy
        
        Examples:
        - layer0000_0_1e+04_1_2e-01_layerblocks.20.space_mixing.activation.npy
        - layer0006_0_1e+5_1_1e+0_layerblocks.20.space_mixing.activation_tstart6.npy
        
        Returns:
            Dictionary with keys: layer_idx, trajectory_idx, parameters, tag, time_str
        """
        metadata = {
            'layer_idx': None,
            'trajectory_idx': None,
            'parameters': {},
            'tag': None,
            'time_str': None
        }
        
        # Remove extension
        name_without_ext = Path(filename).stem
        
        # Extract layer index (e.g., "layer0000" -> 0)
        layer_match = re.match(r'^layer(\d+)', name_without_ext)
        if layer_match:
            try:
                metadata['layer_idx'] = int(layer_match.group(1))
            except ValueError:
                pass
        
        # Remove layer prefix to get remaining parts
        remaining = re.sub(r'^layer\d+_?', '', name_without_ext)
        
        # Extract time string (pattern: _tstart{time})
        time_match = re.search(r'_tstart(\d+)', remaining)
        if time_match:
            metadata['time_str'] = time_match.group(1)
            remaining = re.sub(r'_tstart\d+', '', remaining)
        
        # Split by underscores
        parts = remaining.split('_')
        
        if not parts:
            return metadata
        
        # First part should be trajectory_idx (a number)
        if parts[0].isdigit():
            try:
                metadata['trajectory_idx'] = int(parts[0])
                parts = parts[1:]
            except ValueError:
                pass
        
        # Find the index of the part that starts with "layer" (this is the tag prefix)
        layer_tag_idx = None
        for i, part in enumerate(parts):
            if part == 'layer':
                layer_tag_idx = i
                break
        
        if layer_tag_idx is not None:
            # Everything before "layer" are parameter pairs
            param_parts = parts[:layer_tag_idx]
            # Everything after "layer" is the tag
            tag_parts = parts[layer_tag_idx + 1:]
            metadata['tag'] = '_'.join(tag_parts) if tag_parts else None
            
            # Extract parameters (pairs of name_value)
            # Parameters are in format: {name}_{value} where value is in scientific notation
            i = 0
            while i < len(param_parts) - 1:
                param_name = param_parts[i]
                param_value_str = param_parts[i + 1]
                
                # Check if param_value_str looks like a number (scientific notation)
                if re.match(r'^[\d.e+-]+$', param_value_str):
                    try:
                        value = float(param_value_str)
                        metadata['parameters'][param_name] = value
                        i += 2
                        continue
                    except ValueError:
                        pass
                i += 1
        else:
            # No "layer" found, try to extract parameters from all parts
            # This handles edge cases where the format might be slightly different
            i = 0
            while i < len(parts) - 1:
                param_name = parts[i]
                param_value_str = parts[i + 1]
                
                if re.match(r'^[\d.e+-]+$', param_value_str):
                    try:
                        value = float(param_value_str)
                        metadata['parameters'][param_name] = value
                        i += 2
                        continue
                    except ValueError:
                        pass
                i += 1
        
        return metadata

    @property
    def d_in(self) -> int:
        if not hasattr(self, '_d_in'):
            # Compute d_in from data if not already set
            if not self.data:
                raise ValueError("Cannot compute d_in: no data loaded")
            arr = zarr.open(str(self.data[0].full_path), mode='r')
            self._d_in = arr.shape[1]
        return self._d_in

    def _load_and_split(self):
        """Load all .zarr files from the directory and set data based on split parameter."""
        if not self.path.exists():
            raise ValueError(f"Activations path does not exist: {self.path}")

        # Find all .zarr directories
        zarr_files = sorted([p for p in self.path.iterdir()
                            if p.suffix == '.zarr' and p.is_dir()])

        if not zarr_files:
            raise ValueError(f"No .zarr files found in {self.path}")
        
        # Create dataset items
        all_items = []
        for file_path in zarr_files:
            metadata = self._extract_metadata(file_path.name)
            item = ActivationsDatasetItem(
                filename=file_path.name,
                full_path=file_path,
                layer_idx=metadata['layer_idx'],
                trajectory_idx=metadata['trajectory_idx'],
                parameters=metadata['parameters'],
                tag=metadata['tag'],
                time_str=metadata['time_str']
            )
            all_items.append(item)
        
        # If split is None, use all items
        if self.split is None:
            self.data = all_items
        else:
            # Split into train/test with reproducible ordering
            rng = np.random.default_rng(self.seed)
            indices = np.arange(len(all_items))
            rng.shuffle(indices)
            
            n_test = int(len(all_items) * self.test_size)
            
            test_indices = indices[:n_test]
            train_indices = indices[n_test:]
            
            test_items = [all_items[i] for i in test_indices]
            train_items = [all_items[i] for i in train_indices]
            
            # Set data based on split parameter
            if self.split == "train":
                self.data = train_items
            elif self.split == "test":
                self.data = test_items
            else:
                raise ValueError(f"Invalid split: {self.split}. Must be None, 'train', or 'test'")
    
    def __iter__(self) -> Iterator[ActivationsDatasetItem]:
        """
        Iterator that returns dataset members based on the split parameter.
        
        Yields:
            ActivationsDatasetItem instances from the data list
        """
        for item in self.data:
            yield item

    def __len__(self) -> int:
        """Return the number of items in the dataset."""
        return len(self.data)

    def to_dataloader(self, batch_size: int = 4096, num_workers: int = 4, seed: Optional[int] = None) -> DataLoader:
        """
        Create a PyTorch DataLoader for this dataset.

        Args:
            batch_size: Number of samples per batch
            num_workers: Number of DataLoader workers
            seed: Random seed for shuffling (defaults to self.seed)

        Returns:
            PyTorch DataLoader ready for training
        """
        from walrus_workshop.data import LazyZarrDataset

        file_paths = [item.full_path for item in self.data]
        seed = seed if seed is not None else self.seed

        ds = LazyZarrDataset(file_paths, d_in=self.d_in, batch_size=batch_size, seed=seed)
        return DataLoader(ds, num_workers=num_workers, batch_size=None)