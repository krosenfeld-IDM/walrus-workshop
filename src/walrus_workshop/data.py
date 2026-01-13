from torch.utils.data import Dataset, IterableDataset
import numpy as np
import torch
from typing import List, Tuple, Union

def split_test_train(
    data: List,
    random_state: Union[int, np.random.Generator] = 42,
    test_size: float = 0.2,
    train_size: float = None,
    shuffle: bool = True,
) -> Tuple[List, List]:
    """
    Split a list into reproducible test and train splits.
    
    Args:
        data: The list to split.
        random_state: Either an integer seed or a numpy random Generator for reproducibility.
        test_size: Proportion of data to use for testing (default: 0.2).
        train_size: Proportion of data to use for training. If None, uses 1 - test_size.
        shuffle: Whether to shuffle the data before splitting (default: True).
    
    Returns:
        A tuple of (train_list, test_list).
    
    Examples:
        >>> data = list(range(100))
        >>> train, test = split_test_train(data, random_state=42, test_size=0.2)
        >>> len(train), len(test)
        (80, 20)
    """
    if train_size is None:
        train_size = 1.0 - test_size
    
    if test_size + train_size > 1.0:
        raise ValueError(f"test_size ({test_size}) + train_size ({train_size}) cannot exceed 1.0")
    
    # Create or use random generator
    if isinstance(random_state, int):
        rng = np.random.default_rng(random_state)
    elif isinstance(random_state, np.random.Generator):
        rng = random_state
    else:
        raise TypeError(f"random_state must be int or np.random.Generator, got {type(random_state)}")
    
    # Convert to numpy array for indexing
    indices = np.arange(len(data))
    
    # Shuffle if requested
    if shuffle:
        rng.shuffle(indices)
    
    # Calculate split point
    n_total = len(data)
    n_test = int(n_total * test_size)
    n_train = int(n_total * train_size)
    
    # Ensure we don't exceed total length
    n_test = min(n_test, n_total - n_train)
    n_train = min(n_train, n_total - n_test)
    
    # Split indices
    test_indices = indices[:n_test]
    train_indices = indices[n_test:n_test + n_train]
    
    # Return lists with original data
    train_list = [data[i] for i in train_indices]
    test_list = [data[i] for i in test_indices]
    
    return train_list, test_list

class NumpyListDataset(Dataset):
    def __init__(self, numpy_arrays, device="cpu", move_to_device=True):
        """
        Args:
            numpy_arrays: List of np.ndarray, each shape [A, d_in]
        """
        # Concatenate all arrays into one large tensor
        # If dataset is too huge for RAM, you would map index -> specific array
        self.data = torch.from_numpy(np.concatenate(numpy_arrays, axis=0)).float()

        # Optional: Move to GPU immediately if VRAM allows for speed
        if move_to_device:
            self.data = self.data.to(device)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class LazyNumpyDataset(IterableDataset):
    """
    Memory-efficient dataset that streams from .npy files without loading all into RAM.

    Uses numpy's mmap_mode="r" to memory-map files, allowing the OS to page data
    in/out as needed. This keeps RAM usage bounded regardless of total dataset size.
    """

    def __init__(self, file_paths, d_in, batch_size=4096, seed=1132026):
        """
        Args:
            file_paths: List of paths to .npy files
            d_in: Expected feature dimension
            batch_size: Number of samples per batch
            seed: Random seed for shuffling
        """
        super().__init__()
        self.files = file_paths
        self.d_in = d_in
        self.batch = batch_size
        self.seed = seed

        # Pre-compute file metadata using mmap (doesn't load data into RAM)
        self.file_meta = []
        for f in self.files:
            arr = np.load(f, mmap_mode="r")
            assert arr.ndim == 2 and arr.shape[1] == d_in, f"{f} has shape {arr.shape}"
            self.file_meta.append({"path": f, "n_samples": arr.shape[0]})

        self.total_samples = sum(m["n_samples"] for m in self.file_meta)
        self.total_batches = (self.total_samples + batch_size - 1) // batch_size

    def __iter__(self):
        # Handle multi-worker sharding
        worker = torch.utils.data.get_worker_info()
        nw = worker.num_workers if worker else 1
        wid = worker.id if worker else 0
        rng = np.random.default_rng(self.seed + 997 * wid)

        # Each worker gets a shard of files
        file_shard = self.file_meta[wid::nw]
        rng.shuffle(file_shard)

        for md in file_shard:
            X = np.load(md["path"], mmap_mode="r")  # Memory-mapped, not loaded
            n = X.shape[0]
            perm = rng.permutation(n)
            for start in range(0, n, self.batch):
                sel = perm[start : start + self.batch]
                if len(sel) == 0:
                    break
                # np.asarray forces a copy from mmap to contiguous array
                yield torch.from_numpy(np.asarray(X[sel, :])).float()
