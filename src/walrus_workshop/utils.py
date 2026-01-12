from typing import List, Tuple, Union
import numpy as np
import inspect


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

def filter_kwargs(kwargs, module):
    """
    Filter a dictionary of kwargs to only include keys that the module accepts.
    """

    # Get the list of arguments SAE expects
    module_args = set(inspect.signature(module.__init__).parameters)

    # Create a filtered dict containing ONLY keys that SAE accepts
    model_kwargs = {k: v for k, v in kwargs.items() if k in module_args}    
    return model_kwargs 