"""
Functions for working with the Well dataset.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional
import numpy as np

from walrus_workshop import paths


@dataclass
class WellDatasetItem:
    """Represents a single dataset file with extracted simulation parameters."""

    filename: str
    full_path: Path
    parameters: Dict[str, float] = field(default_factory=dict)

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
class WellDataSet:
    name: str
    split: Optional[str] = None
    test_size: float = 0.2
    seed: int = 42
    source_split: str = "train"
    well_base_path: str = paths.data / "datasets"

    def __post_init__(self):
        self.path = Path(self.well_base_path) / self.name / "data" / self.source_split

        # Initialize data list
        self.data: List[WellDatasetItem] = []

        # Load and split the dataset
        self._load_and_split()

    def _extract_parameters(self, filename: str) -> Dict[str, float]:
        """
        Extract simulation parameters from filename.

        General-purpose parameter extraction that handles various naming conventions:
        - ParamName_Value (e.g., Ra_1e6, Pr_0.7)
        - ParamNameValue (e.g., Reynolds1.5e3, Schmidt2.0e-1)
        - Multiple parameters in any order

        Works with any parameter names (Reynolds, Schmidt, Ra, Pr, etc.) and any number
        of parameters. Values are converted to float when possible.

        Examples:
        - shear_flow_Reynolds1.5e3_Schmidt2.0e-1.hdf5
        - rayleigh_benard_Ra_1e6_Pr_0.7.hdf5
        - dataset_Param1_Value1_Param2_Value2.hdf5

        Returns:
            Dictionary mapping parameter names to float values
        """
        parameters = {}

        # Remove extension
        name_without_ext = Path(filename).stem

        # Common dataset name prefixes to skip (case-insensitive)
        common_prefixes = {
            "shear",
            "flow",
            "well",
            "dataset",
            "rayleigh",
            "benard",
            "ben",
        }

        # Split by underscores to get potential parameter-value pairs
        parts = name_without_ext.split("_")
        processed_indices = set()  # Track which parts we've already processed

        # Pattern 1: Handle "ParamName_Value" format (e.g., Ra_1e6, Pr_0.7)
        # Look for pairs where the second part looks like a number
        i = 0
        while i < len(parts) - 1:
            param_name = parts[i].strip()
            param_value_str = parts[i + 1].strip()

            # Skip if this part was already processed or is a common prefix
            if i in processed_indices or param_name.lower() in common_prefixes:
                i += 1
                continue

            # Check if param_value_str looks like a number
            # Matches: integers, floats, scientific notation (e.g., 1e6, 1.5e-3, 0.7)
            if re.match(r"^[\d.e+-]+$", param_value_str):
                try:
                    value = float(param_value_str)
                    # Only add if param_name looks like a parameter (not empty, not starting with digit)
                    if (
                        param_name
                        and not param_name[0].isdigit()
                        and len(param_name) > 0
                    ):
                        parameters[param_name] = value
                        processed_indices.add(i)
                        processed_indices.add(i + 1)
                        i += 2  # Skip both parts
                        continue
                except ValueError:
                    pass

            i += 1

        return parameters

    def _load_and_split(self):
        """Load all HDF5 files from the split directory and set data based on split parameter."""
        if not self.path.exists():
            raise ValueError(f"Dataset path does not exist: {self.path}")

        # Find all .hdf5 files
        hdf5_files = sorted(self.path.glob("*.hdf5"))

        if not hdf5_files:
            raise ValueError(f"No .hdf5 files found in {self.path}")

        # Create dataset items
        all_items = []
        for file_path in hdf5_files:
            parameters = self._extract_parameters(file_path.name)
            item = WellDatasetItem(
                filename=file_path.name, full_path=file_path, parameters=parameters
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
                raise ValueError(
                    f"Invalid split: {self.split}. Must be None, 'train', or 'test'"
                )

    def __iter__(self) -> Iterator[WellDatasetItem]:
        """
        Iterator that returns dataset members based on the split parameter.

        Yields:
            WellDatasetItem instances from the data list
        """
        for item in self.data:
            yield item
