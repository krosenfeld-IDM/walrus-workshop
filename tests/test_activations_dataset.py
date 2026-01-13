"""
Tests for the ActivationsDataSet class.
"""

import pytest
import numpy as np
from pathlib import Path
from walrus_workshop.activation import ActivationsDataSet, ActivationsDatasetItem
from walrus_workshop import paths


def test_activations_dataset_item_str():
    """Test that ActivationsDatasetItem returns full path when converted to string."""
    item = ActivationsDatasetItem(
        filename="layer0000_0_1e+04_1_2e-01_layerblocks.20.space_mixing.activation.npy",
        full_path=Path("/path/to/layer0000_0_1e+04_1_2e-01_layerblocks.20.space_mixing.activation.npy"),
        layer_idx=0,
        trajectory_idx=0,
        parameters={"0": 1e4, "1": 2e-1},
        tag="blocks.20.space_mixing.activation"
    )
    assert str(item) == "/path/to/layer0000_0_1e+04_1_2e-01_layerblocks.20.space_mixing.activation.npy"


def test_activations_dataset_item_metadata():
    """Test that ActivationsDatasetItem stores metadata correctly."""
    item = ActivationsDatasetItem(
        filename="layer0006_0_1e+5_1_1e+0_layerblocks.20.space_mixing.activation_tstart6.npy",
        full_path=Path("/path/to/test.npy"),
        layer_idx=6,
        trajectory_idx=0,
        parameters={"0": 1e5, "1": 1e0},
        tag="blocks.20.space_mixing.activation",
        time_str="6"
    )
    assert item.layer_idx == 6
    assert item.trajectory_idx == 0
    assert item.parameters["0"] == 1e5
    assert item.parameters["1"] == 1e0
    assert item.tag == "blocks.20.space_mixing.activation"
    assert item.time_str == "6"


def test_extract_metadata_basic_format(tmp_path):
    """Test metadata extraction for basic activation filename format."""
    # Create a temporary directory structure
    test_dir = tmp_path / "train" / "blocks.20.space_mixing.activation" / "shear_flow"
    test_dir.mkdir(parents=True)
    # Create a dummy .npy file so the dataset can initialize
    np.save(test_dir / "dummy.npy", np.array([1, 2, 3]))
    
    dataset = ActivationsDataSet(
        name="shear_flow",
        layer_name="blocks.20.space_mixing.activation",
        split="train",
        activations_base_path=tmp_path,
        source_split="train",
        test_size=0.2,
        seed=42
    )
    
    # Test the metadata extraction method directly
    filename = "layer0000_0_1e+04_1_2e-01_layerblocks.20.space_mixing.activation.npy"
    metadata = dataset._extract_metadata(filename)
    
    assert metadata['layer_idx'] == 0
    assert metadata['trajectory_idx'] == 0
    assert "0" in metadata['parameters']
    assert "1" in metadata['parameters']
    assert metadata['parameters']["0"] == 1e4
    assert metadata['parameters']["1"] == 2e-1
    assert metadata['tag'] == "blocks.20.space_mixing.activation"


def test_extract_metadata_with_tstart(tmp_path):
    """Test metadata extraction for activation filename with tstart."""
    # Create a temporary directory structure
    test_dir = tmp_path / "train" / "blocks.20.space_mixing.activation" / "shear_flow"
    test_dir.mkdir(parents=True)
    np.save(test_dir / "dummy.npy", np.array([1, 2, 3]))
    
    dataset = ActivationsDataSet(
        name="shear_flow",
        layer_name="blocks.20.space_mixing.activation",
        split="train",
        activations_base_path=tmp_path,
        source_split="train",
        test_size=0.2,
        seed=42
    )
    
    filename = "layer0006_0_1e+5_1_1e+0_layerblocks.20.space_mixing.activation_tstart6.npy"
    metadata = dataset._extract_metadata(filename)
    
    assert metadata['layer_idx'] == 6
    assert metadata['trajectory_idx'] == 0
    assert metadata['parameters']["0"] == 1e5
    assert metadata['parameters']["1"] == 1e0
    assert metadata['tag'] == "blocks.20.space_mixing.activation"
    assert metadata['time_str'] == "6"


def test_extract_metadata_multiple_parameters(tmp_path):
    """Test metadata extraction with multiple parameter pairs."""
    # Create a temporary directory structure
    test_dir = tmp_path / "train" / "blocks.20.space_mixing.activation" / "shear_flow"
    test_dir.mkdir(parents=True)
    np.save(test_dir / "dummy.npy", np.array([1, 2, 3]))
    
    dataset = ActivationsDataSet(
        name="shear_flow",
        layer_name="blocks.20.space_mixing.activation",
        split="train",
        activations_base_path=tmp_path,
        source_split="train",
        test_size=0.2,
        seed=42
    )
    
    filename = "layer0012_0_5e+04_1_2e+00_layerblocks.20.space_mixing.activation.npy"
    metadata = dataset._extract_metadata(filename)
    
    assert metadata['layer_idx'] == 12
    assert metadata['trajectory_idx'] == 0
    assert metadata['parameters']["0"] == 5e4
    assert metadata['parameters']["1"] == 2e0
    assert metadata['tag'] == "blocks.20.space_mixing.activation"


def test_dataset_split_none(tmp_path):
    """Test that split=None uses the entire dataset."""
    # Create a temporary directory structure with multiple files
    test_dir = tmp_path / "train" / "blocks.20.space_mixing.activation" / "shear_flow"
    test_dir.mkdir(parents=True)
    
    # Create multiple dummy .npy files
    for i in range(10):
        np.save(test_dir / f"layer{i:04d}_0_1e+04_1_2e-01_layerblocks.20.space_mixing.activation.npy", 
                np.array([1, 2, 3]))
    
    # Create dataset with split=None
    dataset = ActivationsDataSet(
        name="shear_flow",
        layer_name="blocks.20.space_mixing.activation",
        split=None,
        activations_base_path=tmp_path,
        source_split="train",
        test_size=0.2,
        seed=42
    )
    
    # Verify all files are in the data list
    assert len(dataset.data) == 10
    assert all(isinstance(item, ActivationsDatasetItem) for item in dataset.data)
    
    # Verify iteration works
    items = list(dataset)
    assert len(items) == 10


def test_dataset_split_train_vs_test(tmp_path):
    """Test that train and test splits are different and non-overlapping."""
    # Create a temporary directory structure with multiple files
    test_dir = tmp_path / "train" / "blocks.20.space_mixing.activation" / "shear_flow"
    test_dir.mkdir(parents=True)
    
    # Create multiple dummy .npy files
    for i in range(20):
        np.save(test_dir / f"layer{i:04d}_0_1e+04_1_2e-01_layerblocks.20.space_mixing.activation.npy",
                np.array([1, 2, 3]))
    
    # Create train and test datasets
    train_dataset = ActivationsDataSet(
        name="shear_flow",
        layer_name="blocks.20.space_mixing.activation",
        split="train",
        activations_base_path=tmp_path,
        source_split="train",
        test_size=0.2,
        seed=42
    )
    
    test_dataset = ActivationsDataSet(
        name="shear_flow",
        layer_name="blocks.20.space_mixing.activation",
        split="test",
        activations_base_path=tmp_path,
        source_split="train",
        test_size=0.2,
        seed=42
    )
    
    # Verify they are different sizes
    assert len(train_dataset.data) > 0
    assert len(test_dataset.data) > 0
    assert len(train_dataset.data) + len(test_dataset.data) == 20
    
    # Verify no overlap
    train_filenames = {item.filename for item in train_dataset.data}
    test_filenames = {item.filename for item in test_dataset.data}
    assert train_filenames.isdisjoint(test_filenames)


def test_dataset_without_layer_name(tmp_path):
    """Test that dataset works without layer_name parameter."""
    # Create a temporary directory structure without layer_name subdirectory
    test_dir = tmp_path / "train" / "shear_flow"
    test_dir.mkdir(parents=True)
    
    # Create multiple dummy .npy files
    for i in range(5):
        np.save(test_dir / f"layer{i:04d}_0_1e+04_1_2e-01_layerblocks.20.space_mixing.activation.npy",
                np.array([1, 2, 3]))
    
    # Create dataset without layer_name
    dataset = ActivationsDataSet(
        name="shear_flow",
        layer_name=None,
        split=None,
        activations_base_path=tmp_path,
        source_split="train",
        test_size=0.2,
        seed=42
    )
    
    # Verify all files are loaded
    assert len(dataset.data) == 5
    assert all(isinstance(item, ActivationsDatasetItem) for item in dataset.data)


@pytest.mark.skipif(
    not (paths.experiments / "well" / "activations" / "train" / "blocks.20.space_mixing.activation" / "shear_flow").exists(),
    reason="activations directory not found at expected path"
)
def test_activations_dataset_load():
    """Test loading actual activations dataset and verify train/test split."""
    # The actual path structure is experiments/well/activations/train/{layer_name}/{name}
    activations_base_path = paths.experiments / "well" / "activations"
    
    # Count actual files in the directory
    activations_path = activations_base_path / "train" / "blocks.20.space_mixing.activation" / "shear_flow"
    actual_files = list(activations_path.glob("*.npy"))
    total_files = len(actual_files)
    
    if total_files == 0:
        pytest.skip("No activation files found in directory")
    
    # Create train and test datasets
    train_dataset = ActivationsDataSet(
        name="shear_flow",
        layer_name="blocks.20.space_mixing.activation",
        split="train",
        activations_base_path=activations_base_path,
        source_split="train",
        test_size=0.2,
        seed=42
    )
    
    test_dataset = ActivationsDataSet(
        name="shear_flow",
        layer_name="blocks.20.space_mixing.activation",
        split="test",
        activations_base_path=activations_base_path,
        source_split="train",
        test_size=0.2,
        seed=42
    )
    
    # Verify we loaded all files
    assert len(train_dataset.data) + len(test_dataset.data) == total_files
    
    # Verify test size is approximately correct (within rounding)
    expected_test = int(total_files * 0.2)
    assert len(test_dataset.data) == expected_test or len(test_dataset.data) == expected_test + 1
    assert len(train_dataset.data) == total_files - len(test_dataset.data)
    
    # Verify all items have metadata extracted
    for item in train_dataset.data + test_dataset.data:
        assert isinstance(item, ActivationsDatasetItem)
        assert item.filename.endswith(".npy")
        assert item.full_path.exists()
        # Check that metadata was extracted
        assert item.layer_idx is not None  # layer_idx should be extracted
        assert item.tag is not None


@pytest.mark.skipif(
    not (paths.experiments / "well" / "activations" / "train" / "blocks.20.space_mixing.activation" / "shear_flow").exists(),
    reason="activations directory not found at expected path"
)
def test_activations_dataset_iterators():
    """Test that iterators work correctly for train and test sets."""
    activations_base_path = paths.experiments / "well" / "activations"
    
    train_dataset = ActivationsDataSet(
        name="shear_flow",
        layer_name="blocks.20.space_mixing.activation",
        split="train",
        activations_base_path=activations_base_path,
        source_split="train",
        test_size=0.2,
        seed=42
    )
    
    test_dataset = ActivationsDataSet(
        name="shear_flow",
        layer_name="blocks.20.space_mixing.activation",
        split="test",
        activations_base_path=activations_base_path,
        source_split="train",
        test_size=0.2,
        seed=42
    )
    
    # Test train iterator
    train_items = list(train_dataset)
    assert len(train_items) == len(train_dataset.data)
    assert all(isinstance(item, ActivationsDatasetItem) for item in train_items)
    
    # Test test iterator
    test_items = list(test_dataset)
    assert len(test_items) == len(test_dataset.data)
    assert all(isinstance(item, ActivationsDatasetItem) for item in test_items)
    
    # Verify no overlap between train and test
    train_filenames = {item.filename for item in train_dataset.data}
    test_filenames = {item.filename for item in test_dataset.data}
    assert train_filenames.isdisjoint(test_filenames)


@pytest.mark.skipif(
    not (paths.experiments / "well" / "activations" / "train" / "blocks.20.space_mixing.activation" / "shear_flow").exists(),
    reason="activations directory not found at expected path"
)
def test_activations_dataset_reproducibility():
    """Test that the dataset split is reproducible with the same seed."""
    activations_base_path = paths.experiments / "well" / "activations"
    
    # Create two train datasets with the same seed
    train_dataset1 = ActivationsDataSet(
        name="shear_flow",
        layer_name="blocks.20.space_mixing.activation",
        split="train",
        activations_base_path=activations_base_path,
        source_split="train",
        test_size=0.2,
        seed=42
    )
    
    train_dataset2 = ActivationsDataSet(
        name="shear_flow",
        layer_name="blocks.20.space_mixing.activation",
        split="train",
        activations_base_path=activations_base_path,
        source_split="train",
        test_size=0.2,
        seed=42
    )
    
    # Create two test datasets with the same seed
    test_dataset1 = ActivationsDataSet(
        name="shear_flow",
        layer_name="blocks.20.space_mixing.activation",
        split="test",
        activations_base_path=activations_base_path,
        source_split="train",
        test_size=0.2,
        seed=42
    )
    
    test_dataset2 = ActivationsDataSet(
        name="shear_flow",
        layer_name="blocks.20.space_mixing.activation",
        split="test",
        activations_base_path=activations_base_path,
        source_split="train",
        test_size=0.2,
        seed=42
    )
    
    # Verify they produce the same split
    assert len(train_dataset1.data) == len(train_dataset2.data)
    assert len(test_dataset1.data) == len(test_dataset2.data)
    
    train_filenames1 = {item.filename for item in train_dataset1.data}
    train_filenames2 = {item.filename for item in train_dataset2.data}
    assert train_filenames1 == train_filenames2
    
    test_filenames1 = {item.filename for item in test_dataset1.data}
    test_filenames2 = {item.filename for item in test_dataset2.data}
    assert test_filenames1 == test_filenames2


@pytest.mark.skipif(
    not (paths.experiments / "well" / "activations" / "train" / "blocks.20.space_mixing.activation" / "shear_flow").exists(),
    reason="activations directory not found at expected path"
)
def test_activations_dataset_metadata_extraction():
    """Test that metadata is correctly extracted from activation filenames."""
    activations_base_path = paths.experiments / "well" / "activations"
    
    train_dataset = ActivationsDataSet(
        name="shear_flow",
        layer_name="blocks.20.space_mixing.activation",
        split="train",
        activations_base_path=activations_base_path,
        source_split="train",
        test_size=0.2,
        seed=42
    )
    
    test_dataset = ActivationsDataSet(
        name="shear_flow",
        layer_name="blocks.20.space_mixing.activation",
        split="test",
        activations_base_path=activations_base_path,
        source_split="train",
        test_size=0.2,
        seed=42
    )
    
    # Check that all items have metadata extracted
    for item in train_dataset.data + test_dataset.data:
        # The filename format is: layer{idx}_{traj}_{param1}_{val1}_{param2}_{val2}_layer{tag}[_tstart{time}].npy
        # So we should extract layer_idx, trajectory_idx, parameters, and tag
        assert item.layer_idx is not None, f"Missing layer_idx in {item.filename}"
        assert item.trajectory_idx is not None, f"Missing trajectory_idx in {item.filename}"
        assert item.tag is not None, f"Missing tag in {item.filename}"
        assert len(item.parameters) > 0, f"Missing parameters in {item.filename}"
        
        # Verify layer_idx is an integer
        assert isinstance(item.layer_idx, (int, type(None)))
        
        # Verify parameters are floats
        for param_name, param_value in item.parameters.items():
            assert isinstance(param_value, float), f"Parameter {param_name} is not a float in {item.filename}"
