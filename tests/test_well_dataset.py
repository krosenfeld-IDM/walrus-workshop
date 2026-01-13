"""
Tests for the WellDataSet class.
"""

import pytest
from pathlib import Path
from walrus_workshop.well import WellDataSet, WellDatasetItem
from walrus_workshop import paths


def test_well_dataset_item_str():
    """Test that WellDatasetItem returns full path when converted to string."""
    item = WellDatasetItem(
        filename="test.hdf5",
        full_path=Path("/path/to/test.hdf5"),
        parameters={"Reynolds": 1.5e3}
    )
    assert str(item) == "/path/to/test.hdf5"


def test_well_dataset_item_parameters():
    """Test that WellDatasetItem stores parameters correctly."""
    item = WellDatasetItem(
        filename="test.hdf5",
        full_path=Path("/path/to/test.hdf5"),
        parameters={"Reynolds": 1.5e3, "Schmidt": 2.0e-1}
    )
    assert item.parameters["Reynolds"] == 1.5e3
    assert item.parameters["Schmidt"] == 2.0e-1
    assert len(item.parameters) == 2


def test_extract_parameters_shear_flow_format(tmp_path):
    """Test parameter extraction for shear_flow naming convention."""
    # Create a temporary directory structure to avoid loading actual dataset
    test_dir = tmp_path / "test" / "train"
    test_dir.mkdir(parents=True)
    # Create a dummy file so the dataset can initialize
    (test_dir / "dummy.hdf5").touch()
    
    dataset = WellDataSet(
        name="test",
        split="train",
        well_base_path=tmp_path,
        source_split="train",
        test_size=0.2,
        seed=42
    )
    
    # Test the parameter extraction method directly
    filename = "shear_flow_Reynolds_5e4_Schmidt_5e-1.hdf5"
    params = dataset._extract_parameters(filename)
    
    assert "Reynolds" in params
    assert "Schmidt" in params
    assert params["Reynolds"] == 5e4
    assert params["Schmidt"] == 5e-1


def test_extract_parameters_rayleigh_benard_format(tmp_path):
    """Test parameter extraction for rayleigh_benard naming convention."""
    # Create a temporary directory structure
    test_dir = tmp_path / "test" / "train"
    test_dir.mkdir(parents=True)
    (test_dir / "dummy.hdf5").touch()
    
    dataset = WellDataSet(
        name="test",
        split="train",
        well_base_path=tmp_path,
        source_split="train",
        test_size=0.2,
        seed=42
    )
    
    # Test with Ra_Pr format
    filename = "rayleigh_benard_Ra_1e6_Pr_0.7.hdf5"
    params = dataset._extract_parameters(filename)
    
    assert "Ra" in params
    assert "Pr" in params
    assert params["Ra"] == 1e6
    assert params["Pr"] == 0.7


def test_extract_parameters_combined_format(tmp_path):
    """Test parameter extraction with combined ParamNameValue format."""
    # Create a temporary directory structure
    test_dir = tmp_path / "test" / "train"
    test_dir.mkdir(parents=True)
    (test_dir / "dummy.hdf5").touch()
    
    dataset = WellDataSet(
        name="test",
        split="train",
        well_base_path=tmp_path,
        source_split="train",
        test_size=0.2,
        seed=42
    )
    
    filename = "shear_flow_Reynolds_1.5e3_Schmidt_2.0e-1.hdf5"
    params = dataset._extract_parameters(filename)
    
    assert "Reynolds" in params
    assert "Schmidt" in params
    assert params["Reynolds"] == 1.5e3
    assert params["Schmidt"] == 2.0e-1


@pytest.mark.skipif(
    not (paths.data / "datasets" / "shear_flow" / "data" / "train").exists(),
    reason="shear_flow dataset not found at expected path"
)
def test_shear_flow_dataset_load():
    """Test loading the shear_flow train dataset and verify train/test split."""
    # The actual path structure is data/datasets/shear_flow/data/train
    # So we need to adjust the well_base_path accordingly
    well_base_path = paths.data / "datasets" / "shear_flow" / "data"
    
    dataset = WellDataSet(
        name="",  # Empty name since we're already in the data directory
        split="train",
        well_base_path=well_base_path,
        source_split="train",
        test_size=0.2,
        seed=42
    )
    
    # Count actual files in the directory
    train_path = paths.data / "datasets" / "shear_flow" / "data" / "train"
    actual_files = list(train_path.glob("*.hdf5"))
    total_files = len(actual_files)
    
    # Verify we loaded all files
    assert len(dataset.train) + len(dataset.test) == total_files
    
    # Verify test size is approximately correct (within rounding)
    expected_test = int(total_files * 0.2)
    assert len(dataset.test) == expected_test or len(dataset.test) == expected_test + 1
    assert len(dataset.train) == total_files - len(dataset.test)
    
    # Verify all items have parameters extracted
    for item in dataset.train + dataset.test:
        assert isinstance(item, WellDatasetItem)
        assert item.filename.endswith(".hdf5")
        assert item.full_path.exists()
        # Check that parameters were extracted (should have at least Reynolds and Schmidt)
        assert len(item.parameters) > 0


@pytest.mark.skipif(
    not (paths.data / "datasets" / "shear_flow" / "data" / "train").exists(),
    reason="shear_flow dataset not found at expected path"
)
def test_shear_flow_dataset_iterators():
    """Test that iterators work correctly for train and test sets."""
    well_base_path = paths.data / "datasets" / "shear_flow" / "data"
    
    dataset = WellDataSet(
        name="",
        split="train",
        well_base_path=well_base_path,
        source_split="train",
        test_size=0.2,
        seed=42
    )
    
    # Test train iterator
    train_items = list(dataset.iter_train())
    assert len(train_items) == len(dataset.train)
    assert all(isinstance(item, WellDatasetItem) for item in train_items)
    
    # Test test iterator
    test_items = list(dataset.iter_test())
    assert len(test_items) == len(dataset.test)
    assert all(isinstance(item, WellDatasetItem) for item in test_items)
    
    # Test default iterator (should iterate over train when split="train")
    default_items = list(dataset)
    assert len(default_items) == len(dataset.train)
    
    # Verify no overlap between train and test
    train_filenames = {item.filename for item in dataset.train}
    test_filenames = {item.filename for item in dataset.test}
    assert train_filenames.isdisjoint(test_filenames)


@pytest.mark.skipif(
    not (paths.data / "datasets" / "shear_flow" / "data" / "train").exists(),
    reason="shear_flow dataset not found at expected path"
)
def test_shear_flow_dataset_reproducibility():
    """Test that the dataset split is reproducible with the same seed."""
    well_base_path = paths.data / "datasets" / "shear_flow" / "data"
    
    # Create two datasets with the same seed
    dataset1 = WellDataSet(
        name="",
        split="train",
        well_base_path=well_base_path,
        source_split="train",
        test_size=0.2,
        seed=42
    )
    
    dataset2 = WellDataSet(
        name="",
        split="train",
        well_base_path=well_base_path,
        source_split="train",
        test_size=0.2,
        seed=42
    )
    
    # Verify they produce the same split
    assert len(dataset1.train) == len(dataset2.train)
    assert len(dataset1.test) == len(dataset2.test)
    
    train_filenames1 = {item.filename for item in dataset1.train}
    train_filenames2 = {item.filename for item in dataset2.train}
    assert train_filenames1 == train_filenames2
    
    test_filenames1 = {item.filename for item in dataset1.test}
    test_filenames2 = {item.filename for item in dataset2.test}
    assert test_filenames1 == test_filenames2


@pytest.mark.skipif(
    not (paths.data / "datasets" / "shear_flow" / "data" / "train").exists(),
    reason="shear_flow dataset not found at expected path"
)
def test_shear_flow_dataset_parameter_extraction():
    """Test that parameters are correctly extracted from shear_flow filenames."""
    well_base_path = paths.data / "datasets" / "shear_flow" / "data"
    
    dataset = WellDataSet(
        name="",
        split="train",
        well_base_path=well_base_path,
        source_split="train",
        test_size=0.2,
        seed=42
    )
    
    # Check that all items have Reynolds and Schmidt parameters
    for item in dataset.train + dataset.test:
        # The filename format is: shear_flow_Reynolds_X_Schmidt_Y.hdf5
        # So we should extract both parameters
        assert "Reynolds" in item.parameters, f"Missing Reynolds in {item.filename}"
        assert "Schmidt" in item.parameters, f"Missing Schmidt in {item.filename}"
        
        # Verify parameters are floats
        assert isinstance(item.parameters["Reynolds"], float)
        assert isinstance(item.parameters["Schmidt"], float)
        
        # Verify parameters are positive (reasonable for these physical quantities)
        assert item.parameters["Reynolds"] > 0
        assert item.parameters["Schmidt"] > 0
