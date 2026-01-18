from pathlib import Path

root = Path(__file__).parents[2]

data = root / "data"

experiments = root / "experiments"

checkpoints = root / "checkpoints"

configs = root / "configs"

well_base_path = data / "datasets"