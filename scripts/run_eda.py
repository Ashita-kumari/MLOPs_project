from pathlib import Path

from heart_disease_mlops.data import read_dataset
from heart_disease_mlops.eda import create_eda_artifacts

ROOT = Path(__file__).resolve().parents[1]
features, target = read_dataset(ROOT / "data/raw/heart.csv")
frame = features.assign(target=target)
paths = create_eda_artifacts(frame, ROOT / "artifacts/eda")
print("Created EDA artifacts:")
for path in paths:
    print(path.relative_to(ROOT))