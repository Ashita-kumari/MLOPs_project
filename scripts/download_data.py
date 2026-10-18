from pathlib import Path

from heart_disease_mlops.data import download_dataset

ROOT = Path(__file__).resolve().parents[1]
frame = download_dataset(ROOT / "data/raw/heart.csv")
print(f"Downloaded {len(frame)} rows to data/raw/heart.csv")