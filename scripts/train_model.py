import argparse
from pathlib import Path

from heart_disease_mlops.train import train

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune and track heart disease classifiers")
    parser.add_argument("--data", type=Path, default=ROOT / "data/raw/heart.csv")
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models")
    parser.add_argument("--artifact-dir", type=Path, default=ROOT / "artifacts/training")
    args = parser.parse_args()
    print(train(args.data, args.model_dir, args.artifact_dir))


if __name__ == "__main__":
    main()
