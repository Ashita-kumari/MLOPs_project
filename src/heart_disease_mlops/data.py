"""UCI data acquisition and input validation."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

import pandas as pd

from heart_disease_mlops.schema import CSV_COLUMNS, FEATURES, TARGET

UCI_ARCHIVE_URL = "https://archive.ics.uci.edu/static/public/45/heart+disease.zip"


def download_dataset(destination: str | Path) -> pd.DataFrame:
    """Download and normalize the processed Cleveland data to a CSV file."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(UCI_ARCHIVE_URL, timeout=30) as response:
        archive_bytes = response.read()

    with ZipFile(BytesIO(archive_bytes)) as archive:
        member = next(
            (name for name in archive.namelist() if name.endswith("processed.cleveland.data")),
            None,
        )
        if member is None:
            raise ValueError("UCI archive does not contain processed.cleveland.data")
        with archive.open(member) as source:
            frame = pd.read_csv(source, header=None, names=CSV_COLUMNS, na_values="?")

    frame[TARGET] = (pd.to_numeric(frame[TARGET], errors="raise") > 0).astype("int64")
    frame.to_csv(destination, index=False)
    return frame


def read_dataset(path: str | Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load a normalized dataset and validate its feature/target columns."""
    frame = pd.read_csv(path)
    missing = set(FEATURES + [TARGET]).difference(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("Dataset contains no rows")
    target = pd.to_numeric(frame[TARGET], errors="raise").astype("int64")
    if not target.isin([0, 1]).all():
        raise ValueError("Target must contain only binary values 0 and 1")
    return frame[FEATURES], target