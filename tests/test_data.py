import pandas as pd
import pytest

from heart_disease_mlops.data import read_dataset
from heart_disease_mlops.eda import create_eda_artifacts
from heart_disease_mlops.schema import FEATURES


def test_read_dataset_returns_expected_features_and_binary_target(tmp_path, sample_frame):
    path = tmp_path / "heart.csv"
    sample_frame.to_csv(path, index=False)

    X, y = read_dataset(path)

    assert list(X.columns) == FEATURES
    assert y.isin([0, 1]).all()
    assert X["ca"].isna().any()


def test_read_dataset_rejects_missing_columns(tmp_path):
    path = tmp_path / "incomplete.csv"
    pd.DataFrame({"age": [50], "target": [0]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing required columns"):
        read_dataset(path)


def test_read_dataset_rejects_non_binary_target(tmp_path, sample_frame):
    sample_frame.loc[0, "target"] = 2
    path = tmp_path / "invalid.csv"
    sample_frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="binary values"):
        read_dataset(path)


def test_eda_generates_readable_figures(tmp_path, sample_frame):
    from matplotlib import image

    paths = create_eda_artifacts(sample_frame, tmp_path)

    assert {path.name for path in paths} == {
        "numeric_distributions.png",
        "class_balance.png",
        "correlation_heatmap.png",
        "missing_values.png",
        "numeric_by_target.png",
        "categorical_by_target.png",
    }
    for path in paths:
        pixels = image.imread(path)
        assert pixels.shape[0] > 100
        assert pixels.shape[1] > 100
        assert pixels.std() > 0