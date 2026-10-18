import numpy as np
import pytest

from heart_disease_mlops.pipeline import make_model_pipelines, make_preprocessor
from heart_disease_mlops.train import create_evaluation_artifacts


def test_preprocessor_imputes_missing_values_and_encodes_categories(sample_frame):
    X = sample_frame.drop(columns="target")
    transformed = make_preprocessor().fit_transform(X)

    assert transformed.shape[0] == len(X)
    assert transformed.shape[1] > X.shape[1]
    dense = transformed.toarray() if hasattr(transformed, "toarray") else transformed
    assert np.isfinite(dense).all()


def test_logistic_pipeline_fits_and_returns_probabilities(sample_frame):
    X = sample_frame.drop(columns="target")
    y = sample_frame["target"]
    pipeline, _ = make_model_pipelines()["logistic_regression"]

    pipeline.fit(X, y)

    assert pipeline.predict(X[:3]).shape == (3,)
    assert pipeline.predict_proba(X[:3]).shape == (3, 2)


@pytest.mark.parametrize(
    "model_name", ["logistic_regression", "random_forest", "adaboost"]
)
def test_evaluation_artifacts_for_each_model(model_name, tmp_path, sample_frame):
    from matplotlib import image

    features = sample_frame.drop(columns="target")
    target = sample_frame["target"]
    model, _ = make_model_pipelines()[model_name]
    model.fit(features, target)

    paths = create_evaluation_artifacts(model, features, target, tmp_path / model_name)

    assert {path.name for path in paths} == {
        "confusion_matrix.png", "roc_curve.png", "precision_recall_curve.png",
        "calibration_curve.png", "prediction_probabilities.png", "feature_effects.png",
        "model_pipeline.html",
    }
    for path in paths:
        if path.suffix == ".png":
            assert image.imread(path).std() > 0
        else:
            classifier_name = type(model.named_steps["classifier"]).__name__
            assert classifier_name in path.read_text(encoding="utf-8")