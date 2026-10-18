"""Tune, evaluate, track, and serialize candidate classifiers."""

from __future__ import annotations

import json
import os
from pathlib import Path
from shutil import copy2

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
from mlflow.models import infer_signature
from sklearn.calibration import CalibrationDisplay
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.utils import estimator_html_repr

from heart_disease_mlops.data import read_dataset
from heart_disease_mlops.pipeline import make_model_pipelines


def create_evaluation_artifacts(model, features, target, output_dir: str | Path) -> list[Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    probabilities = model.predict_proba(features)[:, 1]
    predictions = model.predict(features)
    target = np.asarray(target)
    artifacts = []

    for filename, display, values, title in [
        ("confusion_matrix.png", ConfusionMatrixDisplay, predictions, "Confusion matrix"),
        ("roc_curve.png", RocCurveDisplay, probabilities, "ROC curve"),
        ("precision_recall_curve.png", PrecisionRecallDisplay, probabilities, "Precision-recall"),
        ("calibration_curve.png", CalibrationDisplay, probabilities, "Calibration diagnostic"),
    ]:
        fig, axis = plt.subplots(figsize=(6, 5), constrained_layout=True)
        options = {"n_bins": 5, "strategy": "quantile"} if display is CalibrationDisplay else {}
        display.from_predictions(target, values, ax=axis, **options)
        axis.set_title(f"Held-out {title}")
        path = output_dir / filename
        fig.savefig(path, dpi=160)
        plt.close(fig)
        artifacts.append(path)

    fig, axis = plt.subplots(figsize=(7, 5), constrained_layout=True)
    for label, color in [(0, "#287b76"), (1, "#b65b32")]:
        axis.hist(
            probabilities[target == label], bins=np.linspace(0, 1, 11).tolist(),
            alpha=0.6, color=color, label=f"Actual label {label}",
        )
    axis.axvline(0.5, color="#333333", linestyle="--", label="Decision threshold: 0.5")
    axis.set(
        title="Held-out probabilities by actual label",
        xlabel="Model probability of disease label", ylabel="Patients",
    )
    axis.legend()
    path = output_dir / "prediction_probabilities.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    artifacts.append(path)

    classifier = model.named_steps["classifier"]
    names = model.named_steps["preprocessor"].get_feature_names_out()
    if hasattr(classifier, "coef_"):
        effects = classifier.coef_[0]
        title = "Largest logistic coefficients (transformed features; not causal)"
        xlabel = "Signed coefficient"
    else:
        effects = classifier.feature_importances_
        title = (
            f"{type(classifier).__name__} impurity importance "
            "(may favor high-cardinality features)"
        )
        xlabel = "Impurity-based importance"
    selected = np.argsort(np.abs(effects))[-15:]
    fig, axis = plt.subplots(figsize=(11, 7), constrained_layout=True)
    axis.barh(names[selected], effects[selected], color="#287b76")
    axis.set(title=title, xlabel=xlabel)
    path = output_dir / "feature_effects.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    artifacts.append(path)

    path = output_dir / "model_pipeline.html"
    path.write_text(estimator_html_repr(model), encoding="utf-8")
    artifacts.append(path)
    return artifacts


def train(data_path: str | Path, model_dir: str | Path, artifact_dir: str | Path) -> dict:
    X, y = read_dataset(data_path)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "file:./mlruns")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("uci-heart-disease")
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    candidates = []
    for name, (pipeline, grid) in make_model_pipelines().items():
        with mlflow.start_run(run_name=name) as run:
            search = GridSearchCV(
                pipeline,
                grid,
                scoring="roc_auc",
                cv=cv,
                n_jobs=-1,
                refit=True,
                error_score="raise",
            )
            search.fit(X_train, y_train)
            mlflow.log_params({"model_family": name, "cv_folds": 5, **search.best_params_})
            mlflow.log_metric("cv_roc_auc", float(search.best_score_))
            mlflow.sklearn.log_model(
                search.best_estimator_,
                artifact_path="model",
                input_example=X_train.head(2),
                signature=infer_signature(X_train, search.best_estimator_.predict(X_train)),
            )
            candidates.append(
                {"name": name, "search": search, "run_id": run.info.run_id}
            )

    selected = max(candidates, key=lambda candidate: candidate["search"].best_score_)
    model = selected["search"].best_estimator_
    metrics = {
        "selected_model": selected["name"],
        "best_cv_roc_auc": float(selected["search"].best_score_),
        "test_size": int(len(y_test)),
        "selected_run_id": selected["run_id"],
    }
    comparison = []
    for candidate in candidates:
        estimator = candidate["search"].best_estimator_
        predictions = estimator.predict(X_test)
        probabilities = estimator.predict_proba(X_test)[:, 1]
        scores = {
            "test_accuracy": float(accuracy_score(y_test, predictions)),
            "test_precision": float(precision_score(y_test, predictions, zero_division=0)),
            "test_recall": float(recall_score(y_test, predictions, zero_division=0)),
            "test_roc_auc": float(roc_auc_score(y_test, probabilities)),
        }
        candidate_dir = artifact_dir / candidate["name"]
        with mlflow.start_run(run_id=candidate["run_id"]):
            mlflow.log_metrics(scores)
            mlflow.log_dict(scores, "evaluation/test_metrics.json")
            paths = create_evaluation_artifacts(estimator, X_test, y_test, candidate_dir)
            for path in paths:
                mlflow.log_artifact(str(path), "evaluation")
                if candidate is selected:
                    copy2(path, artifact_dir / path.name)
            results = candidate["search"].cv_results_
            fig, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
            axis.errorbar(
                np.arange(len(results["params"])), results["mean_test_score"],
                yerr=results["std_test_score"], fmt="o", capsize=4,
            )
            axis.set(
                title=f"{candidate['name']}: tuning mean and fold standard deviation",
                xlabel="Parameter grid candidate index", ylabel="CV ROC-AUC", ylim=(0, 1.05),
            )
            tuning_path = candidate_dir / "tuning_scores.png"
            fig.savefig(tuning_path, dpi=160)
            plt.close(fig)
            mlflow.log_artifact(str(tuning_path), "tuning")
            mlflow.log_dict(
                {str(index): params for index, params in enumerate(results["params"])},
                "tuning/parameter_grid.json",
            )
        comparison.append({"model": candidate["name"], "cv_roc_auc": float(
            candidate["search"].best_score_
        ), **scores})
        if candidate is selected:
            metrics.update(scores)

    fig, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    positions = np.arange(len(comparison))
    colors = ["#287b76", "#b65b32", "#5477a8"]
    axis.bar(
        positions,
        [row["cv_roc_auc"] for row in comparison],
        color=colors[:len(comparison)],
    )
    axis.set_xticks(positions, [row["model"].replace("_", " ") for row in comparison])
    axis.set(title="Model selection: best mean 5-fold CV ROC-AUC", ylabel="CV ROC-AUC", ylim=(0, 1))
    comparison_path = artifact_dir / "model_cv_comparison.png"
    fig.savefig(comparison_path, dpi=160)
    plt.close(fig)
    comparison_json = artifact_dir / "model_comparison.json"
    comparison_json.write_text(json.dumps(comparison, indent=2) + "\n", encoding="utf-8")
    for candidate in candidates:
        with mlflow.start_run(run_id=candidate["run_id"]):
            mlflow.log_artifact(str(comparison_path), "comparison")
            mlflow.log_artifact(str(comparison_json), "comparison")

    import joblib

    model_path = model_dir / "heart_disease_pipeline.joblib"
    joblib.dump(model, model_path)
    metrics_path = artifact_dir / "test_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return metrics