"""Generate compact EDA plots from the normalized dataset."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from heart_disease_mlops.schema import (
    CATEGORICAL_FEATURES,
    FEATURES,
    NUMERIC_FEATURES,
    TARGET,
)


def create_eda_artifacts(frame: pd.DataFrame, output_dir: str | Path) -> list[Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", palette="crest")
    artifacts = []

    fig, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    for axis, feature in zip(axes.flat, NUMERIC_FEATURES):
        sns.histplot(data=frame, x=feature, hue=TARGET, bins=18, element="step", ax=axis)
        axis.set_title(feature)
    path = output_dir / "numeric_distributions.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    artifacts.append(path)

    fig, axis = plt.subplots(figsize=(6, 4), constrained_layout=True)
    balance = frame[TARGET].value_counts().sort_index().rename(
        index={0: "No disease", 1: "Disease"}
    )
    sns.barplot(x=balance.index, y=balance.values, ax=axis)
    axis.set(xlabel="Binarized target", ylabel="Patients", title="Class balance")
    path = output_dir / "class_balance.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    artifacts.append(path)

    numeric = frame[NUMERIC_FEATURES + [TARGET]].copy()
    categorical = [column for column in FEATURES if column not in NUMERIC_FEATURES]
    for column in categorical:
        numeric[column] = pd.to_numeric(frame[column], errors="coerce")
    fig, axis = plt.subplots(figsize=(11, 8), constrained_layout=True)
    sns.heatmap(numeric.corr(), cmap="vlag", center=0, annot=True, fmt=".2f", ax=axis)
    axis.set_title("Feature correlation (numeric encoding)")
    path = output_dir / "correlation_heatmap.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    artifacts.append(path)

    fig, axis = plt.subplots(figsize=(9, 5), constrained_layout=True)
    missing = frame[FEATURES].isna().sum().sort_values()
    axis.barh(missing.index, missing.values, color="#287b76")
    for position, count in enumerate(missing.values):
        axis.text(count + 0.05, position, str(count), va="center")
    axis.set(
        title="Missing values before imputation",
        xlabel="Missing records",
        xlim=(0, max(int(missing.max()) + 1, 1)),
    )
    path = output_dir / "missing_values.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    artifacts.append(path)

    fig, axes = plt.subplots(2, 3, figsize=(12, 8), constrained_layout=True)
    for axis, feature in zip(axes.flat, NUMERIC_FEATURES):
        sns.boxplot(data=frame, x=TARGET, y=feature, ax=axis)
        axis.set(title=feature, xlabel="Disease label (0 = absent, 1 = present)")
    path = output_dir / "numeric_by_target.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    artifacts.append(path)

    fig, axes = plt.subplots(3, 3, figsize=(13, 11), constrained_layout=True)
    for axis, feature in zip(axes.flat, CATEGORICAL_FEATURES):
        sns.countplot(data=frame, x=feature, hue=TARGET, ax=axis)
        axis.set(title=feature, ylabel="Patients")
    for axis in axes.flat[len(CATEGORICAL_FEATURES):]:
        axis.set_visible(False)
    fig.suptitle("Categorical features by disease label (original UCI codes)")
    path = output_dir / "categorical_by_target.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    artifacts.append(path)
    return artifacts