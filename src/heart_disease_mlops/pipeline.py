"""Reusable preprocessing and classifier pipelines."""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from heart_disease_mlops.schema import CATEGORICAL_FEATURES, NUMERIC_FEATURES


def make_preprocessor() -> ColumnTransformer:
    numeric = Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric, NUMERIC_FEATURES),
            ("categorical", categorical, CATEGORICAL_FEATURES),
        ]
    )


def make_model_pipelines(random_state: int = 42) -> dict[str, tuple[Pipeline, dict]]:
    """Return baseline families and compact, reproducible tuning grids."""
    return {
        "logistic_regression": (
            Pipeline(
                [
                    ("preprocessor", make_preprocessor()),
                    ("classifier", LogisticRegression(max_iter=2000, random_state=random_state)),
                ]
            ),
            {"classifier__C": [0.1, 1.0, 10.0], "classifier__class_weight": [None, "balanced"]},
        ),
        "random_forest": (
            Pipeline(
                [
                    ("preprocessor", make_preprocessor()),
                    (
                        "classifier",
                        RandomForestClassifier(random_state=random_state, n_jobs=1),
                    ),
                ]
            ),
            {
                "classifier__n_estimators": [150, 300],
                "classifier__max_depth": [None, 5],
                "classifier__min_samples_leaf": [1, 3],
            },
        ),
        "adaboost": (
            Pipeline(
                [
                    ("preprocessor", make_preprocessor()),
                    (
                        "classifier",
                        AdaBoostClassifier(algorithm="SAMME", random_state=random_state),
                    ),
                ]
            ),
            {
                "classifier__n_estimators": [50, 100],
                "classifier__learning_rate": [0.05, 0.5, 1.0],
            },
        ),
    }