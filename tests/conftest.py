import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_frame():
    rng = np.random.default_rng(11)
    rows = 40
    frame = pd.DataFrame(
        {
            "age": rng.integers(30, 80, rows),
            "sex": rng.integers(0, 2, rows),
            "cp": rng.integers(0, 4, rows),
            "trestbps": rng.integers(90, 180, rows),
            "chol": rng.integers(150, 350, rows),
            "fbs": rng.integers(0, 2, rows),
            "restecg": rng.integers(0, 3, rows),
            "thalach": rng.integers(90, 200, rows),
            "exang": rng.integers(0, 2, rows),
            "oldpeak": rng.uniform(0, 5, rows),
            "slope": rng.integers(0, 3, rows),
            "ca": rng.integers(0, 4, rows).astype(float),
            "thal": rng.integers(0, 4, rows),
            "target": np.tile([0, 1], rows // 2),
        }
    )
    frame.loc[0, "ca"] = np.nan
    return frame