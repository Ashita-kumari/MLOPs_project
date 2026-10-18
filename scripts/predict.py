"""Send one sample prediction to a running local API."""

import json
from urllib.request import Request, urlopen

sample = {
    "age": 63,
    "sex": 1,
    "cp": 1,
    "trestbps": 145,
    "chol": 233,
    "fbs": 1,
    "restecg": 0,
    "thalach": 150,
    "exang": 0,
    "oldpeak": 2.3,
    "slope": 0,
    "ca": 0,
    "thal": 1,
}
request = Request(
    "http://127.0.0.1:8000/predict",
    data=json.dumps(sample).encode(),
    headers={"Content-Type": "application/json"},
    method="POST",
)
with urlopen(request, timeout=10) as response:
    print(response.read().decode())