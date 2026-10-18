# Heart Disease MLOps

An end-to-end, reproducible MLOps project for the UCI Heart Disease Cleveland subset. The repository includes data acquisition, EDA, three tuned classifiers, MLflow tracking, a persisted preprocessing/model pipeline, a FastAPI service, tests, CI, Docker, Kubernetes, and Prometheus metrics.

> This educational model is not a medical device and must not be used for diagnosis or treatment. The dataset is small, historical, and not representative of every population.

## Dataset and target

`scripts/download_data.py` downloads the processed Cleveland file from the UCI Machine Learning Repository. It contains 303 records, 13 input features, and an integer disease target. The source target values 1-4 are mapped to positive (1); 0 is mapped to negative (0). The `ca` and `thal` fields use `?` for missing values. The original data is retained in `data/raw/heart.csv`; preprocessing is fitted only on training folds inside the sklearn pipeline.

## Quick start

Python 3.11 is recommended. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
python scripts/download_data.py
python scripts/run_eda.py
python scripts/train_model.py
pytest
```

The EDA figures are written to `artifacts/eda/`. Training compares tuned Logistic Regression, Random Forest, and AdaBoost pipelines with stratified 5-fold ROC-AUC, freezes the CV-based selection, then generates holdout diagnostics for all three. Holdout plots are for reporting, not further tuning. Parameters, metrics, plots, and models are logged to local MLflow (`./mlruns`); the selected serving pipeline is saved to `models/heart_disease_pipeline.joblib`.

### Plots and diagrams

EDA generates distributions, class balance, correlation, missing-value counts, numeric boxplots by target, and categorical counts by target.

Each of `artifacts/training/logistic_regression/`, `artifacts/training/random_forest/`, and `artifacts/training/adaboost/` contains:

- Confusion matrix, ROC curve, precision-recall curve, and calibration diagnostic.
- Predicted-probability distributions grouped by actual label.
- Feature effects: signed logistic coefficients or estimator impurity importance.
- Tuning scores: CV mean and fold standard deviation for each grid candidate.
- `model_pipeline.html`: an expandable sklearn diagram; open it in a browser.

The training root contains `model_cv_comparison.png`, `model_comparison.json`, and copies of the selected model's evaluation figures. In each new MLflow run, open `Artifacts/evaluation`, `Artifacts/tuning`, and `Artifacts/comparison`. Earlier runs are unchanged.

Coefficients and forest importance have different scales and are not causal effects. Calibration on this small holdout does not establish clinical reliability. Tuning error bars show fold standard deviation, not confidence intervals.

View tracked runs with `mlflow ui --backend-store-uri ./mlruns` and open `http://127.0.0.1:5000`.

## Run the API

```bash
uvicorn heart_disease_mlops.api:app --host 0.0.0.0 --port 8000
```

The service exposes `/health`, `/predict`, and Prometheus `/metrics`. `/predict` expects one JSON object with all 13 raw feature names. Example:

```bash
curl -X POST http://127.0.0.1:8000/predict \\
  -H 'Content-Type: application/json' \\
  -d '{"age":63,"sex":1,"cp":1,"trestbps":145,"chol":233,"fbs":1,"restecg":0,"thalach":150,"exang":0,"oldpeak":2.3,"slope":0,"ca":0,"thal":1}'
```

The response contains the binary prediction, a human-readable label, confidence, and class probabilities. Requests are logged without the submitted health values. The API reports probabilities, not clinical risk.

## Docker

Train the model first, then build and run:

```bash
docker build -t heart-disease-mlops:local .
docker run --rm -p 8000:8000 heart-disease-mlops:local
```

In another terminal, call `/health` and `/predict` using the sample request above. The image includes the fitted model and installs only serving dependencies.

To run the API, Prometheus, and Grafana together, use Docker Compose after training. Stop standalone containers using ports 8000, 9090, or 3000 first:

```bash
docker compose up --build
```

The API is at `http://127.0.0.1:8000`, Prometheus at `http://127.0.0.1:9090`, and Grafana at `http://127.0.0.1:3000` (local demo login `admin` / `admin`). Grafana provisions the Prometheus source and the **Heart Disease API Monitoring** dashboard automatically. Generate predictions at the API docs and allow one 15-second scrape interval to see counts and request rates update. This local demo password is not suitable for a network-exposed deployment.

## Kubernetes (Minikube / Docker Desktop)

The Kubernetes manifests run the API, Prometheus, and Grafana together. Prometheus scrapes the API Service at `heart-disease-api:80`; Grafana queries the in-cluster `prometheus` Service. The API uses local image `heart-disease-mlops:v1` with `imagePullPolicy: Never`, so it will not pull from a registry. Build the image first. For Minikube:

```bash
eval "$(minikube docker-env)"
docker build -t heart-disease-mlops:v1 .
kubectl create configmap grafana-dashboard --from-file=heart-disease-api.json=monitoring/grafana/dashboards/heart-disease-api.json --dry-run=client -o yaml | kubectl apply -f -
kubectl apply -f deploy/kubernetes/
kubectl rollout status deployment/heart-disease-api
kubectl rollout status deployment/prometheus
kubectl rollout status deployment/grafana
minikube service heart-disease-api --url
```

For Docker Desktop Kubernetes, select the single-node `kubeadm` provisioner and a compatible Docker image store; its `kind` provisioner has separate local image requirements. Verify the current context, Ready nodes, and `docker image inspect heart-disease-mlops:v1`. Create/update the `grafana-dashboard` ConfigMap from `monitoring/grafana/dashboards/heart-disease-api.json`, then apply `deploy/kubernetes/` and wait for all three deployments. In separate terminals, port-forward the API to 8004, Prometheus to 9090, and Grafana to 3000. Confirm Prometheus `/targets` is UP and inspect **Dashboards > MLOps > Heart Disease API Monitoring** in Grafana. See `WINDOWS_SETUP.md` for the full PowerShell workflow. Since `imagePullPolicy: Never` prevents registry access, use a registry-configured manifest for remote clusters.

## Development and reproducibility

```bash
ruff check src tests scripts
pytest
python scripts/train_model.py --help
```

`make all` downloads data, generates EDA artifacts, trains the model, and runs tests. Set `MLFLOW_TRACKING_URI` to use a remote tracking server. Set `MODEL_PATH` to serve a model artifact from another location. The project pins direct dependencies in `requirements.txt`; the serving image uses `requirements-serving.txt`.

Latest three-model comparison: Logistic Regression CV ROC-AUC 0.9113 (holdout ROC-AUC 0.9610), Random Forest 0.9003 (0.9524), and AdaBoost with SAMME 0.8942 (0.9448). Logistic Regression remains selected by CV. All holdout metrics are for the same 61 records and are comparison evidence for this split, not a clinical performance claim.

## Repository map

```text
src/heart_disease_mlops/  data, preprocessing, training, and API package
scripts/                 download, EDA, training, and inference entry points
tests/                   unit and API contract tests
.github/workflows/       lint, test, train, and artifact upload
deploy/kubernetes/       API, Prometheus, and Grafana Deployments/Services
monitoring/              Prometheus scrape config and Grafana data source/dashboard provisioning
REPORT.md                assignment report draft and evidence checklist
```

## Assignment evidence still requiring your own run

The workflow and manifests are included, but this workspace cannot publish a GitHub repository, create cloud credentials, record a video, or capture screenshots of your deployment. Run the pipeline and local deployment yourself, save genuine screenshots/video under `screenshots/`, then update `REPORT.md` with your repository URL and actual run metrics. Do not present unexecuted cloud deployment or results as completed.

The output is a classifier for this dataset's label, not a clinical risk score.
