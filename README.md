# Heart Disease MLOps

This project builds and deploys a binary classifier for the processed Cleveland subset of the UCI Heart Disease dataset. The repository includes data acquisition, EDA, three tuned classifiers, MLflow tracking, a persisted preprocessing/model pipeline, a FastAPI service, tests, CI, Docker, Kubernetes, and Prometheus metrics.

# Objective: 

Design, develop, and deploy a scalable, reproducible machine learning
solution using modern MLOps best practices. The assignment emphasizes practical
automation, experiment tracking, CI/CD pipelines, containerization, cloud deployment,
and monitoring, mirroring real-world production scenarios.

# Dataset: Title: Heart Disease UCI Dataset

• Source: UCI Machine Learning Repository
• CSV containing 14+ features (age, sex, blood pressure, cholesterol, etc.) and a
  binary target (presence/absence of heart disease).

# Problem Statement: 

Build a machine learning classifier to predict the risk of heart
disease based on patient health data, and deploy the solution as a cloud-ready, monitored
API.

## Project details

- Dataset: 303 records, 13 input features; four missing `ca` values and two missing `thal` values.
- Target: original `num` value 0 is class 0; values 1-4 are class 1. Observed counts are 164 class 0 and 139 class 1.
- Preprocessing: numeric median imputation and scaling; categorical imputation and one-hot encoding, fitted inside scikit-learn pipelines.
- Model selection: stratified five-fold cross-validation; candidates are Logistic Regression, Random Forest, and AdaBoost (SAMME). The selected model is Logistic Regression.
- Serving: `GET /health`, `POST /predict`, and `GET /metrics`; the trained pipeline is included in the Docker image.
- Monitoring: Prometheus scrapes the API Service; Grafana loads the **Heart Disease API Monitoring** dashboard.

| Model | CV ROC-AUC | Holdout accuracy | Precision | Recall | Holdout ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression (selected) | 0.911344 | 0.885246 | 0.838710 | 0.928571 | 0.961039 |
| Random Forest | 0.899917 | 0.885246 | 0.838710 | 0.928571 | 0.950216 |
| AdaBoost (SAMME) | 0.894218 | 0.868852 | 0.857143 | 0.857143 | 0.944805 |

These results are from one 61-record holdout and the captured Rocky Linux MLflow runs. They are not evidence of clinical performance or generalization to other populations.

## Kubernetes deployment

The instructions below use Minikube and Docker. Docker, Minikube, and `kubectl` must be installed, and Docker must be running. Run commands from the repository root. The repository includes the trained model at `models/heart_disease_pipeline.joblib`.

### Start the cluster and prepare the image

```bash
minikube start --driver=docker
kubectl config use-context minikube
kubectl get nodes -o wide
docker build -t heart-disease-mlops:v1 .
minikube image load heart-disease-mlops:v1
```

The API deployment uses `imagePullPolicy: Never`; the image must be loaded into Minikube with the exact tag shown above.

### Apply Kubernetes resources

Create or update the Grafana dashboard ConfigMap before applying the manifests:

```bash
kubectl create configmap grafana-dashboard \
  --from-file=heart-disease-api.json=monitoring/grafana/dashboards/heart-disease-api.json \
  --dry-run=client -o yaml | kubectl apply -f -

kubectl apply -f deploy/kubernetes/
kubectl rollout status deployment/heart-disease-api --timeout=180s
kubectl rollout status deployment/prometheus --timeout=180s
kubectl rollout status deployment/grafana --timeout=180s
kubectl get deployments,pods,services -o wide
```

Expected: two API replicas are ready; Prometheus and Grafana each have one ready replica. On local Minikube, the API LoadBalancer external IP may remain pending; use port-forwarding instead of expecting a public URL.

### Test the API

In a separate terminal, keep this port-forward running:

```bash
kubectl port-forward service/heart-disease-api 8004:80
```

- Swagger UI: <http://127.0.0.1:8004/docs>
- Health check: <http://127.0.0.1:8004/health>; expected response: `{"status":"ok"}`
- Use `POST /predict` in Swagger UI with the 13 raw feature values in the request schema.

A successful prediction returns HTTP 200, a binary dataset-label prediction, and class probabilities. Exact values depend on the trained artifact; they are not clinical risk estimates.

### Verify Prometheus and Grafana

Run each port-forward in a separate terminal:

```bash
kubectl port-forward service/prometheus 9090:9090
kubectl port-forward service/grafana 3000:3000
```

- Prometheus targets: <http://127.0.0.1:9090/targets>; `heart-disease-api` should be `UP`.
- Grafana: <http://127.0.0.1:3000>, local demo login `admin` / `admin`.
- Dashboard: **Dashboards > MLOps > Heart Disease API Monitoring**.

After making a prediction request, wait at least one 15-second scrape interval and refresh Grafana. The captured dashboard evidence showed the API target UP, but the successful-predictions panel showed “No data”; verify the panel during a fresh run before claiming that counter is working. The demo credentials are for local testing only.

For troubleshooting, the full standalone [Kubernetes deployment and verification guide](KUBERNETES_DEPLOYMENT.md) covers common image, ConfigMap, pod, and port-forward issues.

### Remove the deployment

```bash
kubectl delete -f deploy/kubernetes/
kubectl delete configmap grafana-dashboard
```
