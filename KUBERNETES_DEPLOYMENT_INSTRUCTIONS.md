# Kubernetes Deployment and Verification

This guide deploys the API, Prometheus, and Grafana to a local Minikube cluster. The service is not publicly hosted. On Minikube, the API `LoadBalancer` external address is expected to remain pending; use `kubectl port-forward` for local access.

## Prerequisites

Install and start Docker, Minikube, and `kubectl`. Run the commands below from the project root. The repository includes the trained pipeline at `models/heart_disease_pipeline.joblib` and the Grafana dashboard at `monitoring/grafana/dashboards/heart-disease-api.json`.

## Build and deploy

Start Minikube and confirm that `kubectl` is using its context:

```bash
minikube start --driver=docker
kubectl config use-context minikube
kubectl get nodes -o wide
```

Build the API image and load it into Minikube. The deployment uses `imagePullPolicy: Never`, so the image must be available inside the cluster under this exact name and tag.

```bash
docker build -t heart-disease-mlops:v1 .
minikube image load heart-disease-mlops:v1
```

Create or update the dashboard ConfigMap before applying the manifests. The Grafana deployment mounts this ConfigMap at startup.

```bash
kubectl create configmap grafana-dashboard \
  --from-file=heart-disease-api.json=monitoring/grafana/dashboards/heart-disease-api.json \
  --dry-run=client -o yaml | kubectl apply -f -

kubectl apply -f deploy/kubernetes/
```

Wait for all three deployments to become ready, then inspect pods and services:

```bash
kubectl rollout status deployment/heart-disease-api --timeout=180s
kubectl rollout status deployment/prometheus --timeout=180s
kubectl rollout status deployment/grafana --timeout=180s
kubectl get deployments,pods,services -o wide
```

Expected state: the API deployment has `2/2` ready replicas; Prometheus and Grafana each have `1/1`; their pods are `Running` and ready. The API Service is type `LoadBalancer`. A pending external IP is normal for local Minikube and does not prevent port-forward access.

## Test the API

In a dedicated terminal, start the port-forward and leave it running:

```bash
kubectl port-forward service/heart-disease-api 8004:80
```

Open the interactive API documentation at <http://127.0.0.1:8004/docs>. Check health at <http://127.0.0.1:8004/health>; the expected response is `{"status":"ok"}`. Use `POST /predict` in the docs page with this example request body:

```json
{
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
  "thal": 1
}
```

A successful request returns HTTP 200 and a prediction with class probabilities. The exact prediction and probabilities depend on the model artifact; do not treat them as clinical risk estimates.

## Verify monitoring

In separate terminals, start local port-forwards for Prometheus and Grafana:

```bash
kubectl port-forward service/prometheus 9090:9090
```

```bash
kubectl port-forward service/grafana 3000:3000
```

Open <http://127.0.0.1:9090/targets> and confirm the `heart-disease-api` target is `UP`. Open Grafana at <http://127.0.0.1:3000> and sign in with the local-demo credentials `admin` / `admin`. View **Dashboards > MLOps > Heart Disease API Monitoring**. Make a prediction request, wait at least one 15-second Prometheus scrape interval, and refresh the dashboard.
