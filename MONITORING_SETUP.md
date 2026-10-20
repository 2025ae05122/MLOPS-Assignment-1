# Monitoring Setup

This project demonstrates API monitoring using Prometheus-compatible metrics and structured API request logs.

## 1. Monitoring Architecture

The monitoring stack contains:

- `heart-api`: FastAPI model-serving application
- `heart-prometheus`: Prometheus metrics collection
- `heart-grafana`: Grafana visualisation service

Prometheus collects metrics from:

```text
http://heart-api:8000/metrics
```

The services are defined in:

```text
monitoring/docker-compose.yml
monitoring/prometheus.yml
```

## 2. Start the Monitoring Stack

Run the following command from the repository root:

```bash
docker compose -f monitoring/docker-compose.yml up -d
```

This starts the API, Prometheus and Grafana containers.

## 3. Verify the Services

```bash
docker compose -f monitoring/docker-compose.yml ps
```

The following services should be running:

```text
heart-api
heart-prometheus
heart-grafana
```

## 4. Verify the API Health Endpoint

```bash
curl -s http://localhost:8000/health | python -m json.tool
```

A successful response confirms that the API container is running and the model is available.

## 5. Verify the API Metrics Endpoint

```bash
curl -s http://localhost:8000/metrics
```

The API exposes Prometheus-compatible metrics including:

```text
heart_api_requests_total
heart_api_request_duration_seconds
heart_api_predictions_total
```

To display only request, prediction and latency metrics:

```bash
curl -s http://localhost:8000/metrics \
  | grep -Ei "request|predict|latency|duration"
```

## 6. Generate Prediction Traffic

Run the following command from the repository root:

```bash
for i in {1..10}; do
  curl -s -X POST http://localhost:8000/predict \
    -H "Content-Type: application/json" \
    --data @sample/sample_request.json
  echo
done
```

This generates prediction requests so the Prometheus counters contain observable values.

## 7. Open Prometheus

Open Prometheus in a browser:

```text
http://localhost:9090
```

Open the target-health page directly:

```text
http://localhost:9090/targets
```

The expected target is:

```text
Job: heart-disease-api
Endpoint: http://heart-api:8000/metrics
State: UP
```

## 8. Query Prediction Metrics

On the Prometheus query page, execute:

```promql
heart_api_predictions_total
```

This shows the number of model predictions grouped by prediction value.

To query the total number of API requests, execute:

```promql
heart_api_requests_total
```

To calculate average request duration, execute:

```promql
sum(heart_api_request_duration_seconds_sum)
/
sum(heart_api_request_duration_seconds_count)
```

## 9. View API Request Logs

### Docker Compose logs

```bash
docker logs heart-api --tail=20
```

### Kubernetes deployment logs

```bash
kubectl logs deployment/heart-disease-api --tail=100 \
  | grep -E "GET|POST|health|predict|prediction|confidence" \
  | tail -n 10
```

The logs demonstrate request method, path, response status and request duration.

## 10. Open Grafana

Open Grafana in a browser:

```text
http://localhost:3000
```

If Grafana is used, configure Prometheus as the data source with:

```text
http://prometheus:9090
```

Grafana is optional for this submission because monitoring is already demonstrated through Prometheus target health, Prometheus metric queries and API request logs.

## 11. Monitoring Evidence

The repository contains monitoring evidence in:

```text
screenshots/monitoring/01_prometheus_target_up.png
screenshots/monitoring/02_api_metrics_query.png
screenshots/api/01_kubernetes_api_request_logs.png
```

The screenshots demonstrate:

- Prometheus successfully scraping the API metrics endpoint
- The `heart-disease-api` target in the `UP` state
- Model prediction metrics collected by Prometheus
- Successful API request logs from the Kubernetes deployment

## 12. Stop the Monitoring Stack

```bash
docker compose -f monitoring/docker-compose.yml down
```

To remove the monitoring volumes as well:

```bash
docker compose -f monitoring/docker-compose.yml down -v
```

Use the second command only when stored Prometheus and Grafana data is no longer needed.

## 13. Related Files

```text
monitoring/docker-compose.yml
monitoring/prometheus.yml
api/app.py
sample/sample_request.json
KUBERNETES_DEPLOYMENT_SUMMARY.md
```
