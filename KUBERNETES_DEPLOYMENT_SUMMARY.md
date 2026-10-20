# Local Minikube API Access

The API was deployed and tested locally using Minikube. A permanent public URL is not provided because the Minikube service URL is available only while the local cluster and service tunnel are running.

## 1. Start Minikube

```bash
minikube start
```

Verify that the cluster is running:

```bash
minikube status
```

Expected status:

```text
host: Running
kubelet: Running
apiserver: Running
kubeconfig: Configured
```

## 2. Select the Minikube Kubernetes context

```bash
minikube update-context
kubectl config use-context minikube
```

## 3. Deploy the API

Run these commands from the repository root:

```bash
kubectl apply -f kubernetes/deployment.yaml
kubectl apply -f kubernetes/service.yaml
```

## 4. Verify the deployment

```bash
kubectl get deployments
kubectl get pods
kubectl get services
```

Expected result:

- Deployment is available.
- Pod status is `Running`.
- Pod readiness is `1/1`.
- The `heart-disease-api` service is created.
