# MLflow on Minikube Setup Instructions
## Install Minikube
Follow the official Minikube installation guide for your operating system
Start Minikube with: 
```
minikube start
```
## Create MLflow namespace
Run: 
```
kubectl create namespace mlflow
```
## Apply MLflow deployment and service
Save the provided YAML content to a file named 
```
mlflow-deployment.yaml
```
Apply the configuration: 
```
kubectl apply -f mlflow-deployment.yaml
```
## Verify deployment and service
Check deployment status: 
```
kubectl get deployments -n mlflow
```
Check service status: 
```
kubectl get services -n mlflow
```
## Port forward to localhost 
Run: 
```
kubectl port-forward service/mlflow-service 5000:5000 -n mlflow
```
This will forward requests from localhost:5000 to the MLflow service 
## Access MLflow UI
Open a web browser and navigate to http://localhost:5000
