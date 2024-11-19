# MLflow on Minikube Setup Instructions
MLflow is an open-source platform that streamlines the entire machine learning lifecycle. Its key features include:
Experiment Tracking and Reproducibility
MLflow provides robust capabilities for tracking experiments and ensuring reproducibility:
1. Logs parameters, metrics, and artifacts for each model training run
2. Enables easy comparison between different models or configurations
3. Facilitates reproduction of previous experiments
4. Helps understand how various parameters impact model performance

In this specific setup:
MLflow is deployed on a local Minikube cluster
The MLflow service is accessible on localhost:5000 via port forwarding
This configuration allows you to leverage MLflow's capabilities while running it in a containerized environment on your local machine, providing a scalable and portable solution for managing your ML experiments.

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
