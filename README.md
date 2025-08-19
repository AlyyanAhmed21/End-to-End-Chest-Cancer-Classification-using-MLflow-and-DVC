# End-to-End-Chest-Cancer-Classification-using-MLflow-and-DVC

import dagshub
dagshub.init(repo_owner='AlyyanAhmed21', repo_name='End-to-End-Chest-Cancer-Classification-using-MLflow-and-DVC', mlflow=True)

import mlflow
with mlflow.start_run():
  mlflow.log_param('parameter name', 'value')
  mlflow.log_metric('metric name', 1)

export MLFLOW_TRACKING_URI=https://dagshub.com/AlyyanAhmed21/End-to-End-Chest-Cancer-Classification-using-MLflow-and-DVC.mlflow

export MLFLOW_TRACKING_USERNAME=AlyyanAhmed21 

export MLFLOW_TRACKING_PASSWORD=776454e991d86ea3a96179a4dc1ef72fbc134642

$env:MLFLOW_TRACKING_URI = "https://dagshub.com/AlyyanAhmed21/End-to-End-Chest-Cancer-Classification-using-MLflow-and-DVC.mlflow"
$env:MLFLOW_TRACKING_USERNAME = "AlyyanAhmed21"
$env:MLFLOW_TRACKING_PASSWORD = "776454e991d86ea3a96179a4dc1ef72fbc134642"