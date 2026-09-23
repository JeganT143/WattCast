import mlflow
from mlflow import MlflowClient

mlflow.set_tracking_uri("http://localhost:5000")

client = MlflowClient()

model_name = "DummyRegressorModel"
model_version = 1  
alias = "champion"  

# Assign the alias to the version of the model
client.set_registered_model_alias(model_name, alias, model_version)

print(f"Alias '{alias}' has been set for model '{model_name}' version {model_version}.")


# verify the alias assignment
model_info = client.get_model_version_by_alias(model_name, alias)

print(f"Model '{model_name}' version {model_info.version} is now associated with alias '{alias}'.")