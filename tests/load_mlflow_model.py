import mlflow
import numpy as np

# Connect to the mlflow server
mlflow.set_tracking_uri("http://localhost:5000")

model_name = "DummyRegressorModel"
alias = "champion"

# Load the model using the alias
model_uri = f"models:/{model_name}@{alias}"
model = mlflow.sklearn.load_model(model_uri)

print(f"Model loaded successfully from alias '{alias}'.")

# make a prediction using the loaded model
X_new = np.array([[6]])
prediction = model.predict(X_new)
print(f"Prediction for input {X_new}: {prediction}")
