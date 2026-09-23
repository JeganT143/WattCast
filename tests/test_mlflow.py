import mlflow
import mlflow.sklearn
from sklearn.dummy import DummyRegressor
import numpy as np

# Connect to the MLFlow server
mlflow.set_tracking_uri("http://localhost:5000")

# Create Fake Data
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([10, 20, 30, 40, 50])

# Create and train a Dummy Regressor model
model = DummyRegressor(strategy="mean")
model.fit(X, y)

print("Model trained successfully.")
print("Predictions:", model.predict(np.array([[6]])))

with mlflow.start_run() as run:
    # log the model
    model_info = mlflow.sklearn.log_model(model, "dummy_model")
    print(f"Model logged with run ID: {run.info.run_id}")

    # Register the model
    registered_model = mlflow.register_model(model_info.model_uri, "DummyRegressorModel")
    print(f"Model registered with name: {registered_model.name} and version: {registered_model.version}")
    