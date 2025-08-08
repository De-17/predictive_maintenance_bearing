```python
from fastapi import FastAPI, UploadFile, File
import numpy as np
import pandas as pd
import mlflow.pyfunc
from scipy.signal import decimate
from src.preprocessing.envelope import envl_freq
from scipy.stats import norm
from sklearn.metrics import mean_absolute_error

app = FastAPI()

# Load registered models (update version numbers as needed)
model_classifier = mlflow.pyfunc.load_model("models:/BearingModel/1")
model_ae = mlflow.pyfunc.load_model("models:/BearingModel/2")

@app.post("/predict/classifier")
async def predict_classifier(file: UploadFile = File(...)):
    """Predict bearing fault class (healthy, inner, outer, rolling) from vibration data."""
    df = pd.read_csv(file.file)
    data = df.values
    # Preprocess with envelope
    env_processor = envl_freq(sampling_frequency=5000, lowcut=2500, highcut=5000)
    data = np.array([env_processor.calculate_envelope_hilbert(x) for x in data])
    predictions = model_classifier.predict(data)
    class_labels = {0: "Healthy", 1: "Inner", 2: "Outer", 3: "Rolling"}
    predicted_classes = [class_labels[np.argmax(pred)] for pred in predictions]
    return {"predictions": predicted_classes}

@app.post("/predict/anomaly")
async def predict_anomaly(file: UploadFile = File(...)):
    """Detect anomalies using autoencoder reconstruction error."""
    df = pd.read_csv(file.file)
    data = decimate(df.values, 4)[:, :4000]  # From Autoencoder notebook
    errors = []
    for row in data:
        pred = model_ae.predict(row.reshape(1, -1, 1))
        errors.append(mean_absolute_error(row.flatten(), pred.flatten()))
    mu, sigma = norm.fit(errors)
    threshold = norm.ppf(0.99, mu, sigma)
    anomalies = [e > threshold for e in errors]
    return {
        "errors": errors,
        "threshold": threshold,
        "anomalies": anomalies,
        "anomaly_count": sum(anomalies)
    }
```