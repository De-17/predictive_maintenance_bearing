# Predictive Maintenance for Bearings

An end-to-end MLOps pipeline for bearing fault classification and anomaly detection using Paderborn, CWRU, and IMS datasets. Supports CNN classifier, Autoencoder, Siamese, Triplet learning, and EWC for continual learning.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/iamved/predictive_maintenance_bearings.git
   cd predictive_maintenance_bearings
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Pull datasets with DVC:
   ```bash
   dvc pull
   ```

## Usage

Run the main pipeline:
```bash
python src/main_pipeline.py
```

Start the FastAPI server:
```bash
uvicorn src.deployment_app:app --host 0.0.0.0 --port 8000
```

Test endpoints:
- Classifier: `POST /predict/classifier` with a CSV file of vibration data.
- Anomaly detection: `POST /predict/anomaly` with a CSV file.

## Directory Structure

- `data/`: Datasets (IMS, Paderborn, CWRU), managed with DVC.
- `src/`: Source code for preprocessing, models, and training.
- `notebooks/`: Jupyter notebooks for exploration (Triplet_learning.ipynb, etc.).
- `models/`: Checkpoint directories (embedding_ckpt/, classifier_ckpt/, etc.).
- `tests/`: Unit tests for preprocessing and models.

## Datasets

- **IMS**: Vibration data from NASA Bearing Data Center (`data/IMS/`).
- **Paderborn**: Bearing data with healthy and faulty conditions (`data/Paderborn/`).
- **CWRU**: Case Western Reserve University bearing data (`data/CWRU/`).

## Models

- **Embedding**: CNN for feature extraction using triplet loss.
- **Classifier**: Predicts bearing fault types (healthy, inner, outer, rolling).
- **Autoencoder**: Detects anomalies via reconstruction error.
- **Siamese**: Learns robust embeddings for fault classification.
- **EWC**: Continual learning to adapt to new data without forgetting.

## CI/CD

GitHub Actions workflow (`.github/workflows/ci-cd.yml`) automates testing and deployment.

## License

MIT License