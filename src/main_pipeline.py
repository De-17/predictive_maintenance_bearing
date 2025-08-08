
import os
import mlflow
import mlflow.tensorflow
from mlflow.models.signature import infer_signature
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, mean_absolute_error, confusion_matrix
from sklearn.model_selection import train_test_split
from scipy.stats import norm
from statsmodels.tsa.seasonal import STL
import plotly.graph_objs as go
from plotly.offline import iplot

# Import project modules
from src.preprocessing.Paderborn_data import Import_Data_Pad, Preprocessing_Pad
from src.preprocessing.CWRU_data import Preprocessing_CWRU
from src.preprocessing.IMS_data import Import_Data_IMS, Preprocessing_IMS
from src.preprocessing.envelope import envl_freq
from src.models.CNN_model import Embedding, Classification
from src.models.AE_model import Autoencoder
from src.training.train_embeddings import Train_Embeddings
from src.training.train_classifier import Training_classifier
from src.training.train_AE import cal_error, Training_AE
from src.training.pretraining import Pretraining
from src.training.EWC import Fisher_matrix, EWC
from src.training.Triplet_loss import Triplet_Loss
from src.training.train_siamese import Bearings_Network

# MLflow setup
mlflow.set_tracking_uri("http://localhost:5000")  # Run `mlflow ui` first
mlflow.set_experiment("Predictive_Maintenance_Bearings_Advanced")

# Configuration
DATA_PATH_PAD_HEALTHY15 = "data/Paderborn/healthy15"
DATA_PATH_PAD_HEALTHY09 = "data/Paderborn/healthy09"
DATA_PATH_PAD_INNER15 = "data/Paderborn/inner15"
DATA_PATH_PAD_INNER09 = "data/Paderborn/inner09"
DATA_PATH_PAD_OUTER15 = "data/Paderborn/outer15"
DATA_PATH_PAD_OUTER09 = "data/Paderborn/outer09"
DATA_PATH_CWRU = "data/CWRU"
IMS_TEST1_PATH = "data/IMS/Test_1.csv"
IMS_TEST2_PATH = "data/IMS/Test_2.csv"

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

LEARNING_RATE = 0.001
BATCH_SIZE = 32
EPOCHS = 100
EMBEDDING_DIM = 20
NUM_CLASSES = 4
AE_INPUT_SIZE = 4000  # From Autoencoder notebook

def load_and_preprocess_data(dataset_type="paderborn"):
    """Data ingestion and preprocessing for Paderborn, CWRU, or IMS datasets."""
    if dataset_type == "paderborn":
        data_loader = Import_Data_Pad(
            DATA_PATH_PAD_HEALTHY15, DATA_PATH_PAD_HEALTHY09,
            DATA_PATH_PAD_INNER15, DATA_PATH_PAD_INNER09,
            DATA_PATH_PAD_OUTER15, DATA_PATH_PAD_OUTER09
        )
        preprocessor = Preprocessing_Pad()
        X_train, X_test, y_train, y_test, load_train, load_test = preprocessor.get_X_y()

    elif dataset_type == "cwru":
        preprocessor = Preprocessing_CWRU()
        preprocessor.loadData()  # Assumes CWRU data in data/CWRU/
        X_train, X_test, y_train, y_test, load_train, load_test = preprocessor.get_X_y()

    elif dataset_type == "ims":
        test1 = Import_Data_IMS(IMS_TEST1_PATH)
        test2 = Import_Data_IMS(IMS_TEST2_PATH)
        healthy1 = test1.t_2[:500, :]
        healthy2 = test2.t_0[:200, :]
        healthy3 = test1.t_3[:300, :]
        inner = test1.t_2
        outer = test2.t_0
        rolling = test1.t_3
        prep = Preprocessing_IMS(inner=inner, outer=outer, rolling=rolling, healthy=np.concatenate((healthy1, healthy2, healthy3)))
        X_train, X_test = prep.get_X_y()
        y_train, y_test = np.zeros(len(X_train)), np.zeros(len(X_test))  # For AE

        # Envelope processing
        env_processor = envl_freq(sampling_frequency=5000, lowcut=2500, highcut=5000)
        X_train = np.array([env_processor.calculate_envelope_hilbert(x) for x in X_train])
        X_test = np.array([env_processor.calculate_envelope_hilbert(x) for x in X_test])

    else:
        raise ValueError("Unsupported dataset type")

    mlflow.log_param("dataset_type", dataset_type)
    mlflow.log_param("train_samples", len(X_train))
    mlflow.log_param("test_samples", len(X_test))

    return X_train, X_test, y_train, y_test

def get_time_failure(error, threshold, points=5, proportion=0.4):
    """Identify the timestamp where failure begins using STL decomposition."""
    decompose = STL(error, period=100).fit()
    error_trend = decompose.trend
    idx_unhealthy = np.argwhere(error_trend >= threshold).flatten()
    for i in idx_unhealthy:
        idx_next = [j for j in range(i, i + points)]
        next_health = [k in idx_unhealthy for k in idx_next]
        prop = sum(next_health) / points
        if prop >= 3 / 5:
            idx_next_tot = np.arange(i, error_trend.shape[0])
            next_health_tot = [k in idx_unhealthy for k in idx_next_tot]
            prop_tot = sum(next_health_tot) / idx_next_tot.shape[0]
            if prop_tot > proportion:
                return i
    return None

def train_model(X_train, y_train, model_type="classifier", previous_model=None, X_prev=None, y_prev=None):
    """Train specified model type with MLflow tracking."""
    with mlflow.start_run(nested=True):
        if model_type == "embedding":
            model = Embedding(n_input=AE_INPUT_SIZE)
            trainer = Train_Embeddings(
                net=model, learning_rate=LEARNING_RATE, training_iters=EPOCHS,
                batch_size=BATCH_SIZE, display_step=10, triplet_strategy="batch_hard",
                margin=1.0, squared=False, filepath=os.path.join(MODEL_DIR, "embedding_ckpt")
            )
            trainer.fit(X_train, y_train)

        elif model_type == "classifier":
            embedding_model = Embedding(n_input=AE_INPUT_SIZE)
            model = Classification(embedding=embedding_model, num_classes=NUM_CLASSES)
            trainer = Training_classifier(
                net=model, learning_rate=LEARNING_RATE, training_iters=EPOCHS,
                batch_size=BATCH_SIZE, display_step=10, filepath=os.path.join(MODEL_DIR, "classifier_ckpt")
            )
            trainer.fit(X_train, y_train)

        elif model_type == "autoencoder":
            model = Autoencoder()
            trainer = Training_AE(
                net=model, learning_rate=LEARNING_RATE, training_iters=EPOCHS,
                batch_size=BATCH_SIZE, display_step=10, early_stopping=10,
                filepath=os.path.join(MODEL_DIR, "ae_ckpt")
            )
            X_train_ae, X_val_ae = train_test_split(X_train, test_size=0.2)
            trainer.fit(X_train_ae, X_val_ae)

        elif model_type == "siamese":
            model = Bearings_Network(type_net="Siamese").model
            # Placeholder: Implement pair generation and fit as per Siamese.ipynb
            # hist = model.fit(...)

        elif model_type == "ewc":
            if previous_model is None:
                raise ValueError("Previous model required for EWC")
            fisher = Fisher_matrix(X_prev, y_prev, previous_model, task="cross_entropy", batch_size=BATCH_SIZE)
            model = previous_model
            ewc_trainer = EWC(
                net=model, learning_rate=LEARNING_RATE, training_iters=EPOCHS,
                batch_size=BATCH_SIZE, display_step=10, filepath=os.path.join(MODEL_DIR, "ewc_ckpt"),
                fisher_matrix=fisher.fisher_matrix, lamb=1.0
            )
            ewc_trainer.fit(X_train, y_train, X_prev, y_prev)

        # Log and save
        signature = infer_signature(X_train[:10], model.predict(X_train[:10]))
        mlflow.tensorflow.log_model(model, "model", signature=signature)
        mlflow.log_param("model_type", model_type)
        return model

def evaluate_model(model, X_test, y_test, model_type="classifier"):
    """Evaluate model and log metrics."""
    predictions = model.predict(X_test)

    if model_type in ["classifier", "ewc"]:
        acc = accuracy_score(y_test, np.argmax(predictions, axis=1))
        cm = confusion_matrix(y_test, np.argmax(predictions, axis=1))
        mlflow.log_metric("accuracy", acc)
        print(f"Accuracy: {acc}")
        print("Confusion Matrix:\n", cm)
        np.savetxt(os.path.join(MODEL_DIR, "cm.txt"), cm, fmt="%d")
        mlflow.log_artifact(os.path.join(MODEL_DIR, "cm.txt"))

    elif model_type == "autoencoder":
        errors = cal_error(X_test, model)
        mu, sigma = norm.fit(errors)
        threshold = norm.ppf(0.99, mu, sigma)
        mlflow.log_metric("mae_mean", mu)
        mlflow.log_metric("mae_std", sigma)
        mlflow.log_metric("threshold", threshold)
        print(f"MAE Mean: {mu}, Std: {sigma}, Threshold: {threshold}")

        # Visualize reconstruction error
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=np.arange(len(errors)), y=errors, mode="lines", name="Reconstruction Error"))
        fig.add_trace(go.Scatter(y=[threshold] * len(errors), name="Threshold", line=dict(color="red")))
        time_failure = get_time_failure(errors, threshold, points=10)
        if time_failure:
            fig.add_trace(go.Scatter(x=[time_failure] * 100, y=np.linspace(0, max(errors), 100), name="Failure Start", line=dict(color="orange")))
        fig.update_layout(title="Reconstruction Error", xaxis_title="Timestamp", yaxis_title="Error")
        fig.write(os.path.join(MODEL_DIR, "error_plot.html"))
        mlflow.log_artifact(os.path.join(MODEL_DIR, "error_plot.html"))

if __name__ == "__main__":
    with mlflow.start_run(run_name="Full_Pipeline"):
        # Train classifier on CWRU (or Paderborn)
        X_train_cwru, X_test_cwru, y_train_cwru, y_test_cwru = load_and_preprocess_data("cwru")
        model_classifier = train_model(X_train_cwru, y_train_cwru, model_type="classifier")
        evaluate_model(model_classifier, X_test_cwru, y_test_cwru, "classifier")

        # Continual learning with EWC on IMS
        X_train_ims, X_test_ims, y_train_ims, y_test_ims = load_and_preprocess_data("ims")
        model_ewc = train_model(X_train_ims, y_train_ims, model_type="ewc", previous_model=model_classifier, X_prev=X_train_cwru, y_prev=y_train_cwru)
        evaluate_model(model_ewc, X_test_ims, y_test_ims, "ewc")
        evaluate_model(model_ewc, X_test_cwru, y_test_cwru, "ewc")  # Check forgetting

        # Autoencoder for anomaly detection
        model_ae = train_model(X_train_ims, y_train_ims, model_type="autoencoder")
        evaluate_model(model_ae, X_test_ims, y_test_ims, "autoencoder")

        # Register and deploy
        run_id = mlflow.active_run().info.run_id
        model_uri = f"runs:/{run_id}/model"
        mlflow.register_model(model_uri, "BearingModel")
        print(f"Model URI: {model_uri}")
