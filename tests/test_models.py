import pytest
import tensorflow as tf
import numpy as np
from src.models.CNN_model import Embedding, Classification
from src.models.AE_model import Autoencoder
from src.models.Triplet_loss import Triplet_Loss

def test_embedding_model():
    """Test Embedding model forward pass."""
    model = Embedding(n_input=1000)
    input_data = tf.random.uniform((10, 1000, 1))
    output = model(input_data, is_training=False)
    assert output.shape == (10, 20), "Embedding output shape incorrect"

def test_classification_model():
    """Test Classification model forward pass."""
    embedding_model = Embedding(n_input=1000)
    model = Classification(embedding=embedding_model, n_input=1000, num_classes=4)
    input_data = tf.random.uniform((10, 1000, 1))
    output = model(input_data, is_training=False)
    assert output.shape == (10, 4), "Classification output shape incorrect"

def test_autoencoder_model():
    """Test Autoencoder model forward pass."""
    model = Autoencoder()
    input_data = tf.random.uniform((10, 4000, 1))
    output = model(input_data, is_training=False)
    assert output.shape == (10, 4000, 1), "Autoencoder output shape incorrect"

def test_triplet_loss():
    """Test Triplet Loss calculation."""
    labels = tf.constant([0, 0, 1, 1], dtype=tf.float32)
    embeddings = tf.random.uniform((4, 20))
    triplet_loss = Triplet_Loss(labels, embeddings, margin=1.0, squared=False)
    loss, fraction, _ = triplet_loss.batch_all_triplet_loss()
    assert loss.shape == (), "Triplet loss should be scalar"
    assert 0 <= fraction <= 1, "Fraction of positive triplets out of range"