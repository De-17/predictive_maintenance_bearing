import pytest
import numpy as np
from src.preprocessing.Paderborn_data import Preprocessing_Pad
from src.preprocessing.CWRU_data import Preprocessing_CWRU
from src.preprocessing.IMS_data import Preprocessing_IMS
from src.preprocessing.envelope import envl_freq

def test_paderborn_normalize():
    """Test normalization in Paderborn_data.py."""
    prep = Preprocessing_Pad()
    data = np.random.rand(100, 1000)
    normalized = prep.normalize(data, high=1.0, low=-1.0)
    assert normalized.min() >= -1.0, "Normalization failed: min value below -1"
    assert normalized.max() <= 1.0, "Normalization failed: max value above 1"

def test_cwru_normalize():
    """Test normalization in CWRU_data.py."""
    prep = Preprocessing_CWRU()
    data = np.random.rand(100, 1000)
    normalized = prep.normalize(data, high=1.0, low=-1.0)
    assert normalized.min() >= -1.0, "Normalization failed: min value below -1"
    assert normalized.max() <= 1.0, "Normalization failed: max value above 1"

def test_ims_preprocess():
    """Test IMS preprocessing (downsampling and normalization)."""
    inner = np.random.rand(1000, 5000)
    outer = np.random.rand(1000, 5000)
    rolling = np.random.rand(1000, 5000)
    healthy = np.random.rand(1000, 5000)
    prep = Preprocessing_IMS(inner, outer, rolling, healthy)
    processed = prep.preprocess(healthy, num_points=5000, step=1000, l=1000)
    assert processed.shape[1] == 1000, "Preprocessing failed: incorrect output shape"
    assert np.abs(processed).max() <= 1.0, "Normalization failed in IMS preprocessing"

def test_envelope_hilbert():
    """Test envelope calculation in envelope.py."""
    env_processor = envl_freq(sampling_frequency=5000, lowcut=2500, highcut=5000)
    signal = np.random.rand(1000)
    envelope = env_processor.calculate_envelope_hilbert(signal)
    assert len(envelope) == len(signal), "Envelope length mismatch"
    assert np.all(envelope >= 0), "Envelope contains negative values"