import pytest
import numpy as np
from sklearn.datasets import make_classification
from brain_tumor_analysis import train_model, XGBOOST_AVAILABLE

@pytest.fixture
def sample_data():
    """Create synthetic data for testing"""
    X, y = make_classification(
        n_samples=100,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        random_state=42
    )
    return X, y

def test_train_model(sample_data):
    """Test model training function"""
    X, y = sample_data
    model, metrics = train_model(X, y)
    
    # Check if model is trained
    assert model is not None
    
    # Check metrics exist and are valid
    assert 'accuracy' in metrics
    assert 'precision' in metrics
    assert 'recall' in metrics
    assert 'f1' in metrics
    assert 'auc_roc' in metrics
    
    # Check metric values are in valid range
    for metric_name, value in metrics.items():
        assert isinstance(value, float)
        assert 0 <= value <= 1
    
    # Test model predictions
    test_pred = model.predict(X[:1])
    assert test_pred.shape == (1,)
    assert test_pred[0] in [0, 1]
    
    # Test probability predictions
    test_prob = model.predict_proba(X[:1])
    assert test_prob.shape == (1, 2)
    assert np.allclose(np.sum(test_prob, axis=1), 1)
    
    # Check if model includes XGBoost when available
    if XGBOOST_AVAILABLE:
        assert any('xgb' in est[0] for est in model.named_estimators_.items())
    else:
        assert all('xgb' not in est[0] for est in model.named_estimators_.items())
