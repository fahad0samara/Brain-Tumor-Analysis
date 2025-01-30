import pytest
import pandas as pd
import numpy as np
from brain_tumor_preprocessing import clean_data, engineer_features, prepare_features

@pytest.fixture
def sample_data():
    """Create sample data for testing"""
    return pd.DataFrame({
        'Age': [45, 60, np.nan, 35],
        'Gender': ['Male', 'Female', 'Male', None],
        'Tumor_Size': [2.5, 3.0, 1.8, 4.2],
        'Genetic_Risk': [7, 5, 3, 8],
        'Survival_Rate(%)': [80, 65, 90, 55],
        'Brain_Tumor_Present': ['Yes', 'No', 'Yes', 'No']
    })

def test_clean_data(sample_data):
    """Test data cleaning function"""
    cleaned_data = clean_data(sample_data)
    
    # Check no missing values
    assert cleaned_data.isnull().sum().sum() == 0
    
    # Check data types
    assert cleaned_data['Age'].dtype == np.float64
    assert cleaned_data['Gender'].dtype == object
    
    # Check value ranges
    assert cleaned_data['Age'].between(0, 100).all()
    assert cleaned_data['Tumor_Size'].between(0, 10).all()
    assert cleaned_data['Genetic_Risk'].between(0, 10).all()
    assert cleaned_data['Survival_Rate(%)'].between(0, 100).all()

def test_engineer_features(sample_data):
    """Test feature engineering function"""
    cleaned_data = clean_data(sample_data)
    engineered_data = engineer_features(cleaned_data)
    
    # Check new features exist
    assert 'Risk_Score' in engineered_data.columns
    assert 'Medical_Complexity' in engineered_data.columns
    assert 'Age_Risk' in engineered_data.columns
    assert 'Tumor_Severity' in engineered_data.columns
    
    # Check new features have no missing values
    new_features = ['Risk_Score', 'Medical_Complexity', 'Age_Risk', 'Tumor_Severity']
    assert engineered_data[new_features].isnull().sum().sum() == 0
    
    # Check value ranges
    assert engineered_data['Risk_Score'].between(0, 1).all()
    assert engineered_data['Medical_Complexity'].dtype in [np.int64, np.float64]

def test_prepare_features(sample_data):
    """Test feature preparation for ML"""
    cleaned_data = clean_data(sample_data)
    engineered_data = engineer_features(cleaned_data)
    X, y = prepare_features(engineered_data)
    
    # Check output shapes
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert len(X) == len(y)
    
    # Check target values
    assert y.isin([0, 1]).all()
    
    # Check no missing values
    assert X.isnull().sum().sum() == 0
    assert y.isnull().sum() == 0
