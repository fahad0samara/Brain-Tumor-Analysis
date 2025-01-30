import pytest
import json
import os
import pandas as pd
from datetime import datetime
from patient_history import PatientHistory

@pytest.fixture
def temp_history_file(tmp_path):
    """Create a temporary history file"""
    history_file = tmp_path / "test_history.json"
    return str(history_file)

@pytest.fixture
def patient_history(temp_history_file):
    """Create a PatientHistory instance with temp file"""
    return PatientHistory(temp_history_file)

@pytest.fixture
def sample_record():
    """Create a sample patient record"""
    return {
        'age': 45,
        'gender': 'Male',
        'tumor_size': 2.5,
        'genetic_risk': 7,
        'survival_rate': 75,
        'smoking': True,
        'alcohol': False,
        'family_history': True,
        'notes': 'Test notes',
        'prediction': 1,
        'probability': 0.85
    }

def test_add_record(patient_history, sample_record):
    """Test adding a new patient record"""
    patient_id = "test123"
    patient_history.add_record(patient_id, sample_record)
    
    # Check record was added
    history = patient_history.get_patient_history(patient_id)
    assert len(history) == 1
    assert history[0]['age'] == sample_record['age']
    assert history[0]['prediction'] == sample_record['prediction']
    assert 'timestamp' in history[0]

def test_delete_patient(patient_history, sample_record):
    """Test deleting a patient"""
    patient_id = "test123"
    patient_history.add_record(patient_id, sample_record)
    
    # Verify patient exists
    assert patient_id in patient_history.get_all_patients()
    
    # Delete patient
    success = patient_history.delete_patient(patient_id)
    assert success
    
    # Verify patient was deleted
    assert patient_id not in patient_history.get_all_patients()

def test_get_risk_trend(patient_history, sample_record):
    """Test getting risk probability trend"""
    patient_id = "test123"
    
    # Add multiple records with different probabilities
    records = []
    for i in range(3):
        record = sample_record.copy()
        record['probability'] = 0.5 + i * 0.1
        patient_history.add_record(patient_id, record)
        records.append(record)
    
    # Get trend
    dates, probabilities = patient_history.get_risk_trend(patient_id)
    
    # Check results
    assert len(dates) == 3
    assert len(probabilities) == 3
    assert all(isinstance(date, str) for date in dates)
    assert all(isinstance(prob, float) for prob in probabilities)
    assert probabilities == [0.5, 0.6, 0.7]

def test_export_patient_report(patient_history, sample_record):
    """Test exporting patient report"""
    patient_id = "test123"
    patient_history.add_record(patient_id, sample_record)
    
    # Export report
    df = patient_history.export_patient_report(patient_id)
    
    # Check DataFrame
    assert df is not None
    assert len(df) == 1
    assert 'age' in df.columns
    assert 'prediction' in df.columns
    assert 'timestamp' in df.columns
    
    # Check data types
    assert df['age'].dtype == 'int64'
    assert df['prediction'].dtype == 'int64'
    assert pd.api.types.is_datetime64_any_dtype(df['timestamp'])
