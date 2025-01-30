import pytest
import pandas as pd
import os
from datetime import datetime
from report_generator import ReportGenerator

@pytest.fixture
def sample_patient_data():
    """Create sample patient data"""
    return pd.DataFrame({
        'age': [45, 60, 35],
        'gender': ['Male', 'Female', 'Male'],
        'tumor_size': [2.5, 3.0, 1.8],
        'genetic_risk': [7, 5, 3],
        'survival_rate': [80, 65, 90],
        'prediction': [1, 0, 1],
        'probability': [0.85, 0.35, 0.75],
        'timestamp': pd.date_range(start='2025-01-01', periods=3)
    })

def test_report_generator(sample_patient_data, tmp_path):
    """Test PDF report generation"""
    output_file = tmp_path / "test_report.pdf"
    
    # Create report
    report = ReportGenerator()
    patient_data = sample_patient_data.iloc[0].to_dict()
    report.generate_report(patient_data, str(output_file))
    
    # Check file exists
    assert output_file.exists()
    assert output_file.stat().st_size > 0
