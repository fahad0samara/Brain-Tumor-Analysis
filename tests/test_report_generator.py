import pytest
import pandas as pd
import os
from datetime import datetime
from report_generator import ReportGenerator, PLOTLY_AVAILABLE

@pytest.fixture
def sample_patient_data():
    """Create sample patient data"""
    return {
        'age': 45,
        'gender': 'Male',
        'tumor_size': 2.5,
        'genetic_risk': 7,
        'survival_rate': 80,
        'prediction': 1,
        'probability': 0.85,
        'smoking': True,
        'alcohol': False,
        'family_history': True,
        'notes': 'Test notes',
        'trend_data': {
            'dates': ['2025-01-01', '2025-01-02', '2025-01-03'],
            'values': [0.85, 0.82, 0.79]
        }
    }

def test_report_generator(sample_patient_data, tmp_path):
    """Test PDF report generation"""
    output_file = tmp_path / "test_report.pdf"
    
    # Create report
    report = ReportGenerator()
    report.generate_report(sample_patient_data, str(output_file))
    
    # Check file exists
    assert output_file.exists()
    assert output_file.stat().st_size > 0
    
    # Test with no trend data
    output_file2 = tmp_path / "test_report2.pdf"
    data_no_trend = sample_patient_data.copy()
    del data_no_trend['trend_data']
    report.generate_report(data_no_trend, str(output_file2))
    assert output_file2.exists()
    assert output_file2.stat().st_size > 0
