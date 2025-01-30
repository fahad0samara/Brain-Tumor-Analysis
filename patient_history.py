import pandas as pd
import json
from datetime import datetime
import os

class PatientHistory:
    def __init__(self, history_file='patient_history.json'):
        self.history_file = history_file
        self._load_history()
    
    def _load_history(self):
        """Load patient history from file"""
        if os.path.exists(self.history_file):
            with open(self.history_file, 'r') as f:
                self.history = json.load(f)
        else:
            self.history = {}
    
    def _save_history(self):
        """Save patient history to file"""
        with open(self.history_file, 'w') as f:
            json.dump(self.history, f, indent=4)
    
    def add_record(self, patient_id, record):
        """Add a new patient record"""
        if patient_id not in self.history:
            self.history[patient_id] = []
        
        # Add timestamp to record
        record['timestamp'] = datetime.now().isoformat()
        
        self.history[patient_id].append(record)
        self._save_history()
    
    def delete_patient(self, patient_id):
        """Delete a patient and all their records"""
        if patient_id in self.history:
            del self.history[patient_id]
            self._save_history()
            return True
        return False
    
    def get_patient_history(self, patient_id):
        """Get history for a specific patient"""
        return self.history.get(patient_id, [])
    
    def get_all_patients(self):
        """Get list of all patient IDs"""
        return list(self.history.keys())
    
    def get_risk_trend(self, patient_id):
        """Get risk probability trend for a patient"""
        history = self.get_patient_history(patient_id)
        if not history:
            return [], []
        
        dates = [record['timestamp'] for record in history]
        probabilities = [record['probability'] for record in history]
        return dates, probabilities
    
    def export_patient_report(self, patient_id):
        """Export patient history as DataFrame"""
        history = self.get_patient_history(patient_id)
        if not history:
            return None
        
        df = pd.DataFrame(history)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
