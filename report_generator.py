import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from fpdf import FPDF
import os
from datetime import datetime
import base64
import io
import matplotlib.pyplot as plt
import seaborn as sns

class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'Brain Tumor Analysis Report', 0, 1, 'C')
        self.ln(10)
    
    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

class ReportGenerator:
    def __init__(self):
        self.pdf = PDF()
    
    def create_patient_report(self, patient_data, trend_data=None):
        """Create a detailed PDF report for a patient"""
        self.pdf.add_page()
        
        # Patient Information
        self.pdf.set_font('Arial', 'B', 12)
        self.pdf.cell(0, 10, 'Patient Information', 0, 1)
        self.pdf.set_font('Arial', '', 10)
        self.pdf.cell(0, 10, f"Name: {patient_data['name']}", 0, 1)
        self.pdf.cell(0, 10, f"Age: {patient_data['age']}", 0, 1)
        self.pdf.cell(0, 10, f"Gender: {patient_data['gender']}", 0, 1)
        
        # Clinical Measurements
        self.pdf.ln(5)
        self.pdf.set_font('Arial', 'B', 12)
        self.pdf.cell(0, 10, 'Clinical Measurements', 0, 1)
        self.pdf.set_font('Arial', '', 10)
        self.pdf.cell(0, 10, f"Tumor Size: {patient_data['tumor_size']} cm", 0, 1)
        self.pdf.cell(0, 10, f"Genetic Risk: {patient_data['genetic_risk']}", 0, 1)
        self.pdf.cell(0, 10, f"Survival Rate: {patient_data['survival_rate']}%", 0, 1)
        
        # Risk Factors
        self.pdf.ln(5)
        self.pdf.set_font('Arial', 'B', 12)
        self.pdf.cell(0, 10, 'Risk Factors', 0, 1)
        self.pdf.set_font('Arial', '', 10)
        self.pdf.cell(0, 10, f"Smoking: {'Yes' if patient_data['smoking'] else 'No'}", 0, 1)
        self.pdf.cell(0, 10, f"Alcohol: {'Yes' if patient_data['alcohol'] else 'No'}", 0, 1)
        self.pdf.cell(0, 10, f"Family History: {'Yes' if patient_data['family_history'] else 'No'}", 0, 1)
        
        # Prediction Results
        self.pdf.ln(5)
        self.pdf.set_font('Arial', 'B', 12)
        self.pdf.cell(0, 10, 'Prediction Results', 0, 1)
        self.pdf.set_font('Arial', '', 10)
        risk_level = 'High' if patient_data['prediction'] == 1 else 'Low'
        self.pdf.cell(0, 10, f"Risk Level: {risk_level}", 0, 1)
        self.pdf.cell(0, 10, f"Probability: {patient_data['probability']*100:.1f}%", 0, 1)
        
        # Add trend chart if available
        if trend_data and len(trend_data['dates']) > 1:
            plt.figure(figsize=(10, 4))
            plt.plot(trend_data['dates'], [p*100 for p in trend_data['probabilities']], 
                    marker='o')
            plt.title('Risk Probability Trend')
            plt.xlabel('Date')
            plt.ylabel('Risk Probability (%)')
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Save plot to memory
            img_stream = io.BytesIO()
            plt.savefig(img_stream, format='png')
            img_stream.seek(0)
            
            # Add to PDF
            self.pdf.ln(10)
            self.pdf.image(img_stream, x=10, w=190)
            plt.close()
        
        # Notes
        if patient_data.get('notes'):
            self.pdf.ln(5)
            self.pdf.set_font('Arial', 'B', 12)
            self.pdf.cell(0, 10, 'Additional Notes', 0, 1)
            self.pdf.set_font('Arial', '', 10)
            self.pdf.multi_cell(0, 10, patient_data['notes'])
        
        return self.pdf

def generate_comparison_report(patients_data):
    """Generate a comparison report for multiple patients"""
    pdf = PDF()
    pdf.add_page()
    
    # Create comparison table
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, 'Patient Comparison', 0, 1)
    
    # Table header
    pdf.set_font('Arial', 'B', 8)
    headers = ['Name', 'Age', 'Risk Level', 'Probability']
    col_width = 190 / len(headers)
    for header in headers:
        pdf.cell(col_width, 10, header, 1)
    pdf.ln()
    
    # Table data
    pdf.set_font('Arial', '', 8)
    for patient in patients_data:
        pdf.cell(col_width, 10, patient['name'], 1)
        pdf.cell(col_width, 10, str(patient['age']), 1)
        pdf.cell(col_width, 10, 'High' if patient['prediction'] == 1 else 'Low', 1)
        pdf.cell(col_width, 10, f"{patient['probability']*100:.1f}%", 1)
        pdf.ln()
    
    return pdf

def generate_statistical_analysis(patients_data):
    """Generate statistical analysis of patient data"""
    df = pd.DataFrame(patients_data)
    
    # Calculate statistics
    stats = {
        'avg_age': df['age'].mean(),
        'avg_tumor_size': df['tumor_size'].mean(),
        'avg_genetic_risk': df['genetic_risk'].mean(),
        'avg_survival_rate': df['survival_rate'].mean(),
        'high_risk_percentage': (df['prediction'] == 1).mean() * 100,
        'smoking_percentage': df['smoking'].mean() * 100,
        'alcohol_percentage': df['alcohol'].mean() * 100,
        'family_history_percentage': df['family_history'].mean() * 100
    }
    
    # Create correlation matrix
    numeric_cols = ['age', 'tumor_size', 'genetic_risk', 'survival_rate', 'probability']
    correlation = df[numeric_cols].corr()
    
    # Create heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0)
    plt.title('Feature Correlation Matrix')
    
    # Save plot to memory
    heatmap_stream = io.BytesIO()
    plt.savefig(heatmap_stream, format='png', bbox_inches='tight')
    heatmap_stream.seek(0)
    plt.close()
    
    return stats, heatmap_stream
