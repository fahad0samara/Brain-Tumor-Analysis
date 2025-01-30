from fpdf import FPDF
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import base64
import io
import os

try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Plotly not available, will use matplotlib for visualizations")

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
        
    def generate_report(self, patient_data, output_file):
        """Generate a PDF report for a patient"""
        self.pdf.add_page()
        
        # Header
        self.pdf.set_font('Arial', 'B', 16)
        self.pdf.cell(0, 10, 'Brain Tumor Analysis Report', ln=True, align='C')
        self.pdf.ln(10)
        
        # Patient Information
        self.pdf.set_font('Arial', 'B', 12)
        self.pdf.cell(0, 10, 'Patient Information', ln=True)
        self.pdf.set_font('Arial', '', 12)
        
        info_items = [
            ('Age', patient_data['age']),
            ('Gender', patient_data['gender']),
            ('Tumor Size', f"{patient_data['tumor_size']:.1f} cm"),
            ('Genetic Risk', f"{patient_data['genetic_risk']}/10"),
            ('Survival Rate', f"{patient_data['survival_rate']}%")
        ]
        
        for label, value in info_items:
            self.pdf.cell(50, 10, label + ':', 0)
            self.pdf.cell(0, 10, str(value), ln=True)
        
        # Risk Assessment
        self.pdf.ln(10)
        self.pdf.set_font('Arial', 'B', 12)
        self.pdf.cell(0, 10, 'Risk Assessment', ln=True)
        self.pdf.set_font('Arial', '', 12)
        
        prediction = "High Risk" if patient_data['prediction'] == 1 else "Low Risk"
        probability = f"{patient_data['probability']*100:.1f}%"
        
        self.pdf.cell(50, 10, 'Prediction:', 0)
        self.pdf.cell(0, 10, prediction, ln=True)
        self.pdf.cell(50, 10, 'Probability:', 0)
        self.pdf.cell(0, 10, probability, ln=True)
        
        # Additional Factors
        self.pdf.ln(10)
        self.pdf.set_font('Arial', 'B', 12)
        self.pdf.cell(0, 10, 'Additional Risk Factors', ln=True)
        self.pdf.set_font('Arial', '', 12)
        
        factors = [
            ('Smoking', patient_data.get('smoking', False)),
            ('Alcohol', patient_data.get('alcohol', False)),
            ('Family History', patient_data.get('family_history', False))
        ]
        
        for factor, value in factors:
            self.pdf.cell(50, 10, factor + ':', 0)
            self.pdf.cell(0, 10, 'Yes' if value else 'No', ln=True)
        
        # Add visualization if trend data is available
        if 'trend_data' in patient_data:
            self._add_trend_visualization(patient_data['trend_data'])
        
        # Notes
        if patient_data.get('notes'):
            self.pdf.ln(10)
            self.pdf.set_font('Arial', 'B', 12)
            self.pdf.cell(0, 10, 'Notes', ln=True)
            self.pdf.set_font('Arial', '', 12)
            self.pdf.multi_cell(0, 10, patient_data['notes'])
        
        # Save the report
        self.pdf.output(output_file)
    
    def _add_trend_visualization(self, trend_data):
        """Add trend visualization using plotly or matplotlib"""
        if PLOTLY_AVAILABLE:
            self._add_plotly_visualization(trend_data)
        else:
            self._add_matplotlib_visualization(trend_data)
    
    def _add_plotly_visualization(self, trend_data):
        """Create visualization using plotly"""
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=trend_data['dates'],
            y=trend_data['values'],
            mode='lines+markers',
            name='Risk Trend'
        ))
        fig.update_layout(
            title='Risk Trend Over Time',
            xaxis_title='Date',
            yaxis_title='Risk Score'
        )
        # Save to temporary file
        temp_file = 'temp_plot.png'
        fig.write_image(temp_file)
        self.pdf.image(temp_file, x=10, w=190)
        import os
        os.remove(temp_file)
    
    def _add_matplotlib_visualization(self, trend_data):
        """Create visualization using matplotlib"""
        plt.figure(figsize=(10, 6))
        plt.plot(trend_data['dates'], trend_data['values'], 'o-')
        plt.title('Risk Trend Over Time')
        plt.xlabel('Date')
        plt.ylabel('Risk Score')
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        # Save to temporary file
        temp_file = 'temp_plot.png'
        plt.savefig(temp_file)
        plt.close()
        self.pdf.image(temp_file, x=10, w=190)
        import os
        os.remove(temp_file)

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
