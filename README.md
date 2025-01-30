# Brain Tumor Prediction App 🧠

An advanced machine learning application for predicting brain tumor risks using patient data and clinical measurements.

## Features 🌟

- **Advanced Prediction Model**
  - Ensemble learning (Random Forest, XGBoost, Gradient Boosting)
  - Hyperparameter optimization
  - Cross-validation
  - Feature importance analysis

- **Patient Management**
  - Patient history tracking
  - Risk trend analysis
  - Statistical insights
  - Patient comparison

- **Analysis & Reporting**
  - PDF report generation
  - Statistical analysis
  - Risk factor visualization
  - Comparative analysis

## Installation 🚀

1. Clone the repository:
```bash
git clone https://github.com/yourusername/brain-tumor-prediction.git
cd brain-tumor-prediction
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage 💻

1. Run the Streamlit app:
```bash
python -m streamlit run app.py
```

2. Enter patient information:
   - Age
   - Gender
   - Tumor Size
   - Genetic Risk Score
   - Survival Rate
   - Additional factors

3. View predictions and analysis in the different tabs:
   - Patient Input
   - Model Insights
   - Patient History
   - Analysis & Comparison
   - About

## Model Details 🤖

The prediction model uses an ensemble of:
- Random Forest Classifier
- XGBoost Classifier
- Gradient Boosting Classifier

Features include:
- Age and gender
- Tumor measurements
- Genetic risk factors
- Survival rates
- Medical complexity scores
- Interaction features

## Requirements 📋

- Python 3.8+
- scikit-learn
- pandas
- numpy
- streamlit
- xgboost
- fpdf
- plotly
- seaborn
- matplotlib

## Project Structure 📁

```
brain-tumor-prediction/
├── app.py                    # Main Streamlit application
├── brain_tumor_analysis.py   # Model training and analysis
├── brain_tumor_preprocessing.py  # Data preprocessing
├── patient_history.py        # Patient data management
├── report_generator.py       # PDF report generation
├── requirements.txt          # Project dependencies
└── README.md                # Project documentation
```

## Contributing 🤝

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License 📄

This project is licensed under the MIT License - see the LICENSE file for details.
