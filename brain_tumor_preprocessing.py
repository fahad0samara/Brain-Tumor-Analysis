import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
from scipy import stats

def load_and_preprocess_data(file_path):
    """Load and preprocess the brain tumor dataset with advanced feature engineering"""
    # Load data
    df = pd.read_csv(file_path)
    
    # Clean the data
    df = clean_data(df)
    
    # Create features
    df = engineer_features(df)
    
    # Prepare features for ML
    X, y = prepare_features(df)
    
    return df, X, y

def clean_data(df):
    """Clean the data by handling missing values and outliers using robust methods"""
    df = df.copy()
    
    # Fill missing values in categorical columns
    categorical_cols = ['Gender', 'Brain_Tumor_Present']
    for col in categorical_cols:
        if df[col].isnull().any():
            mode_value = df[col].mode()[0]
            df[col].fillna(mode_value, inplace=True)
    
    # Handle missing values in numerical columns
    numeric_cols = ['Age', 'Tumor_Size', 'Genetic_Risk', 'Survival_Rate(%)']
    
    for col in numeric_cols:
        if df[col].isnull().any():
            # Use median for initial filling
            median_value = df[col].median()
            df[col].fillna(median_value, inplace=True)
        
        # Handle outliers using Winsorization
        percentiles = np.percentile(df[col], [1, 99])
        df[col] = np.clip(df[col], percentiles[0], percentiles[1])
    
    return df

def engineer_features(df):
    """Create advanced features using domain knowledge and statistical methods"""
    df = df.copy()
    
    # Advanced Risk Score using weighted features
    df['Risk_Score'] = (
        df['Genetic_Risk'] / 10 * 0.35 +  # Normalize to 0-1 range
        df['Age'].clip(0, 100) / 100 * 0.25 +
        df['Tumor_Size'] / df['Tumor_Size'].max() * 0.25 +
        (100 - df['Survival_Rate(%)']) / 100 * 0.15
    )
    
    # Medical complexity with weighted factors
    df['Medical_Complexity'] = df.apply(lambda x: sum([
        x['Genetic_Risk'] > 7 * 1.5,  # High genetic risk (weighted)
        x['Age'] > 50 * 1.2,          # Advanced age (weighted)
        x['Tumor_Size'] > 3 * 1.3,    # Large tumor (weighted)
        x['Survival_Rate(%)'] < 50 * 1.4  # Low survival rate (weighted)
    ]), axis=1)
    
    # Create interaction features
    df['Age_Risk'] = df['Age'] * df['Genetic_Risk'] / 10
    df['Tumor_Severity'] = df['Tumor_Size'] * (100 - df['Survival_Rate(%)']) / 100
    
    return df

def prepare_features(df):
    """Prepare features for ML with advanced preprocessing"""
    df = df.copy()
    
    # Convert categorical variables to numeric
    gender_map = {'Female': 0, 'Male': 1, 'Other': 2}
    df['Gender'] = df['Gender'].map(gender_map)
    
    # Select features
    feature_cols = [
        'Age', 'Gender', 'Tumor_Size', 'Genetic_Risk', 'Survival_Rate(%)',
        'Risk_Score', 'Medical_Complexity', 'Age_Risk', 'Tumor_Severity'
    ]
    
    # Create X and y
    X = df[feature_cols]
    y = df['Brain_Tumor_Present'].map({'Yes': 1, 'No': 0})
    
    # Use RobustScaler for better handling of outliers
    scaler = RobustScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    
    return X, y
