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
    
    print("Initial Missing Values:")
    print(df.isnull().sum())
    
    # Fill missing values in categorical columns
    categorical_cols = ['Treatment_Received', 'Gender', 'Brain_Tumor_Present']
    for col in categorical_cols:
        if df[col].isnull().any():
            mode_value = df[col].mode()[0]
            df[col].fillna(mode_value, inplace=True)
            print(f"\nFilled {col} missing values with mode: {mode_value}")
    
    # Handle missing values in numerical columns using iterative imputation
    numeric_cols = ['Age', 'Tumor_Size', 'Genetic_Risk', 'Survival_Rate(%)']
    
    for col in numeric_cols:
        print(f"\nColumn: {col}")
        print(f"Missing values: {df[col].isnull().sum()}")
        
        if df[col].isnull().any():
            # Use median for initial filling
            median_value = df[col].median()
            df[col].fillna(median_value, inplace=True)
            print(f"Filled missing values with median: {median_value:.2f}")
        
        # Handle outliers using Winsorization
        percentiles = np.percentile(df[col], [1, 99])
        df[col] = np.clip(df[col], percentiles[0], percentiles[1])
        print(f"Winsorized at 1st and 99th percentiles")
    
    # Final check for missing values
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print("\nWarning: Still have missing values:")
        print(missing[missing > 0])
    else:
        print("\nSuccess: No missing values remain in the dataset!")
    
    return df

def engineer_features(df):
    """Create advanced features using domain knowledge and statistical methods"""
    df = df.copy()
    
    # Advanced Risk Score using weighted features
    df['Risk_Score'] = (
        df['Genetic_Risk'] * 0.35 +  # Increased genetic risk weight
        df['Age'].clip(0, 100) / 100 * 0.25 +
        df['Tumor_Size'] / df['Tumor_Size'].max() * 0.25 +
        (100 - df['Survival_Rate(%)']) / 100 * 0.15  # Added survival rate impact
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
    
    # Create polynomial features for key metrics
    df['Genetic_Risk_Squared'] = df['Genetic_Risk'] ** 2
    df['Tumor_Size_Squared'] = df['Tumor_Size'] ** 2
    
    # Create ratio features
    df['Risk_to_Survival_Ratio'] = df['Genetic_Risk'] / df['Survival_Rate(%)'].clip(1)
    df['Age_Adjusted_Risk'] = df['Genetic_Risk'] * (df['Age'] / 50)  # Normalized by average age
    
    # Create binned features
    df['Age_Group'] = pd.qcut(df['Age'], q=5, labels=['VeryYoung', 'Young', 'Middle', 'Senior', 'Elderly'])
    df['Risk_Level'] = pd.qcut(df['Genetic_Risk'], q=4, labels=['Low', 'Medium', 'High', 'VeryHigh'])
    
    # Verify no missing values in new features
    if df[['Risk_Score', 'Medical_Complexity', 'Age_Risk', 'Tumor_Severity']].isnull().sum().sum() > 0:
        raise ValueError("New features contain missing values!")
    
    return df

def prepare_features(df):
    """Prepare features for ML with advanced preprocessing"""
    df = df.copy()
    
    # Convert categorical variables to numeric using advanced encoding
    gender_map = {'Female': 0, 'Male': 1, 'Other': 2}
    df['Gender'] = df['Gender'].map(gender_map)
    
    # Create dummy variables for binned features
    df = pd.get_dummies(df, columns=['Age_Group', 'Risk_Level'], drop_first=True)
    
    # Select features including new engineered features
    feature_cols = [
        'Age', 'Gender', 'Tumor_Size', 'Genetic_Risk', 'Survival_Rate(%)',
        'Risk_Score', 'Medical_Complexity', 'Age_Risk', 'Tumor_Severity',
        'Genetic_Risk_Squared', 'Tumor_Size_Squared', 'Risk_to_Survival_Ratio',
        'Age_Adjusted_Risk'
    ] + [col for col in df.columns if col.startswith(('Age_Group_', 'Risk_Level_'))]
    
    # Create X and y
    X = df[feature_cols]
    y = df['Brain_Tumor_Present'].map({'Yes': 1, 'No': 0})
    
    # Use RobustScaler for better handling of outliers
    scaler = RobustScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    
    # Verify no missing values
    if X.isnull().sum().sum() > 0:
        raise ValueError("Features contain missing values!")
    if y.isnull().sum() > 0:
        raise ValueError("Target contains missing values!")
    
    return X, y
