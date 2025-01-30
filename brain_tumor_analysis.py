import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.svm import SVC
import joblib
from brain_tumor_preprocessing import load_and_preprocess_data
import warnings
warnings.filterwarnings('ignore')

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not available, will use alternative model")

def train_model(X, y):
    """Train an ensemble model with optimized hyperparameters"""
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Define base models
    rf = RandomForestClassifier(random_state=42)
    gb = GradientBoostingClassifier(random_state=42)
    
    # Define hyperparameter search spaces
    rf_params = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None]
    }
    
    gb_params = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.3],
        'max_depth': [3, 5, 7],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    # Optimize Random Forest
    rf_search = RandomizedSearchCV(rf, rf_params, n_iter=10, cv=5, random_state=42)
    rf_search.fit(X_train, y_train)
    rf_best = rf_search.best_estimator_
    
    # Optimize Gradient Boosting
    gb_search = RandomizedSearchCV(gb, gb_params, n_iter=10, cv=5, random_state=42)
    gb_search.fit(X_train, y_train)
    gb_best = gb_search.best_estimator_
    
    # Create model list
    estimators = [('rf', rf_best), ('gb', gb_best)]
    
    # Add XGBoost if available
    if XGBOOST_AVAILABLE:
        xgb = XGBClassifier(random_state=42)
        xgb_params = {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.1, 0.3],
            'max_depth': [3, 5, 7],
            'min_child_weight': [1, 3, 5],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0]
        }
        xgb_search = RandomizedSearchCV(xgb, xgb_params, n_iter=10, cv=5, random_state=42)
        xgb_search.fit(X_train, y_train)
        xgb_best = xgb_search.best_estimator_
        estimators.append(('xgb', xgb_best))
    
    # Create and train voting classifier
    model = VotingClassifier(estimators=estimators, voting='soft')
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'auc_roc': roc_auc_score(y_test, y_prob)
    }
    
    return model, metrics

def save_model(model, scaler, file_path='brain_tumor_model.joblib'):
    """Save the trained model and scaler"""
    joblib.dump({'model': model, 'scaler': scaler}, file_path)
    print(f"Model saved to {file_path}")

def load_model(file_path='brain_tumor_model.joblib'):
    """Load the trained model and scaler"""
    data = joblib.load(file_path)
    return data['model'], data['scaler']

def main():
    """Main function to train and save the model"""
    # Load and preprocess data
    df, X, y = load_and_preprocess_data('brain_tumor_data.csv')
    
    # Train model
    model, metrics = train_model(X, y)
    
    # Print metrics
    print("\nModel Performance:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

if __name__ == "__main__":
    main()
