import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.svm import SVC
from xgboost import XGBClassifier
import joblib
from brain_tumor_preprocessing import load_and_preprocess_data
import warnings
warnings.filterwarnings('ignore')

def train_model(X, y):
    """Train an ensemble model with optimized hyperparameters"""
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Define base models
    rf = RandomForestClassifier(random_state=42)
    gb = GradientBoostingClassifier(random_state=42)
    xgb = XGBClassifier(random_state=42)
    
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
    
    xgb_params = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.3],
        'max_depth': [3, 5, 7],
        'min_child_weight': [1, 3, 5],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0]
    }
    
    # Optimize each model
    print("Optimizing Random Forest...")
    rf_search = RandomizedSearchCV(rf, rf_params, n_iter=20, cv=5, n_jobs=-1, random_state=42)
    rf_search.fit(X_train, y_train)
    best_rf = rf_search.best_estimator_
    
    print("Optimizing Gradient Boosting...")
    gb_search = RandomizedSearchCV(gb, gb_params, n_iter=20, cv=5, n_jobs=-1, random_state=42)
    gb_search.fit(X_train, y_train)
    best_gb = gb_search.best_estimator_
    
    print("Optimizing XGBoost...")
    xgb_search = RandomizedSearchCV(xgb, xgb_params, n_iter=20, cv=5, n_jobs=-1, random_state=42)
    xgb_search.fit(X_train, y_train)
    best_xgb = xgb_search.best_estimator_
    
    # Create voting classifier
    voting_clf = VotingClassifier(
        estimators=[
            ('rf', best_rf),
            ('gb', best_gb),
            ('xgb', best_xgb)
        ],
        voting='soft'
    )
    
    # Train final model
    print("Training final ensemble model...")
    voting_clf.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = voting_clf.predict(X_test)
    y_prob = voting_clf.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'auc_roc': roc_auc_score(y_test, y_prob)
    }
    
    print("\nModel Performance:")
    for metric, value in metrics.items():
        print(f"{metric.upper()}: {value:.4f}")
    
    # Cross-validation score
    cv_scores = cross_val_score(voting_clf, X, y, cv=5)
    print(f"\nCross-validation accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    return voting_clf, metrics

def save_model(model, scaler, file_path='brain_tumor_model.joblib'):
    """Save the trained model and scaler"""
    model_data = {
        'model': model,
        'scaler': scaler
    }
    joblib.dump(model_data, file_path)
    print(f"\nModel saved to {file_path}")

def load_model(file_path='brain_tumor_model.joblib'):
    """Load the trained model and scaler"""
    model_data = joblib.load(file_path)
    return model_data['model'], model_data['scaler']

def main():
    """Main function to train and save the model"""
    print("Loading and preprocessing data...")
    df, X, y = load_and_preprocess_data('Brain_Tumor_Prediction_Dataset.csv')
    
    print("\nTraining model...")
    model, metrics = train_model(X, y)
    
    print("\nSaving model...")
    save_model(model, StandardScaler())
    
    return metrics

if __name__ == "__main__":
    main()
