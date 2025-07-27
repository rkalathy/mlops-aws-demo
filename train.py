#!/usr/bin/env python3
"""
SageMaker training script for binary classification with optional MLflow tracking
"""
import argparse
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
import joblib
import json
import tempfile

# Try to import MLflow, but make it optional
try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
    print("MLflow available for experiment tracking")
except ImportError:
    MLFLOW_AVAILABLE = False
    print("MLflow not available - running without experiment tracking")

def parse_args():
    parser = argparse.ArgumentParser()
    
    # SageMaker specific arguments
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--training', type=str, default=os.environ.get('SM_CHANNEL_TRAINING'))
    
    # Hyperparameters
    parser.add_argument('--n-estimators', type=int, default=100)
    parser.add_argument('--max-depth', type=int, default=10)
    parser.add_argument('--random-state', type=int, default=42)
    
    # MLflow parameters
    parser.add_argument('--mlflow-tracking-uri', type=str, default=None)
    parser.add_argument('--mlflow-experiment-name', type=str, default='sagemaker-demo')
    
    return parser.parse_args()

def load_data(training_path):
    """Load training and test data"""
    train_df = pd.read_csv(os.path.join(training_path, 'train.csv'))
    test_df = pd.read_csv(os.path.join(training_path, 'test.csv'))
    
    # Separate features and target
    X_train = train_df.drop('target', axis=1)
    y_train = train_df['target']
    X_test = test_df.drop('target', axis=1)
    y_test = test_df['target']
    
    return X_train, y_train, X_test, y_test

def train_model(X_train, y_train, X_test, y_test, args):
    """Train the model with optional MLflow tracking"""
    
    # Configure MLflow if available
    mlflow_context = None
    mlflow_enabled = MLFLOW_AVAILABLE
    
    if mlflow_enabled:
        try:
            if args.mlflow_tracking_uri:
                mlflow.set_tracking_uri(args.mlflow_tracking_uri)
            
            mlflow.set_experiment(args.mlflow_experiment_name)
            mlflow_context = mlflow.start_run(run_name=f"training-job-{args.mlflow_experiment_name}")
            print("MLflow tracking initialized")
        except Exception as e:
            print(f"MLflow initialization failed: {e}, continuing without tracking")
            mlflow_enabled = False
    
    try:
        # Log hyperparameters if MLflow is available
        if mlflow_enabled:
            mlflow.log_param("n_estimators", args.n_estimators)
            mlflow.log_param("max_depth", args.max_depth)
            mlflow.log_param("random_state", args.random_state)
            mlflow.log_param("train_samples", len(X_train))
            mlflow.log_param("test_samples", len(X_test))
            mlflow.log_param("n_features", X_train.shape[1])
        
        # Create and train model
        model = RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            random_state=args.random_state
        )
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Make predictions
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        
        # Calculate comprehensive metrics
        train_accuracy = accuracy_score(y_train, train_pred)
        test_accuracy = accuracy_score(y_test, test_pred)
        train_f1 = f1_score(y_train, train_pred)
        test_f1 = f1_score(y_test, test_pred)
        train_precision = precision_score(y_train, train_pred)
        test_precision = precision_score(y_test, test_pred)
        train_recall = recall_score(y_train, train_pred)
        test_recall = recall_score(y_test, test_pred)
        
        # Log metrics to MLflow if available
        if mlflow_enabled:
            mlflow.log_metric("train_accuracy", train_accuracy)
            mlflow.log_metric("test_accuracy", test_accuracy)
            mlflow.log_metric("train_f1", train_f1)
            mlflow.log_metric("test_f1", test_f1)
            mlflow.log_metric("train_precision", train_precision)
            mlflow.log_metric("test_precision", test_precision)
            mlflow.log_metric("train_recall", train_recall)
            mlflow.log_metric("test_recall", test_recall)
        
        # Log feature importance if MLflow is available
        feature_names = [f'feature_{i}' for i in range(X_train.shape[1])]
        feature_importance = dict(zip(feature_names, model.feature_importances_))
        
        if mlflow_enabled:
            for feature, importance in feature_importance.items():
                mlflow.log_metric(f"importance_{feature}", importance)
            
            # Log model
            mlflow.sklearn.log_model(
                model, 
                "model",
                registered_model_name=f"sagemaker-demo-model",
                input_example=X_test[:5].values if hasattr(X_test, 'values') else X_test[:5]
            )
            
            # Log classification report as artifact
            class_report = classification_report(y_test, test_pred, output_dict=True)
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(class_report, f, indent=2)
                mlflow.log_artifact(f.name, "reports")
        
        # Compile metrics for return
        metrics = {
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'train_f1': train_f1,
            'test_f1': test_f1,
            'train_precision': train_precision,
            'test_precision': test_precision,
            'train_recall': train_recall,
            'test_recall': test_recall,
            'n_estimators': args.n_estimators,
            'max_depth': args.max_depth,
            'feature_importance': feature_importance
        }
        
        print(f"Training Accuracy: {train_accuracy:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Test F1 Score: {test_f1:.4f}")
        print(f"Test Precision: {test_precision:.4f}")
        print(f"Test Recall: {test_recall:.4f}")
        print(f"Classification Report:\n{classification_report(y_test, test_pred)}")
        
        # Get MLflow run info if available
        if mlflow_enabled:
            run = mlflow.active_run()
            if run:
                print(f"MLflow Run ID: {run.info.run_id}")
                metrics['mlflow_run_id'] = run.info.run_id
                metrics['mlflow_experiment_id'] = run.info.experiment_id
        
        return model, metrics
        
    finally:
        # End MLflow run if it was started
        if mlflow_enabled and mlflow_context:
            try:
                mlflow.end_run()
            except:
                pass

def save_model(model, metrics, model_dir):
    """Save the trained model and metrics"""
    # Save the model
    joblib.dump(model, os.path.join(model_dir, 'model.joblib'))
    
    # Save metrics
    with open(os.path.join(model_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Model saved to {model_dir}")

def model_fn(model_dir):
    """Load model for inference (SageMaker inference)"""
    model = joblib.load(os.path.join(model_dir, 'model.joblib'))
    return model

def input_fn(request_body, request_content_type):
    """Parse input data for inference"""
    if request_content_type == 'application/json':
        input_data = json.loads(request_body)
        return np.array(input_data['instances'])
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model):
    """Make predictions"""
    predictions = model.predict(input_data)
    probabilities = model.predict_proba(input_data)
    
    return {
        'predictions': predictions.tolist(),
        'probabilities': probabilities.tolist()
    }

def output_fn(prediction, accept):
    """Format prediction output"""
    if accept == 'application/json':
        return json.dumps(prediction), accept
    else:
        raise ValueError(f"Unsupported accept type: {accept}")

if __name__ == '__main__':
    args = parse_args()
    
    # Load data
    X_train, y_train, X_test, y_test = load_data(args.training)
    
    # Train model
    model, metrics = train_model(X_train, y_train, X_test, y_test, args)
    
    # Save model
    save_model(model, metrics, args.model_dir)