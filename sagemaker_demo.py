#!/usr/bin/env python3
"""
Main MLOps demo script for AWS SageMaker with MLflow tracking
This script orchestrates the entire pipeline: data creation, training, and deployment
"""
import boto3
import os
import time
import subprocess
import json
import shutil
from datetime import datetime
from sagemaker import Session
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.sklearn.model import SKLearnModel
import mlflow
import mlflow.sagemaker

class MLOpsDemo:
    def __init__(self, role_arn, bucket_name, region_name='eu-west-2', mlflow_tracking_uri=None):
        self.role_arn = role_arn
        self.bucket_name = bucket_name
        self.region_name = region_name
        self.session = Session(default_bucket=bucket_name, boto_session=boto3.Session(region_name=region_name))
        self.timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        
        # Initialize MLflow
        if mlflow_tracking_uri:
            mlflow.set_tracking_uri(mlflow_tracking_uri)
        else:
            # Use local file store
            os.makedirs('./mlruns', exist_ok=True)
            mlflow.set_tracking_uri('./mlruns')
        
        mlflow.set_experiment(f'sagemaker-demo-{self.timestamp}')
        print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")
        print(f"MLflow experiment: sagemaker-demo-{self.timestamp}")
        
    def create_sample_data(self):
        """Create and upload sample data using Feature Store"""
        print("Step 1: Creating sample data with Feature Store...")
        
        try:
            # Import feature store functionality
            from feature_store import create_sample_data_with_feature_store
            
            # Create data and ingest into feature store
            feature_group_name, train_df, test_df = create_sample_data_with_feature_store(
                self.role_arn, self.bucket_name, self.region_name
            )
            
            print("Sample data created and ingested into Feature Store successfully!")
            
            # Upload data to S3 (upload directory, not individual files) - for backward compatibility
            train_input = self.session.upload_data(
                path='data',
                bucket=self.bucket_name,
                key_prefix=f'demo-{self.timestamp}/data'
            )
            
            print(f"Data uploaded to S3:")
            print(f"  Data directory: {train_input}")
            print(f"  Feature Store Group: {feature_group_name}")
            
            return train_input
            
        except Exception as e:
            print(f"Error with Feature Store, falling back to original method: {e}")
            # Fallback to original method
            result = subprocess.run(['python', 'create_sample_data.py'], 
                                  capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"Error creating data: {result.stderr}")
                return False
                
            print("Sample data created successfully!")
            
            # Upload data to S3 (upload directory, not individual files)
            train_input = self.session.upload_data(
                path='data',
                bucket=self.bucket_name,
                key_prefix=f'demo-{self.timestamp}/data'
            )
            
            print(f"Data uploaded to S3:")
            print(f"  Data directory: {train_input}")
            
            return train_input
    
    def train_model(self, data_input):
        """Train model using SageMaker with MLflow tracking"""
        print("\nStep 2: Training model on SageMaker...")
        
        # Log experiment metadata to current run
        mlflow.log_param("sagemaker_instance_type", "ml.m5.large")
        mlflow.log_param("framework_version", "0.23-1")
        mlflow.log_param("training_data_path", data_input)
        mlflow.log_param("region", self.region_name)
        mlflow.log_param("timestamp", self.timestamp)
        
        # Create SKLearn estimator with MLflow parameters
        sklearn_estimator = SKLearn(
            entry_point='train.py',
            role=self.role_arn,
            instance_type='ml.m5.large',
            framework_version='0.23-1',
            py_version='py3',
            requirements='sagemaker_requirements.txt',
            hyperparameters={
                'n-estimators': 100,
                'max-depth': 10,
                'mlflow-experiment-name': f'sagemaker-demo-{self.timestamp}'
            },
            output_path=f's3://{self.bucket_name}/demo-{self.timestamp}/output',
            sagemaker_session=self.session,
            base_job_name=f'demo-training-{self.timestamp}'
        )
        
        # Start training job
        sklearn_estimator.fit({'training': data_input})
        
        # Log training job details
        training_job_name = sklearn_estimator.latest_training_job.name
        mlflow.log_param("sagemaker_training_job_name", training_job_name)
        
        # Log model artifacts path
        model_artifacts_path = sklearn_estimator.model_data
        mlflow.log_param("model_artifacts_s3_path", model_artifacts_path)
        
        print("Model training completed!")
        print(f"Training job name: {training_job_name}")
        print(f"Model artifacts: {model_artifacts_path}")
        
        return sklearn_estimator
    
    def deploy_model(self, estimator):
        """Deploy trained model to endpoint with MLflow tracking"""
        print("\nStep 3: Deploying model to endpoint...")
        
        endpoint_name = f'demo-endpoint-{self.timestamp}'
        
        # Log deployment parameters to current run
        mlflow.log_param("endpoint_name", endpoint_name)
        mlflow.log_param("endpoint_instance_type", "ml.t2.medium")
        mlflow.log_param("endpoint_instance_count", 1)
        
        # Deploy model
        predictor = estimator.deploy(
            initial_instance_count=1,
            instance_type='ml.t2.medium',
            endpoint_name=endpoint_name
        )
        
        # Log deployment success
        mlflow.log_metric("deployment_successful", 1)
        
        print(f"Model deployed to endpoint: {endpoint_name}")
        return predictor, endpoint_name
    
    def test_endpoint(self, predictor):
        """Test the deployed endpoint with MLflow tracking"""
        print("\nStep 4: Testing endpoint...")
        
        try:
            # Get test data from feature store
            from feature_store import get_prediction_data_from_feature_store, MLOpsFeatureStore
            
            # Initialize feature store for prediction data
            feature_store = MLOpsFeatureStore(self.role_arn, self.bucket_name, self.region_name)
            
            # Get sample test data
            test_data = get_prediction_data_from_feature_store()
            
            # Optionally store prediction request in feature store for tracking
            try:
                pred_feature_group, pred_group_name = feature_store.setup_prediction_feature_store()
                feature_store.ingest_prediction_data(test_data, pred_feature_group)
                print(f"Prediction request stored in Feature Store: {pred_group_name}")
            except Exception as e:
                print(f"Could not store prediction data in Feature Store: {e}")
                
        except Exception as e:
            print(f"Error accessing Feature Store for predictions, using fallback data: {e}")
            # Fallback to hardcoded data
            test_data = [
                [0.5, -1.2, 0.8, -0.3, 1.1, -0.7, 0.2, 0.9, -0.4, 0.6],
                [-0.8, 1.5, -0.2, 0.7, -1.0, 0.3, -0.9, 0.1, 1.2, -0.5]
            ]
        
        try:
            # Make predictions
            predictions = predictor.predict(test_data)
            
            # Log test results to current run
            mlflow.log_param("test_samples_count", len(test_data))
            mlflow.log_metric("endpoint_test_successful", 1)
            
            # Log predictions as JSON artifact
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump({
                    'test_data': test_data,
                    'predictions': predictions
                }, f, indent=2)
                mlflow.log_artifact(f.name, "predictions")
            
            print("Endpoint test successful!")
            print(f"Predictions: {predictions}")
            return True
            
        except Exception as e:
            mlflow.log_metric("endpoint_test_successful", 0)
            mlflow.log_param("error_message", str(e))
            print(f"Error testing endpoint: {e}")
            return False
    
    def cleanup(self, predictor, endpoint_name):
        """Clean up resources"""
        print(f"\nStep 5: Cleaning up resources...")
        
        try:
            # Delete endpoint
            predictor.delete_endpoint()
            print(f"Endpoint {endpoint_name} deleted")
        except Exception as e:
            print(f"Error during cleanup: {e}")
    
    def run_demo(self, cleanup_after=False):
        """Run the complete MLOps demo with MLflow tracking"""
        print(f"Starting MLOps Demo - {self.timestamp}")
        print("=" * 50)
        
        # End any existing runs
        try:
            mlflow.end_run()
        except:
            pass
        
        with mlflow.start_run(run_name=f"full-demo-{self.timestamp}"):
            # Log demo configuration
            mlflow.log_param("demo_timestamp", self.timestamp)
            mlflow.log_param("cleanup_after", cleanup_after)
            mlflow.log_param("aws_region", self.region_name)
            mlflow.log_param("s3_bucket", self.bucket_name)
            
            try:
                # Step 1: Create and upload data
                data_input = self.create_sample_data()
                
                # Step 2: Train model
                estimator = self.train_model(data_input)
                
                # Step 3: Deploy model
                predictor, endpoint_name = self.deploy_model(estimator)
                
                # Step 4: Test endpoint
                test_success = self.test_endpoint(predictor)
                
                # Log overall demo success
                mlflow.log_metric("demo_successful", 1)
                mlflow.log_metric("endpoint_test_passed", 1 if test_success else 0)
                
                print(f"\n🎉 Demo completed successfully!")
                print(f"Endpoint name: {endpoint_name}")
                print(f"You can now use this endpoint for predictions.")
                print(f"MLflow UI: Run 'mlflow ui' to view experiment results")
                
                # Step 5: Optional cleanup
                if cleanup_after:
                    self.cleanup(predictor, endpoint_name)
                    mlflow.log_param("resources_cleaned_up", True)
                else:
                    print(f"\nNote: Endpoint {endpoint_name} is still running.")
                    print("Remember to delete it manually to avoid charges.")
                    mlflow.log_param("resources_cleaned_up", False)
                
                return True
                
            except Exception as e:
                mlflow.log_metric("demo_successful", 0)
                mlflow.log_param("error_message", str(e))
                print(f"\n❌ Demo failed: {e}")
                return False

def main():
    """Main function"""
    # Get configuration from environment variables
    role_arn = os.environ.get('SAGEMAKER_ROLE_ARN')
    bucket_name = os.environ.get('S3_BUCKET_NAME')
    region_name = os.environ.get('AWS_DEFAULT_REGION', 'eu-west-2')
    cleanup_after = os.environ.get('CLEANUP_AFTER_DEMO', 'false').lower() == 'true'
    mlflow_tracking_uri = os.environ.get('MLFLOW_TRACKING_URI')
    
    # Validate required environment variables
    if not role_arn:
        print("❌ SAGEMAKER_ROLE_ARN environment variable is required")
        return False
    
    if not bucket_name:
        print("❌ S3_BUCKET_NAME environment variable is required")
        return False
    
    print(f"Configuration:")
    print(f"  Role ARN: {role_arn}")
    print(f"  S3 Bucket: {bucket_name}")
    print(f"  Region: {region_name}")
    print(f"  Cleanup after demo: {cleanup_after}")
    print(f"  MLflow tracking URI: {mlflow_tracking_uri or 'Local file store'}")
    print()
    
    # Run the demo
    demo = MLOpsDemo(role_arn, bucket_name, region_name, mlflow_tracking_uri)
    success = demo.run_demo(cleanup_after)
    
    return success

if __name__ == '__main__':
    try:
        success = main()
        exit(0 if success else 1)
    finally:
        # Ensure MLflow run is ended
        try:
            mlflow.end_run()
        except:
            pass