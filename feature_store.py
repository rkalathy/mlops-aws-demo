#!/usr/bin/env python3
"""
SageMaker Feature Store implementation for MLOps demo
"""
import boto3
import pandas as pd
import numpy as np
from datetime import datetime
import time
from sagemaker.feature_store.feature_group import FeatureGroup
from sagemaker.feature_store.feature_definition import FeatureDefinition, FeatureTypeEnum
from sagemaker import Session
import json
import os

class MLOpsFeatureStore:
    def __init__(self, role_arn, bucket_name, region_name='eu-west-2'):
        self.role_arn = role_arn
        self.bucket_name = bucket_name
        self.region_name = region_name
        self.session = Session(default_bucket=bucket_name, boto_session=boto3.Session(region_name=region_name))
        self.feature_store_session = self.session.boto_session.client('sagemaker-featurestore-runtime', region_name=region_name)
        
    def create_feature_group(self, feature_group_name, feature_definitions, record_identifier_name, event_time_name):
        """Create a feature group in SageMaker Feature Store"""
        feature_group = FeatureGroup(
            name=feature_group_name,
            sagemaker_session=self.session
        )
        
        try:
            # Set feature definitions first
            feature_group.feature_definitions = feature_definitions
            
            # Create feature group
            feature_group.create(
                s3_uri=f's3://{self.bucket_name}/feature-store/{feature_group_name}',
                record_identifier_name=record_identifier_name,
                event_time_feature_name=event_time_name,
                role_arn=self.role_arn,
                enable_online_store=True
            )
            
            # Wait for feature group to be created
            print(f"Creating feature group {feature_group_name}...")
            feature_group.wait_for_feature_group_creation_complete()
            print(f"Feature group {feature_group_name} created successfully!")
            
        except Exception as e:
            if "already exists" in str(e):
                print(f"Feature group {feature_group_name} already exists")
            else:
                print(f"Error creating feature group: {e}")
                raise
        
        return feature_group
    
    def prepare_training_features(self):
        """Prepare feature definitions for training data"""
        feature_definitions = []
        
        # Add feature columns
        for i in range(10):  # 10 features as per create_sample_data.py
            feature_definitions.append(
                FeatureDefinition(
                    feature_name=f'feature_{i}',
                    feature_type=FeatureTypeEnum.FRACTIONAL
                )
            )
        
        # Add target column
        feature_definitions.append(
            FeatureDefinition(
                feature_name='target',
                feature_type=FeatureTypeEnum.INTEGRAL
            )
        )
        
        # Add metadata columns
        feature_definitions.extend([
            FeatureDefinition(feature_name='record_id', feature_type=FeatureTypeEnum.STRING),
            FeatureDefinition(feature_name='event_time', feature_type=FeatureTypeEnum.FRACTIONAL),
            FeatureDefinition(feature_name='data_split', feature_type=FeatureTypeEnum.STRING)
        ])
        
        return feature_definitions
    
    def prepare_prediction_features(self):
        """Prepare feature definitions for real-time prediction data"""
        feature_definitions = []
        
        # Add feature columns (no target for prediction data)
        for i in range(10):
            feature_definitions.append(
                FeatureDefinition(
                    feature_name=f'feature_{i}',
                    feature_type=FeatureTypeEnum.FRACTIONAL
                )
            )
        
        # Add metadata columns
        feature_definitions.extend([
            FeatureDefinition(feature_name='record_id', feature_type=FeatureTypeEnum.STRING),
            FeatureDefinition(feature_name='event_time', feature_type=FeatureTypeEnum.FRACTIONAL),
            FeatureDefinition(feature_name='source', feature_type=FeatureTypeEnum.STRING)
        ])
        
        return feature_definitions
    
    def ingest_training_data(self, df, feature_group, data_split='train'):
        """Ingest training data into feature store"""
        # Add metadata columns
        df_copy = df.copy()
        df_copy['record_id'] = [f"{data_split}_{i}" for i in range(len(df_copy))]
        df_copy['event_time'] = time.time()
        df_copy['data_split'] = data_split
        
        # Ingest data
        feature_group.ingest(data_frame=df_copy, max_workers=3, wait=True)
        print(f"Ingested {len(df_copy)} {data_split} records into feature store")
        
        return df_copy
    
    def ingest_prediction_data(self, data_array, feature_group):
        """Ingest real-time prediction data into feature store"""
        # Convert array to DataFrame
        feature_names = [f'feature_{i}' for i in range(len(data_array[0]))]
        df = pd.DataFrame(data_array, columns=feature_names)
        
        # Add metadata columns
        df['record_id'] = [f"pred_{i}_{int(time.time())}" for i in range(len(df))]
        df['event_time'] = time.time()
        df['source'] = 'real_time_prediction'
        
        # Ingest data
        feature_group.ingest(data_frame=df, max_workers=3, wait=True)
        print(f"Ingested {len(df)} prediction records into feature store")
        
        return df
    
    def get_training_data(self, feature_group_name, data_split=None):
        """Retrieve training data from feature store"""
        try:
            # Build query
            query = f"""
            SELECT * FROM "{feature_group_name}"
            """
            
            if data_split:
                query += f" WHERE data_split = '{data_split}'"
            
            # Execute query using Athena
            query_execution = self.session.boto_session.client('athena', region_name=self.region_name)
            
            # For demo purposes, we'll use the offline store S3 path
            s3_path = f's3://{self.bucket_name}/feature-store/{feature_group_name}/offline'
            print(f"Training data available at: {s3_path}")
            
            return s3_path
            
        except Exception as e:
            print(f"Error retrieving training data: {e}")
            return None
    
    def get_online_features(self, feature_group_name, record_ids):
        """Retrieve features for online inference"""
        try:
            records = []
            for record_id in record_ids:
                response = self.feature_store_session.get_record(
                    FeatureGroupName=feature_group_name,
                    RecordIdentifierValueAsString=record_id
                )
                records.append(response['Record'])
            
            return records
            
        except Exception as e:
            print(f"Error retrieving online features: {e}")
            return None
    
    def setup_training_feature_store(self):
        """Set up feature store for training data"""
        feature_group_name = 'mlops-demo-training-features'
        
        # Prepare feature definitions
        feature_definitions = self.prepare_training_features()
        
        # Create feature group
        feature_group = self.create_feature_group(
            feature_group_name=feature_group_name,
            feature_definitions=feature_definitions,
            record_identifier_name='record_id',
            event_time_name='event_time'
        )
        
        return feature_group, feature_group_name
    
    def setup_prediction_feature_store(self):
        """Set up feature store for prediction data"""
        feature_group_name = 'mlops-demo-prediction-features'
        
        # Prepare feature definitions
        feature_definitions = self.prepare_prediction_features()
        
        # Create feature group
        feature_group = self.create_feature_group(
            feature_group_name=feature_group_name,
            feature_definitions=feature_definitions,
            record_identifier_name='record_id',
            event_time_name='event_time'
        )
        
        return feature_group, feature_group_name

def create_sample_data_with_feature_store(role_arn, bucket_name, region_name='eu-west-2'):
    """Create sample data and store it in feature store"""
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    
    # Generate synthetic data (same as create_sample_data.py)
    X, y = make_classification(
        n_samples=1000,
        n_features=10,
        n_informative=8,
        n_redundant=2,
        n_clusters_per_class=1,
        random_state=42
    )
    
    # Create feature names
    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    
    # Create DataFrame
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    
    # Split into train and test
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # Initialize feature store
    feature_store = MLOpsFeatureStore(role_arn, bucket_name, region_name)
    
    # Set up feature store
    feature_group, feature_group_name = feature_store.setup_training_feature_store()
    
    # Ingest training and test data
    feature_store.ingest_training_data(train_df, feature_group, 'train')
    feature_store.ingest_training_data(test_df, feature_group, 'test')
    
    # Also save locally for backward compatibility
    os.makedirs('data', exist_ok=True)
    train_df.to_csv('data/train.csv', index=False)
    test_df.to_csv('data/test.csv', index=False)
    
    print(f"Data ingested into feature store: {feature_group_name}")
    print(f"Training data shape: {train_df.shape}")
    print(f"Test data shape: {test_df.shape}")
    
    return feature_group_name, train_df, test_df

def get_prediction_data_from_feature_store():
    """Get hardcoded prediction data (replaces hardcoded test data in sagemaker_demo.py)"""
    # Return the same test data that was hardcoded in sagemaker_demo.py:140-143
    return [
        [0.5, -1.2, 0.8, -0.3, 1.1, -0.7, 0.2, 0.9, -0.4, 0.6],
        [-0.8, 1.5, -0.2, 0.7, -1.0, 0.3, -0.9, 0.1, 1.2, -0.5]
    ]

if __name__ == "__main__":
    # Example usage
    role_arn = os.environ.get('SAGEMAKER_ROLE_ARN')
    bucket_name = os.environ.get('S3_BUCKET_NAME')
    region_name = os.environ.get('AWS_DEFAULT_REGION', 'eu-west-2')
    
    if role_arn and bucket_name:
        feature_group_name, train_df, test_df = create_sample_data_with_feature_store(
            role_arn, bucket_name, region_name
        )
        print(f"Feature store setup complete: {feature_group_name}")
    else:
        print("Missing required environment variables: SAGEMAKER_ROLE_ARN, S3_BUCKET_NAME")