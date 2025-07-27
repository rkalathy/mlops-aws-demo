#!/usr/bin/env python3
"""
Data Drift Monitoring for MLOps Pipeline
Monitors statistical drift in features and target distribution
"""
import boto3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
from scipy import stats
from sklearn.metrics import ks_2samp
import warnings
from typing import Dict, List, Tuple, Optional
import mlflow
import mlflow.tracking


class DataDriftMonitor:
    def __init__(self, role_arn, bucket_name, region_name='eu-west-2', mlflow_tracking_uri=None):
        self.role_arn = role_arn
        self.bucket_name = bucket_name
        self.region_name = region_name
        self.s3_client = boto3.client('s3', region_name=region_name)
        
        # Initialize MLflow
        if mlflow_tracking_uri:
            mlflow.set_tracking_uri(mlflow_tracking_uri)
        else:
            os.makedirs('./mlruns', exist_ok=True)
            mlflow.set_tracking_uri('./mlruns')
        
        # Set experiment for drift monitoring
        mlflow.set_experiment('data-drift-monitoring')
        
    def load_reference_data(self, reference_data_path: str) -> pd.DataFrame:
        """Load reference dataset (training data) for comparison"""
        try:
            if reference_data_path.startswith('s3://'):
                # Parse S3 path
                s3_parts = reference_data_path.replace('s3://', '').split('/')
                bucket = s3_parts[0]
                key = '/'.join(s3_parts[1:]) + '/train.csv'
                
                # Download from S3
                obj = self.s3_client.get_object(Bucket=bucket, Key=key)
                reference_df = pd.read_csv(obj['Body'])
            else:
                # Local file
                reference_df = pd.read_csv(os.path.join(reference_data_path, 'train.csv'))
            
            print(f"Loaded reference data: {reference_df.shape}")
            return reference_df
            
        except Exception as e:
            print(f"Error loading reference data: {e}")
            # Generate synthetic reference data for demo
            return self._generate_synthetic_reference_data()
    
    def _generate_synthetic_reference_data(self) -> pd.DataFrame:
        """Generate synthetic reference data matching original training distribution"""
        from sklearn.datasets import make_classification
        
        X, y = make_classification(
            n_samples=800,  # Training portion of original 1000 samples
            n_features=10,
            n_informative=8,
            n_redundant=2,
            n_clusters_per_class=1,
            random_state=42
        )
        
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        df = pd.DataFrame(X, columns=feature_names)
        df['target'] = y
        
        print("Generated synthetic reference data for drift monitoring")
        return df
    
    def generate_current_data(self, drift_type='none', drift_magnitude=0.0) -> pd.DataFrame:
        """Generate current data with optional synthetic drift for testing"""
        from sklearn.datasets import make_classification
        
        if drift_type == 'none':
            # No drift - same distribution as reference
            X, y = make_classification(
                n_samples=200,
                n_features=10,
                n_informative=8,
                n_redundant=2,
                n_clusters_per_class=1,
                random_state=123  # Different seed but same parameters
            )
        elif drift_type == 'covariate':
            # Covariate drift - feature distributions change
            X, y = make_classification(
                n_samples=200,
                n_features=10,
                n_informative=8,
                n_redundant=2,
                n_clusters_per_class=1,
                random_state=456
            )
            # Add systematic shift to features
            X = X + drift_magnitude
            
        elif drift_type == 'concept':
            # Concept drift - relationship between features and target changes
            X, y = make_classification(
                n_samples=200,
                n_features=10,
                n_informative=8,
                n_redundant=2,
                n_clusters_per_class=1,
                random_state=42
            )
            # Flip some labels to simulate concept drift
            flip_indices = np.random.choice(len(y), int(len(y) * drift_magnitude), replace=False)
            y[flip_indices] = 1 - y[flip_indices]
            
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        df = pd.DataFrame(X, columns=feature_names)
        df['target'] = y
        
        return df
    
    def calculate_statistical_drift(self, reference_data: pd.DataFrame, current_data: pd.DataFrame) -> Dict:
        """Calculate statistical drift metrics for each feature"""
        drift_results = {}
        feature_columns = [col for col in reference_data.columns if col.startswith('feature_')]
        
        for feature in feature_columns:
            ref_values = reference_data[feature].values
            curr_values = current_data[feature].values
            
            # Kolmogorov-Smirnov test
            ks_statistic, ks_p_value = ks_2samp(ref_values, curr_values)
            
            # Population Stability Index (PSI)
            psi_score = self._calculate_psi(ref_values, curr_values)
            
            # Statistical measures
            ref_mean, curr_mean = np.mean(ref_values), np.mean(curr_values)
            ref_std, curr_std = np.std(ref_values), np.std(curr_values)
            
            # Drift magnitude (relative change in mean)
            mean_drift = abs(curr_mean - ref_mean) / (abs(ref_mean) + 1e-8)
            std_drift = abs(curr_std - ref_std) / (abs(ref_std) + 1e-8)
            
            drift_results[feature] = {
                'ks_statistic': ks_statistic,
                'ks_p_value': ks_p_value,
                'psi_score': psi_score,
                'ref_mean': ref_mean,
                'curr_mean': curr_mean,
                'ref_std': ref_std,
                'curr_std': curr_std,
                'mean_drift': mean_drift,
                'std_drift': std_drift,
                'drift_detected': ks_p_value < 0.05 or psi_score > 0.2
            }
        
        return drift_results
    
    def _calculate_psi(self, reference: np.ndarray, current: np.ndarray, buckets: int = 10) -> float:
        """Calculate Population Stability Index (PSI)"""
        try:
            # Create bins based on reference data
            _, bin_edges = np.histogram(reference, bins=buckets)
            
            # Calculate frequencies
            ref_freq, _ = np.histogram(reference, bins=bin_edges)
            curr_freq, _ = np.histogram(current, bins=bin_edges)
            
            # Convert to percentages and add small epsilon to avoid log(0)
            ref_perc = (ref_freq + 1e-8) / (len(reference) + buckets * 1e-8)
            curr_perc = (curr_freq + 1e-8) / (len(current) + buckets * 1e-8)
            
            # Calculate PSI
            psi = np.sum((curr_perc - ref_perc) * np.log(curr_perc / ref_perc))
            
            return psi
            
        except Exception as e:
            print(f"Error calculating PSI: {e}")
            return 0.0
    
    def calculate_target_drift(self, reference_data: pd.DataFrame, current_data: pd.DataFrame) -> Dict:
        """Calculate drift in target distribution"""
        if 'target' not in reference_data.columns or 'target' not in current_data.columns:
            return {'target_drift_detected': False, 'error': 'Target column not found'}
        
        ref_target = reference_data['target'].values
        curr_target = current_data['target'].values
        
        # Chi-square test for categorical target
        ref_counts = np.bincount(ref_target)
        curr_counts = np.bincount(curr_target, minlength=len(ref_counts))
        
        try:
            chi2_stat, chi2_p_value = stats.chisquare(curr_counts, ref_counts)
        except ValueError:
            chi2_stat, chi2_p_value = 0.0, 1.0
        
        # Class distribution comparison
        ref_class_dist = ref_counts / len(ref_target)
        curr_class_dist = curr_counts / len(curr_target)
        
        return {
            'chi2_statistic': chi2_stat,
            'chi2_p_value': chi2_p_value,
            'ref_class_distribution': ref_class_dist.tolist(),
            'curr_class_distribution': curr_class_dist.tolist(),
            'target_drift_detected': chi2_p_value < 0.05
        }
    
    def generate_drift_report(self, feature_drift: Dict, target_drift: Dict, 
                            current_data: pd.DataFrame) -> Dict:
        """Generate comprehensive drift monitoring report"""
        timestamp = datetime.now().isoformat()
        
        # Overall drift summary
        features_with_drift = [f for f, metrics in feature_drift.items() 
                             if metrics.get('drift_detected', False)]
        
        drift_summary = {
            'timestamp': timestamp,
            'total_features': len(feature_drift),
            'features_with_drift': len(features_with_drift),
            'drift_percentage': len(features_with_drift) / len(feature_drift) * 100,
            'target_drift_detected': target_drift.get('target_drift_detected', False),
            'sample_size': len(current_data),
            'overall_drift_status': 'DRIFT_DETECTED' if (features_with_drift or target_drift.get('target_drift_detected', False)) else 'NO_DRIFT'
        }
        
        # Feature-level details
        feature_summary = {}
        for feature, metrics in feature_drift.items():
            feature_summary[feature] = {
                'drift_detected': metrics['drift_detected'],
                'ks_p_value': metrics['ks_p_value'],
                'psi_score': metrics['psi_score'],
                'mean_drift': metrics['mean_drift'],
                'severity': self._classify_drift_severity(metrics)
            }
        
        report = {
            'drift_summary': drift_summary,
            'feature_drift': feature_summary,
            'target_drift': target_drift,
            'detailed_metrics': feature_drift
        }
        
        return report
    
    def _classify_drift_severity(self, metrics: Dict) -> str:
        """Classify drift severity based on metrics"""
        psi_score = metrics.get('psi_score', 0)
        mean_drift = metrics.get('mean_drift', 0)
        
        if psi_score > 0.5 or mean_drift > 0.5:
            return 'HIGH'
        elif psi_score > 0.2 or mean_drift > 0.2:
            return 'MEDIUM'
        elif psi_score > 0.1 or mean_drift > 0.1:
            return 'LOW'
        else:
            return 'NONE'
    
    def log_to_mlflow(self, drift_report: Dict):
        """Log drift monitoring results to MLflow"""
        with mlflow.start_run(run_name=f"drift-monitoring-{datetime.now().strftime('%Y%m%d-%H%M%S')}"):
            # Log summary metrics
            summary = drift_report['drift_summary']
            mlflow.log_metric("features_with_drift", summary['features_with_drift'])
            mlflow.log_metric("drift_percentage", summary['drift_percentage'])
            mlflow.log_metric("sample_size", summary['sample_size'])
            mlflow.log_param("overall_drift_status", summary['overall_drift_status'])
            mlflow.log_param("target_drift_detected", summary['target_drift_detected'])
            
            # Log feature-level metrics
            for feature, metrics in drift_report['detailed_metrics'].items():
                mlflow.log_metric(f"{feature}_ks_p_value", metrics['ks_p_value'])
                mlflow.log_metric(f"{feature}_psi_score", metrics['psi_score'])
                mlflow.log_metric(f"{feature}_mean_drift", metrics['mean_drift'])
                mlflow.log_param(f"{feature}_drift_detected", metrics['drift_detected'])
            
            # Log full report as artifact
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(drift_report, f, indent=2, default=str)
                mlflow.log_artifact(f.name, "drift_reports")
            
            print(f"Drift monitoring results logged to MLflow")
    
    def save_report_to_s3(self, drift_report: Dict, report_key: str):
        """Save drift report to S3"""
        try:
            report_json = json.dumps(drift_report, indent=2, default=str)
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=f"drift-monitoring/{report_key}",
                Body=report_json,
                ContentType='application/json'
            )
            print(f"Drift report saved to S3: s3://{self.bucket_name}/drift-monitoring/{report_key}")
        except Exception as e:
            print(f"Error saving report to S3: {e}")
    
    def monitor_drift(self, reference_data_path: str = None, 
                     current_data: pd.DataFrame = None,
                     drift_type: str = 'none', 
                     drift_magnitude: float = 0.0) -> Dict:
        """Main drift monitoring function"""
        print("Starting data drift monitoring...")
        
        # Load reference data
        if reference_data_path:
            reference_data = self.load_reference_data(reference_data_path)
        else:
            reference_data = self._generate_synthetic_reference_data()
        
        # Generate or use provided current data
        if current_data is None:
            current_data = self.generate_current_data(drift_type, drift_magnitude)
        
        print(f"Reference data shape: {reference_data.shape}")
        print(f"Current data shape: {current_data.shape}")
        
        # Calculate drift metrics
        feature_drift = self.calculate_statistical_drift(reference_data, current_data)
        target_drift = self.calculate_target_drift(reference_data, current_data)
        
        # Generate report
        drift_report = self.generate_drift_report(feature_drift, target_drift, current_data)
        
        # Log results
        self.log_to_mlflow(drift_report)
        
        # Save to S3
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_key = f"drift_report_{timestamp}.json"
        self.save_report_to_s3(drift_report, report_key)
        
        # Store drift metrics in Feature Store
        try:
            from feature_store import MLOpsFeatureStore
            feature_store = MLOpsFeatureStore(self.role_arn, self.bucket_name, self.region_name)
            feature_store.store_drift_metrics(feature_drift)
            print("Drift metrics stored in Feature Store")
        except Exception as e:
            print(f"Could not store drift metrics in Feature Store: {e}")
        
        # Print summary
        self._print_drift_summary(drift_report)
        
        return drift_report
    
    def _print_drift_summary(self, drift_report: Dict):
        """Print human-readable drift summary"""
        summary = drift_report['drift_summary']
        
        print("\n" + "="*50)
        print("📊 DATA DRIFT MONITORING SUMMARY")
        print("="*50)
        print(f"Timestamp: {summary['timestamp']}")
        print(f"Overall Status: {summary['overall_drift_status']}")
        print(f"Features with Drift: {summary['features_with_drift']}/{summary['total_features']} ({summary['drift_percentage']:.1f}%)")
        print(f"Target Drift Detected: {summary['target_drift_detected']}")
        print(f"Sample Size: {summary['sample_size']}")
        
        if summary['features_with_drift'] > 0:
            print("\n🚨 Features with Detected Drift:")
            for feature, metrics in drift_report['feature_drift'].items():
                if metrics['drift_detected']:
                    print(f"  - {feature}: {metrics['severity']} severity (PSI: {metrics['psi_score']:.3f})")
        
        print("\n💡 Recommendations:")
        if summary['overall_drift_status'] == 'DRIFT_DETECTED':
            print("  • Consider retraining the model with recent data")
            print("  • Investigate root causes of drift")
            print("  • Update feature engineering pipeline if needed")
        else:
            print("  • No immediate action required")
            print("  • Continue monitoring data quality")


def main():
    """Main function for CLI usage"""
    # Get configuration from environment variables
    role_arn = os.environ.get('SAGEMAKER_ROLE_ARN')
    bucket_name = os.environ.get('S3_BUCKET_NAME')
    region_name = os.environ.get('AWS_DEFAULT_REGION', 'eu-west-2')
    mlflow_tracking_uri = os.environ.get('MLFLOW_TRACKING_URI')
    
    # Parse command line arguments for drift simulation
    import argparse
    parser = argparse.ArgumentParser(description='Data Drift Monitoring')
    parser.add_argument('--drift-type', choices=['none', 'covariate', 'concept'], 
                       default='none', help='Type of drift to simulate')
    parser.add_argument('--drift-magnitude', type=float, default=0.0,
                       help='Magnitude of drift (0.0 to 1.0)')
    parser.add_argument('--reference-data-path', type=str, default=None,
                       help='Path to reference data (S3 or local)')
    
    args = parser.parse_args()
    
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
    print(f"  Drift Type: {args.drift_type}")
    print(f"  Drift Magnitude: {args.drift_magnitude}")
    print(f"  MLflow tracking URI: {mlflow_tracking_uri or 'Local file store'}")
    print()
    
    # Initialize drift monitor
    drift_monitor = DataDriftMonitor(role_arn, bucket_name, region_name, mlflow_tracking_uri)
    
    # Run drift monitoring
    try:
        drift_report = drift_monitor.monitor_drift(
            reference_data_path=args.reference_data_path,
            drift_type=args.drift_type,
            drift_magnitude=args.drift_magnitude
        )
        
        return drift_report['drift_summary']['overall_drift_status'] == 'NO_DRIFT'
        
    except Exception as e:
        print(f"❌ Drift monitoring failed: {e}")
        return False


if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)