#!/usr/bin/env python3
"""
Generate MLflow experiment summary report
"""
import mlflow
import json
import os

def generate_mlflow_report():
    """Generate a comprehensive MLflow experiment report"""
    
    if not os.path.exists('./mlruns'):
        print("No MLflow experiments found")
        return
    
    try:
        # Get all experiments
        experiments = mlflow.search_experiments()
        report = {
            'experiments': [],
            'total_runs': 0
        }

        for exp in experiments:
            runs = mlflow.search_runs(experiment_ids=[exp.experiment_id])
            exp_data = {
                'name': exp.name,
                'experiment_id': exp.experiment_id,
                'runs_count': len(runs),
                'runs': []
            }
            
            for _, run in runs.iterrows():
                run_data = {
                    'run_id': run['run_id'],
                    'status': run['status'],
                    'start_time': str(run['start_time']),
                    'end_time': str(run['end_time']),
                    'metrics': {k: v for k, v in run.items() if k.startswith('metrics.')},
                    'params': {k: v for k, v in run.items() if k.startswith('params.')}
                }
                exp_data['runs'].append(run_data)
            
            report['experiments'].append(exp_data)
            report['total_runs'] += len(runs)

        # Save report
        with open('mlflow_report.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)

        print(f'MLflow report generated: {report["total_runs"]} total runs across {len(report["experiments"])} experiments')
        
    except Exception as e:
        print(f"Error generating MLflow report: {e}")

if __name__ == '__main__':
    generate_mlflow_report()