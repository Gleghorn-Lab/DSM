import time
import requests
import io
import csv
import pandas as pd
from datetime import datetime
from typing import Optional, List, Dict


API_KEY = '7147b8da62cc094c11d688dbac739e4689cdc7952d5196a488e5d95a6c2f2da1'


def convert_data_to_csv_bytes(data: List[Dict[str, str]], task_type: str) -> bytes:
    """
    Convert data to CSV format in memory as bytes.
    
    Args:
        data: List of dictionaries containing the sequences
        task_type: The type of task ('ppi' or 'annotation')
        
    Returns:
        CSV data as bytes
    """
    output = io.StringIO()
    
    if task_type == 'ppi':
        fieldnames = ['SeqA', 'SeqB']
    else:  # annotation
        fieldnames = ['seq']
        
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    
    for item in data:
        writer.writerow(item)
    
    return output.getvalue().encode('utf-8')


def send_request(data: List[Dict[str, str]], task_type: str) -> Optional[float]:
    """
    Run a single test against the API with generated data.
    
    Args:
        num_seqs: The number of sequences to generate
        task_type: The API task type (ppi or annotation)
        
    Returns:
        Elapsed time in seconds, or None if the test failed
    """
    
    # Convert data to CSV bytes
    csv_bytes = convert_data_to_csv_bytes(data, task_type)
    
    files = {
        'file': ('input.csv', csv_bytes),
    }
    name = f'api_test_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    if task_type == 'ppi':
        data = {
            'name': name,
            'options': '["ppi", "affinity", "binding"]'
        }
    elif task_type == 'annotation':
        data = {
            'name': name,
        }

    params = {
        'api_key': API_KEY
    }

    response = requests.post(
        f'https://api.synthyra.com/v1/generate/{task_type}',
        params=params,
        files=files,
        data=data
    )
    job_id = response.json().get('job_id', None)
    print(f"Job ID: {job_id}")
    start_time = time.time()

    if not job_id:
        print(f"Error in submission: {response}")
        return None


    while True:
        params = {'job_id': job_id, 'api_key': API_KEY}
        response = requests.get('https://api.synthyra.com/v1/job', params=params)
        
        try:
            status = response.json()
            print(f'\nRunning... {status}')
        except:
            output = io.StringIO(response.text)
            df = pd.read_csv(output)
            print(df.head())
            print(f"Job completed in {time.time() - start_time} seconds")
            return df
        time.sleep(10)


def predict_against_target(target: str, designs: List[str]) -> pd.DataFrame:
    """
    Predict the affinity of a list of designs against a target sequence.
    Sort by predicted affinity.
    """
    data = [{'SeqA': target, 'SeqB': design} for design in designs]
    df = send_request(data, 'ppi')
    return df


if __name__ == "__main__":
    # py -m design.affinity_pred
    import argparse
    import numpy as np
    import os
    from datasets import load_dataset
    from transformers import EvalPrediction

    from metrics.regression import compute_metrics_regression
    from metrics.classification import compute_metrics_single_label_classification
    from metrics.regression_plot import plot_predictions

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default=None, help="Path to the dataset")
    args = parser.parse_args()

    if args.data_path is None:
        data_paths = [
            'Synthyra/haddock_benchmark',
            'Synthyra/AffinityBenchmarkv5.5',
            'Synthyra/AdaptyvBioRound2EGFR',
            'Synthyra/ProteinProteinAffinity'
        ]
    else:
        data_paths = [args.data_path]

    for data_path in data_paths:
        data = load_dataset(data_path, split='train').shuffle(seed=42)
        if len(data) > 1000:
            data = data.select(range(1000))
        seqs_a, seqs_b = data['SeqA'], data['SeqB']
        try:
            labels = data['labels']
        except:
            labels = data['pkd']
        data = [{'SeqA': seqs_a[i], 'SeqB': seqs_b[i]} for i in range(len(seqs_a))]

        df = send_request(data, 'ppi')

        ppi_preds = np.array(df['ppi-pred']).flatten()
        affinity_preds = np.array(df['predicted-pKd']).flatten()
        affinity_labels = np.array(labels).flatten()
        ppi_labels = (affinity_labels > 0.0).astype(int)

        affinity_metrics = compute_metrics_regression(EvalPrediction(predictions=affinity_preds, label_ids=affinity_labels))
        ppi_metrics = compute_metrics_single_label_classification(EvalPrediction(predictions=ppi_preds, label_ids=ppi_labels))

        # Print metrics
        print("\nEvaluation Metrics:")
        for metric, value in affinity_metrics.items():
            print(f"{metric}: {value}")
        for metric, value in ppi_metrics.items():
            print(f"{metric}: {value}")
        # Create and save visualization
        data_name = data_path.split('/')[-1]
        os.makedirs('plots', exist_ok=True)
        plot_predictions(affinity_preds, affinity_labels, affinity_metrics, f'plots/{data_name}_affinity_pred.png')
