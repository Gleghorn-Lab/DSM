import time
import requests
import io
import csv
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Optional, List, Dict

from .utils import aspect_dict


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


def send_request(data: List[Dict[str, str]], task_type: str, api_key: str = None) -> Optional[float]:
    """
    Run a single test against the API with generated data.
    
    Args:
        num_seqs: The number of sequences to generate
        task_type: The API task type (ppi or annotation)
        api_key: API key for Synthyra API
        
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
            'options': '["ppi", "affinity"]'
        }
    elif task_type == 'annotation':
        data = {
            'name': name,
        }

    params = {
        'api_key': api_key
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
        params = {'job_id': job_id, 'api_key': api_key}
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


def predict_annotations(seqs: List[str], api_key: str = None) -> pd.DataFrame:
    data = [{'seq': seq} for seq in seqs]
    df = send_request(data, 'annotation', api_key)
    return df


def parse_annotations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a dictionary of sequence: annotations
    annotations is a list of strings separated by ;
    we remove the confidence score from the annotations
    """
    annotation_dict = {}
    for i, row in df.iterrows():
        seq = row['seqs']
        total_annotations = []
        for aspect, name in aspect_dict.items():
            preds_str = row[name]
            if str(preds_str) == 'nan':
                continue
            preds_str = preds_str.split(';')
            preds_str = [pred.split('(')[0].strip() for pred in preds_str]
            total_annotations.extend(preds_str)
        annotation_dict[seq] = ';'.join(total_annotations)
    return annotation_dict


if __name__ == "__main__":
    # py -m synthyra_api.annotation
    import pandas as pd
    import ast
    import argparse
    from collections import defaultdict
    from datasets import Dataset, load_dataset

    from .utils import describe_prompt, aspect_dict, id2label, annotation_vocab_dict

    parser = argparse.ArgumentParser()
    parser.add_argument("--synthyra_api_key", type=str, default=None, help="Synthyra API key")
    parser.add_argument("--result_path", type=str, default=None, help="Path to save results")
    parser.add_argument("--data_path", type=str, default=None, help="Path to data")
    args = parser.parse_args()


    if args.data_path == 'test':
        train_dataset = load_dataset('lhallee/AV', split='train').shuffle(seed=1234)
        train_dataset = train_dataset.train_test_split(test_size=2000, seed=42)
        valid_dataset = train_dataset['test']
        train_dataset = train_dataset['train']
        valid_dataset = valid_dataset.train_test_split(test_size=1000, seed=24)
        data = valid_dataset['test']
        valid_dataset = valid_dataset['train']
        del train_dataset, valid_dataset
    else:
        df = pd.read_csv(args.data_path)
        data = Dataset.from_pandas(df)

    data = data.map(lambda x: {'annotations': ast.literal_eval(x['annotations'])})
    seqs = data['sequence']
    seqs = [seq[:2000] for seq in seqs]
    annotations = data['annotations']

    if args.result_path is None:
        data = [{'seq': seqs[i]} for i in range(len(seqs))]
        result_df = send_request(data, 'annotation', args.synthyra_api_key)
    else:
        result_df = pd.read_csv(args.result_path)

    metrics = defaultdict(dict)
    # Initialize counters for all aspects
    total_true_positives = defaultdict(int)
    total_predictions = defaultdict(int)
    total_labels = defaultdict(int)
    
    for i, row in result_df.iterrows():
        descriptions, _ = describe_prompt(annotations[i], id2label, annotation_vocab_dict)
        seq = row['seqs']
        assert seq == seqs[i], f'Sequence mismatch: {seq} != {seqs[i]}'
        for aspect, name in aspect_dict.items():
            preds_str = row[name]

            if str(preds_str) == 'nan':
                preds_str = None
            
            labels = descriptions[aspect]
            total_labels[aspect] += len(labels)
            
            # Count true positives
            true_positives = 0
            if preds_str:
                pred_len = len(preds_str.split(';'))
                total_predictions[aspect] += pred_len
                
                for entry in labels:
                    if entry in preds_str:
                        true_positives += 1
                        total_true_positives[aspect] += 1        
    
    # Calculate metrics across all examples
    for aspect in aspect_dict.keys():
        precision = total_true_positives[aspect] / total_predictions[aspect] if total_predictions[aspect] > 0 else 0
        recall = total_true_positives[aspect] / total_labels[aspect] if total_labels[aspect] > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # Store metrics
        metrics[aspect]['precision'] = precision
        metrics[aspect]['recall'] = recall
        metrics[aspect]['f1'] = f1
        metrics[aspect]['accuracy'] = total_true_positives[aspect] / total_labels[aspect] if total_labels[aspect] > 0 else 0

    print("Metrics by aspect:")
    for aspect, scores in metrics.items():
        print(f"{aspect}: Precision={scores['precision']:.2f}, Recall={scores['recall']:.2f}, F1={scores['f1']:.2f}, Accuracy={scores['accuracy']:.2f}")
    