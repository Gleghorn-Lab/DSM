import pandas as pd
import threading
import queue
import os
from synthyra_api.annotation import predict_annotations, parse_annotations


def annotation_worker(sequences, result_queue, worker_id):
    """Worker function to process annotation batches in separate threads."""
    print(f"Worker {worker_id} processing {len(sequences)} sequences")
    annotations = predict_annotations(sequences)
    parsed_annotations = parse_annotations(annotations)
    result_queue.put((worker_id, parsed_annotations))
    print(f"Worker {worker_id} completed")


if __name__ == "__main__":
    # py -m evaluation.annotate_comparisons
    import argparse

    def arg_parser():
        parser = argparse.ArgumentParser()
        parser.add_argument('--token', type=str, default=None)
        parser.add_argument('--input_path', type=str, default='test_compare.csv')
        parser.add_argument('--output_path', type=str, default='test_compare_annotated.csv')
        return parser.parse_args()

    args = arg_parser()
    args.input_path = os.path.join('evaluation', 'comparisons', args.input_path)
    args.output_path = os.path.join('evaluation', 'comparisons', args.output_path)
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    df = pd.read_csv(args.input_path)
    natural_seqs = df['natural'].tolist()
    generated_seqs = df['generated'].tolist()

    # Create queue for thread communication
    result_queue = queue.Queue()
    
    # Start worker threads
    threads = []
    
    # Thread for natural sequences
    natural_thread = threading.Thread(
        target=annotation_worker, 
        args=(natural_seqs, result_queue, "natural")
    )
    natural_thread.start()
    threads.append(natural_thread)
    
    # Thread for generated sequences
    generated_thread = threading.Thread(
        target=annotation_worker, 
        args=(generated_seqs, result_queue, "generated")
    )
    generated_thread.start()
    threads.append(generated_thread)
    
    # Wait for all threads to complete
    for thread in threads:
        thread.join()
    
    # Collect results
    results = {}
    while not result_queue.empty():
        worker_id, annotations = result_queue.get()
        results[worker_id] = annotations
    
    # Access the results
    natural_annotations = results["natural"]
    generated_annotations = results["generated"]

    df['natural_annotations'] = df['natural'].map(natural_annotations)
    df['generated_annotations'] = df['generated'].map(generated_annotations)

    df.to_csv(args.output_path, index=False)
