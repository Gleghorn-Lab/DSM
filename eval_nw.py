import torch
import argparse
import numpy as np
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import EsmTokenizer, EvalPrediction
from tqdm import tqdm

from metrics.regression import compute_metrics_regression
from models.modeling_nw_transformer import NWTransformerCross
from data.dataset_classes import NWDatasetEval
from data.data_collators import NWCollatorCross
from metrics.regression_plot import plot_predictions


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="GleghornLab/AlignmentTransformer", help="Path to the model to evaluate")
    parser.add_argument("--dataset_name", type=str, default="Synthyra/bernett_gold_ppi", help="Name of the dataset to use for evaluation")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for evaluation")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for evaluation")
    parser.add_argument("--max_length", type=int, default=4096, help="Maximum length of sequences fed to the model")
    parser.add_argument("--output_plot", type=str, default="prediction_plot.png", help="Path to save the visualization")
    return parser.parse_args()


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = load_dataset(args.dataset_name, split='test').filter(
        lambda x: len(x['SeqA']) + len(x['SeqB']) <= args.max_length
    )
    data = data.shuffle(seed=42)
    print(data)

    tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")   
    model = NWTransformerCross.from_pretrained(args.model_path)
    model.eval()
    model.to(device)

    eval_dataset = NWDatasetEval(data)
    eval_collator = NWCollatorCross(tokenizer, args.max_length)
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=eval_collator
    )

    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in tqdm(eval_dataloader, total=len(eval_dataloader)):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            preds = outputs.logits.cpu().numpy()
            labels = batch['labels'].cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels)

    # Concatenate all predictions and labels
    all_preds = np.stack(all_preds).flatten()
    all_labels = np.stack(all_labels).flatten()
    
    # Compute evaluation metrics
    metrics = compute_metrics_regression(EvalPrediction(predictions=all_preds, label_ids=all_labels))
    
    # Print metrics
    print("\nEvaluation Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value}")
    
    # Create and save visualization
    plot_predictions(all_preds, all_labels, metrics, args.output_plot)


if __name__ == "__main__":
    args = parse_args()
    main(args)