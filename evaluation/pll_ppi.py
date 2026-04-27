import argparse
import json
import os
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from datasets import load_dataset
from matplotlib import pyplot as plt
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)
from tqdm.auto import tqdm
import wandb

from models.modeling_dsm import DSM
from models.FastPLMs.esm2.modeling_fastesm import FastEsmForMaskedLM


def model_name_from_path(model_path: str) -> str:
    return model_path.replace("/", "_").replace("\\", "_")


def get_logits(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    if hasattr(model, "_get_logits"):
        return model._get_logits(input_ids, attention_mask)
    return model(input_ids=input_ids, attention_mask=attention_mask).logits


def load_model(
    model_path: str,
    device: torch.device,
) -> tuple[torch.nn.Module, object]:
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model_type = getattr(config, "model_type", "")
    if model_type == "dsm":
        model = DSM.from_pretrained(model_path).to(device).eval()
    else:
        model = FastEsmForMaskedLM.from_pretrained(model_path).to(device).eval()
    return model, model.tokenizer


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_target_mask(input_ids: torch.Tensor, eos_token_id: int) -> torch.BoolTensor:
    eos_mask = input_ids == eos_token_id
    eos_cumsum = eos_mask.cumsum(dim=1)
    return (eos_cumsum == 1) & ~eos_mask


def compute_ppll(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    target_mask: torch.BoolTensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    logits = get_logits(model, input_ids, attention_mask)
    log_probs = F.log_softmax(logits, dim=-1)
    token_log_probs = log_probs.gather(-1, input_ids.unsqueeze(-1)).squeeze(-1)
    masked_log_probs = token_log_probs * target_mask.float()
    pll_sum = masked_log_probs.sum(dim=1)
    n_target = target_mask.sum(dim=1).clamp(min=1)
    pll_mean = pll_sum / n_target
    return pll_sum, pll_mean


def compute_pll(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    target_mask: torch.BoolTensor,
    mask_token_id: int,
    position_batch_size: int = 32,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert input_ids.shape[0] == 1, "Traditional PLL only supports batch_size=1"
    target_positions = target_mask[0].nonzero(as_tuple=True)[0]
    n_positions = len(target_positions)

    if n_positions == 0:
        return torch.tensor(0.0, device=input_ids.device), torch.tensor(0.0, device=input_ids.device)

    total_ll = torch.tensor(0.0, device=input_ids.device)

    for batch_start in range(0, n_positions, position_batch_size):
        batch_end = min(batch_start + position_batch_size, n_positions)
        batch_positions = target_positions[batch_start:batch_end]
        batch_size_actual = len(batch_positions)

        masked_ids = input_ids.expand(batch_size_actual, -1).clone()
        batch_attn = attention_mask.expand(batch_size_actual, -1)

        for i, pos in enumerate(batch_positions):
            masked_ids[i, pos] = mask_token_id

        logits = get_logits(model, masked_ids, batch_attn)
        log_probs = F.log_softmax(logits, dim=-1)

        for i, pos in enumerate(batch_positions):
            true_token = input_ids[0, pos]
            total_ll += log_probs[i, pos, true_token]

    pll_mean = total_ll / n_positions
    return total_ll, pll_mean


def tokenize_pair(
    tokenizer,
    seq_a: str,
    seq_b: str,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    template = seq_a + tokenizer.eos_token + seq_b
    input_ids = tokenizer.encode(template, add_special_tokens=True, return_tensors="pt").to(device)
    attention_mask = torch.ones_like(input_ids)
    return input_ids, attention_mask


def score_pair(
    model: torch.nn.Module,
    tokenizer,
    seq_a: str,
    seq_b: str,
    mode: str,
    asymmetric: bool,
    device: torch.device,
    position_batch_size: int = 32,
) -> dict:
    result = {
        "len_a": len(seq_a),
        "len_b": len(seq_b),
    }

    # Direction 1: PLL(B | A) -- score SeqB given SeqA as context
    input_ids, attention_mask = tokenize_pair(tokenizer, seq_a, seq_b, device)
    target_mask = make_target_mask(input_ids, tokenizer.eos_token_id)

    if mode == "ppll":
        pll_sum, pll_mean = compute_ppll(model, input_ids, attention_mask, target_mask)
    else:
        pll_sum, pll_mean = compute_pll(
            model, input_ids, attention_mask, target_mask,
            tokenizer.mask_token_id, position_batch_size,
        )

    result["pll_b_given_a_sum"] = pll_sum.item()
    result["pll_b_given_a_mean"] = pll_mean.item()

    if not asymmetric:
        # Direction 2: PLL(A | B) -- score SeqA given SeqB as context
        input_ids, attention_mask = tokenize_pair(tokenizer, seq_b, seq_a, device)
        target_mask = make_target_mask(input_ids, tokenizer.eos_token_id)

        if mode == "ppll":
            pll_sum, pll_mean = compute_ppll(model, input_ids, attention_mask, target_mask)
        else:
            pll_sum, pll_mean = compute_pll(
                model, input_ids, attention_mask, target_mask,
                tokenizer.mask_token_id, position_batch_size,
            )

        result["pll_a_given_b_sum"] = pll_sum.item()
        result["pll_a_given_b_mean"] = pll_mean.item()
        result["pll_symmetric_sum"] = result["pll_b_given_a_sum"] + result["pll_a_given_b_sum"]
        result["pll_symmetric_mean"] = result["pll_b_given_a_mean"] + result["pll_a_given_b_mean"]
    else:
        result["pll_a_given_b_sum"] = float("nan")
        result["pll_a_given_b_mean"] = float("nan")
        result["pll_symmetric_sum"] = float("nan")
        result["pll_symmetric_mean"] = float("nan")

    return result


def load_datasets(
    num_samples: int,
    max_combined_length: int,
) -> tuple:
    positives = load_dataset("Synthyra/BIOGRID-MV", split="train")
    positives = positives.filter(
        lambda x: len(x["SeqA"]) > 20
        and len(x["SeqB"]) > 20
        and len(x["SeqA"]) + len(x["SeqB"]) < max_combined_length
    )
    negatives = load_dataset("Synthyra/NEGATOME", split="combined")
    negatives = negatives.filter(
        lambda x: len(x["SeqA"]) > 20
        and len(x["SeqB"]) > 20
        and len(x["SeqA"]) + len(x["SeqB"]) < max_combined_length
    )

    n_pos_available = len(positives)
    n_neg_available = len(negatives)
    n_pos = min(num_samples, n_pos_available)
    n_neg = min(num_samples, n_neg_available)

    if n_pos < num_samples:
        print(f"WARNING: Only {n_pos_available} positives available after filtering (requested {num_samples})")
    if n_neg < num_samples:
        print(f"WARNING: Only {n_neg_available} negatives available after filtering (requested {num_samples})")

    positives = positives.shuffle(seed=42).select(range(n_pos))
    negatives = negatives.shuffle(seed=42).select(range(n_neg))

    print(f"Loaded {n_pos} positives (BIOGRID-MV), {n_neg} negatives (NEGATOME)")
    return positives, negatives


def evaluate_and_plot(df: pd.DataFrame, output_dir: str, asymmetric: bool) -> dict:
    os.makedirs(output_dir, exist_ok=True)

    labels = df["label"].values
    metrics = {}

    if asymmetric:
        score_cols = ["pll_b_given_a_sum", "pll_b_given_a_mean"]
    else:
        score_cols = [
            "pll_b_given_a_sum", "pll_b_given_a_mean",
            "pll_a_given_b_sum", "pll_a_given_b_mean",
            "pll_symmetric_sum", "pll_symmetric_mean",
        ]

    for col in score_cols:
        scores = df[col].values
        auroc = roc_auc_score(labels, scores)
        auprc = average_precision_score(labels, scores)
        metrics[f"auroc_{col}"] = auroc
        metrics[f"auprc_{col}"] = auprc
        print(f"  {col}: AUROC={auroc:.4f}  AUPRC={auprc:.4f}")

    # ROC curves
    fig, ax = plt.subplots(figsize=(8, 6))
    for col in score_cols:
        fpr, tpr, _ = roc_curve(labels, df[col].values)
        ax.plot(fpr, tpr, label=f"{col} (AUROC={metrics[f'auroc_{col}']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "roc_curves.png"), dpi=300)
    plt.close(fig)

    # PR curves
    fig, ax = plt.subplots(figsize=(8, 6))
    for col in score_cols:
        precision, recall, _ = precision_recall_curve(labels, df[col].values)
        ax.plot(recall, precision, label=f"{col} (AUPRC={metrics[f'auprc_{col}']:.3f})")
    baseline = labels.mean()
    ax.axhline(baseline, color="k", linestyle="--", alpha=0.3, label=f"baseline ({baseline:.2f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "pr_curves.png"), dpi=300)
    plt.close(fig)

    # Violin plots
    n_cols = len(score_cols)
    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 5))
    if n_cols == 1:
        axes = [axes]
    for ax, col in zip(axes, score_cols):
        pos_scores = df.loc[df["label"] == 1, col].values
        neg_scores = df.loc[df["label"] == 0, col].values
        parts = ax.violinplot([neg_scores, pos_scores], positions=[0, 1], showmedians=True)
        for i, pc in enumerate(parts["bodies"]):
            pc.set_facecolor(["#d62728", "#2ca02c"][i])
            pc.set_alpha(0.6)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Negative", "Positive"])
        ax.set_title(col, fontsize=8)
        ax.set_ylabel("PLL score")
    fig.suptitle("Score Distributions by Label")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "violin_plots.png"), dpi=300)
    plt.close(fig)

    if not asymmetric:
        # Scatter: PLL(B|A) vs PLL(A|B)
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for ax, suffix in zip(axes, ["sum", "mean"]):
            col_ba = f"pll_b_given_a_{suffix}"
            col_ab = f"pll_a_given_b_{suffix}"
            pos_mask = df["label"] == 1
            ax.scatter(
                df.loc[~pos_mask, col_ba], df.loc[~pos_mask, col_ab],
                alpha=0.3, s=10, c="#d62728", label="Negative",
            )
            ax.scatter(
                df.loc[pos_mask, col_ba], df.loc[pos_mask, col_ab],
                alpha=0.3, s=10, c="#2ca02c", label="Positive",
            )
            ax.set_xlabel(f"PLL(B|A) {suffix}")
            ax.set_ylabel(f"PLL(A|B) {suffix}")
            ax.legend()
            ax.set_title(f"Directional Comparison ({suffix})")
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "scatter_directions.png"), dpi=300)
        plt.close(fig)

    # Length vs score
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    combined_len = df["len_a"] + df["len_b"]
    pos_mask = df["label"] == 1
    for ax, col in zip(axes, ["pll_b_given_a_sum", "pll_b_given_a_mean"]):
        ax.scatter(
            combined_len[~pos_mask], df.loc[~pos_mask, col],
            alpha=0.3, s=10, c="#d62728", label="Negative",
        )
        ax.scatter(
            combined_len[pos_mask], df.loc[pos_mask, col],
            alpha=0.3, s=10, c="#2ca02c", label="Positive",
        )
        ax.set_xlabel("Combined Sequence Length")
        ax.set_ylabel(col)
        ax.legend()
        ax.set_title(f"Length vs {col}")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "length_vs_score.png"), dpi=300)
    plt.close(fig)

    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PLL correlation experiment for DSM-ppi")
    parser.add_argument(
        "--model_paths",
        type=str,
        nargs="+",
        default=["Synthyra/DSM_ppi_full"],
    )
    parser.add_argument(
        "--mode",
        choices=["ppll", "pll"],
        default="ppll",
    )
    parser.add_argument(
        "--asymmetric",
        action="store_true",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--max_combined_length",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--position_batch_size",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/pll_ppi",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="DSM",
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--no_wandb",
        action="store_true",
    )
    return parser.parse_args()


def run_single_model(
    model_path: str,
    all_pairs: list[tuple],
    args: argparse.Namespace,
    device: torch.device,
    output_dir: str,
) -> pd.DataFrame:
    print(f"\n{'='*60}")
    print(f"Loading model: {model_path}")
    print(f"{'='*60}")
    model, tokenizer = load_model(model_path, device)

    test_a, test_b = "ACDEF", "GHIKL"
    test_ids, _ = tokenize_pair(tokenizer, test_a, test_b, device)
    decoded = tokenizer.decode(test_ids[0].tolist())
    print(f"Tokenization sanity check: '{test_a}' + '{test_b}' -> '{decoded}'")
    test_mask = make_target_mask(test_ids, tokenizer.eos_token_id)
    n_target = test_mask.sum().item()
    assert n_target == len(test_b), f"Target mask selected {n_target} positions, expected {len(test_b)}"
    print(f"Target mask selects {n_target} positions (expected {len(test_b)})")

    directions_str = "B|A only" if args.asymmetric else "both directions"
    print(f"Scoring {len(all_pairs)} pairs in {args.mode} mode ({directions_str})...")

    results = []
    for idx, (seq_a, seq_b, label, source) in enumerate(tqdm(all_pairs, desc=model_path)):
        scores = score_pair(
            model, tokenizer, seq_a, seq_b,
            mode=args.mode,
            asymmetric=args.asymmetric,
            device=device,
            position_batch_size=args.position_batch_size,
        )
        scores["idx"] = idx
        scores["seq_a"] = seq_a
        scores["seq_b"] = seq_b
        scores["label"] = label
        scores["source"] = source
        results.append(scores)

    df = pd.DataFrame(results)
    col_order = [
        "idx", "seq_a", "seq_b", "len_a", "len_b", "label", "source",
        "pll_b_given_a_sum", "pll_b_given_a_mean",
        "pll_a_given_b_sum", "pll_a_given_b_mean",
        "pll_symmetric_sum", "pll_symmetric_mean",
    ]
    df = df[col_order]

    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "pll_scores.csv")
    df.to_csv(csv_path, index=False)
    print(f"Scores saved to {csv_path}")

    print(f"\nMetrics for {model_path}:")
    metrics = evaluate_and_plot(df, output_dir, args.asymmetric)

    metrics["n_positives"] = int((df["label"] == 1).sum())
    metrics["n_negatives"] = int((df["label"] == 0).sum())
    metrics["mode"] = args.mode
    metrics["model"] = model_path
    metrics["max_combined_length"] = args.max_combined_length
    metrics["asymmetric"] = args.asymmetric

    metrics_path = os.path.join(output_dir, "pll_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}")
    print(f"Plots saved to {output_dir}/")

    if not args.no_wandb:
        prefixed_metrics = {f"{model_name_from_path(model_path)}/{k}": v for k, v in metrics.items()}
        wandb.log(prefixed_metrics)
        plot_names = ["roc_curves", "pr_curves", "violin_plots", "length_vs_score"]
        if not args.asymmetric:
            plot_names.append("scatter_directions")
        for name in plot_names:
            path = os.path.join(output_dir, f"{name}.png")
            if os.path.exists(path):
                wandb.log({f"{model_name_from_path(model_path)}/{name}": wandb.Image(path)})
        wandb.log({
            f"{model_name_from_path(model_path)}/scores_table":
            wandb.Table(dataframe=df.drop(columns=["seq_a", "seq_b"]))
        })

    del model
    torch.cuda.empty_cache()

    return df


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if not args.no_wandb:
        run_name = args.wandb_run_name or f"pll_ppi_{args.mode}_{int(time.time())}"
        wandb.init(project=args.wandb_project, name=run_name, config=vars(args))
    else:
        print("WARNING: wandb disabled")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading datasets...")
    positives, negatives = load_datasets(args.num_samples, args.max_combined_length)

    all_pairs = []
    for i in range(len(positives)):
        all_pairs.append((positives[i]["SeqA"], positives[i]["SeqB"], 1, "BIOGRID-MV"))
    for i in range(len(negatives)):
        all_pairs.append((negatives[i]["SeqA"], negatives[i]["SeqB"], 0, "NEGATOME"))

    for model_path in args.model_paths:
        model_tag = model_name_from_path(model_path)
        output_dir = os.path.join(args.output_dir, model_tag)
        run_single_model(model_path, all_pairs, args, device, output_dir)

    if not args.no_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
