import argparse
import random
import threading
import queue
import os
import pandas as pd
import torch
from datetime import datetime
from tqdm import tqdm
from huggingface_hub import login

from evaluation.binder_info import BINDING_INFO
from models.modeling_esm_diff import ESM_Diff
from synthyra_api.affinity_pred import predict_against_target
from .utils import generate_random_aa_sequence


MODEL_PATH = "GleghornLab/ESM_diff_650"
TEMPERATURE = 1.0
REMASKING = "random"
SLOW = False
PREVIEW = False
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
NUM_NEGATIVE_CONTROLS = 20
NUM_WORKER_THREADS = os.cpu_count() // 4 if os.cpu_count() and os.cpu_count() >=4 else 1


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", type=str, default=None, help="HuggingFace token")
    parser.add_argument("--num_samples", type=int, default=100, help="Designs / template")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--api_batch_size", type=int, default=25)
    parser.add_argument("--synthyra_api_key", type=str, default=None)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--output_file", type=str, default='all_designs.csv', help='Output file name')
    parser.add_argument("--summary_file", type=str, default='summary.txt', help='Summary file name')
    parser.add_argument("--step_divisor", type=int, default=100)
    return parser.parse_args()


def prediction_worker(design_queue: queue.Queue, result_queue: queue.Queue, args: argparse.Namespace):
    """Worker function to process prediction batches in separate threads."""
    while True:
        item = design_queue.get()
        if item is None:  # Signal to terminate
            design_queue.task_done()
            break
        
        target_seq, designs_batch, infos_batch, target_name, true_pkd = item
        
        try:
            df = predict_against_target(
                target=target_seq, # API expects 'target' kwarg
                designs=designs_batch,
                test=args.test,
                api_key=args.synthyra_api_key
            )
            
            mapping = {design_seq: info_str for design_seq, info_str in zip(designs_batch, infos_batch)}
            df["design_info"] = df["SeqB"].map(mapping)
            
            df["Target"] = target_name
            df["True_template_pKd"] = true_pkd
            
            result_queue.put(df)
        except Exception as e:
            print(f"Error in prediction worker for target {target_name} with {len(designs_batch)} designs: {e}")
            pass
        finally:
            design_queue.task_done()


def generate_designs_for_template(
    args: argparse.Namespace,
    target_name: str,
    target_seq: str,
    template: str,
    true_pkd: float,
    num_samples: int,
    batch_size: int,
    tokenizer,
    model,
) -> pd.DataFrame:
    """
    Generates designs for the given template, processes them through the API in parallel,
    and returns a complete DataFrame with the results.
    
    Returns: pd.DataFrame with all design results and metadata
    """
    device = next(model.parameters()).device
    designs, infos, seen = [], [], set()

    print(f"\n=== Generating designs for {target_name} ===")
    pbar = tqdm(total=num_samples, desc="Generating designs")
    while len(designs) < num_samples:
        # ----- create one batch by corrupting the template ---- #
        mask_percentage = random.uniform(0.01, 0.9)

        if random.random() < 0.5:
            # Random sub-region
            length = len(template)
            while True:
                start = random.randint(0, length // 2)
                end = random.randint(start + 1, length)
                if end - start >= length // 4:
                    break
            sub_template = template[start:end]
        else:
            sub_template, start, end = template, 0, len(template)

        tokens = tokenizer.encode(sub_template, add_special_tokens=True, return_tensors="pt").to(device)
        if batch_size > 1:
            tokens = tokens.repeat(batch_size, 1)

        mask_idx = torch.rand_like(tokens.float()) < mask_percentage
        mask_idx[:, 0], mask_idx[:, -1] = False, False  # keep special tokens
        tokens[mask_idx] = tokenizer.mask_token_id

        steps = (tokens[0] == tokenizer.mask_token_id).sum().item() // args.step_divisor
        out = model.mask_diffusion_generate(
            template_tokens=tokens,
            block_wise=False,
            steps=steps,
            temperature=TEMPERATURE,
            remasking=REMASKING,
            preview=PREVIEW,
            slow=SLOW,
            start_with_methionine=False,
        )

        decoded = [model._decode_seq(out[i]) for i in range(out.size(0))]
        for d in decoded:
            if d not in seen:
                designs.append(d)
                infos.append(f"mask-rate:{mask_percentage:.2f}; positions:{start}-{end}")
                seen.add(d)
            if len(designs) >= num_samples:
                break
        pbar.update(len(designs) - pbar.n)
    pbar.close()
    
    # Prepare all items for prediction, including template and negative controls
    print(f"Adding template ({len(template)} aa) for prediction")
    items_to_predict = [(template, "TEMPLATE")] + list(zip(designs, infos))
    
    # Add negative controls if template is available
    if template and len(template) > 0:
        print(f"Adding {NUM_NEGATIVE_CONTROLS} negative controls")
        for _ in range(NUM_NEGATIVE_CONTROLS):
            random_seq = generate_random_aa_sequence(len(template), AMINO_ACIDS)
            items_to_predict.append((random_seq, "NEGATIVE_CONTROL"))
    else:
        print(f"Skipped adding negative controls (template is empty or missing)")
    
    # Set up threading for API predictions
    design_queue = queue.Queue()
    result_queue = queue.Queue()
    all_results_dfs = []
    
    # Start worker threads
    threads = []
    num_threads_to_start = NUM_WORKER_THREADS
    print(f"Starting {num_threads_to_start} prediction worker threads")
    for _ in range(num_threads_to_start):
        t = threading.Thread(target=prediction_worker, args=(design_queue, result_queue, args))
        t.daemon = True
        t.start()
        threads.append(t)
    
    # Submit prediction tasks to the queue in batches
    for i in range(0, len(items_to_predict), args.api_batch_size):
        chunk_data = items_to_predict[i:i+args.api_batch_size]
        designs_for_batch = [d for d, _ in chunk_data]
        infos_for_batch = [info for _, info in chunk_data]
        design_queue.put((target_seq, designs_for_batch, infos_for_batch, target_name, true_pkd))
    
    # Wait for all tasks to be processed
    design_queue.join()
    
    # Collect all results from the result_queue
    while not result_queue.empty():
        try:
            df_from_worker = result_queue.get_nowait()
            all_results_dfs.append(df_from_worker)
        except queue.Empty:
            break
    
    # Signal worker threads to terminate
    for _ in range(num_threads_to_start):
        design_queue.put(None)
    
    # Join all threads
    for t in threads:
        t.join(timeout=5)
    
    # Combine all results
    if all_results_dfs:
        return pd.concat(all_results_dfs, ignore_index=True)
    else:
        return pd.DataFrame()  # Return empty dataframe if no results


def main() -> None:
    args = arg_parser()
    if args.token:
        login(args.token)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ESM_Diff.from_pretrained(MODEL_PATH).to(device).eval()
    tokenizer = model.tokenizer

    all_target_results = []

    for target_name, (
        target_seq,
        target_amino_acids,
        target_amino_idx,
        template,
        true_pkd,
        sequence_source,
        binder_source,
    ) in BINDING_INFO.items():

        if not template or not target_seq:
            print(f"[{target_name}] - skipped (missing template or target sequence).")
            continue

        # Generate designs and get back processed DataFrame with results
        target_df = generate_designs_for_template(
            args,
            target_name,
            target_seq,
            template,
            true_pkd,
            args.num_samples,
            args.batch_size,
            tokenizer,
            model,
        )
        
        if not target_df.empty:
            all_target_results.append(target_df)
            print(f"Completed processing for {target_name} with {len(target_df)} results")

    # Combine all results
    if all_target_results:
        big_df = pd.concat(all_target_results, ignore_index=True)
        big_df.to_csv(args.output_file, index=False)
        print(f"\nSaved complete results to {args.output_file} ({len(big_df)} rows)")
    else:
        print("No results generated")
        return

    lines = [
        f"Multi-target design run  -  {datetime.now():%Y-%m-%d %H:%M UTC}",
        f"num_samples per template: {args.num_samples}",
        f"Negative controls per template: {NUM_NEGATIVE_CONTROLS}",
        "",
    ]
    for tgt in big_df["Target"].unique():
        sub = big_df[big_df["Target"] == tgt].copy()

        # Check if template exists in the results
        template_row = sub[sub["design_info"] == "TEMPLATE"]
        template_pred_pkd = template_row["predicted-pKd"].values[0] if not template_row.empty else float('-inf')

        true_template_pkd = sub["True_template_pKd"].iloc[0]
        abs_err = abs(template_pred_pkd - true_template_pkd) if pd.notna(true_template_pkd) and pd.notna(template_pred_pkd) and template_pred_pkd != float('-inf') else float('nan')

        # exclude template itself, then sort
        gen_sub = sub[sub["design_info"] != "TEMPLATE"].copy() # Keep this to exclude template for "better than template"
        designs_only_sub = gen_sub[gen_sub["design_info"] != "NEGATIVE_CONTROL"].sort_values("predicted-pKd", ascending=False)
        top10 = designs_only_sub.head(10)

        better_cnt = (designs_only_sub["predicted-pKd"] > template_pred_pkd).sum() if template_pred_pkd != float('-inf') else len(designs_only_sub)

        lines.append(f"── {tgt} ──")
        lines.append(f"Template predicted pKd: {template_pred_pkd:.4f}" if template_pred_pkd != float('-inf') else "Template predicted pKd: N/A (not predicted)")
        lines.append(f"Template true pKd     : {true_template_pkd:.4f}" if pd.notna(true_template_pkd) else "Template true pKd     : N/A")
        lines.append(f"|error| (pred-vs-true): {abs_err:.4f}" if pd.notna(abs_err) else "|error| (pred-vs-true): N/A")
        lines.append(f"Designs better than template: {better_cnt} / {args.num_samples}" + (" (all designs, template pKd not available)" if template_pred_pkd == float('-inf') else ""))
        lines.append("Top 10 designs (predicted-pKd / design_info / design):")
        for seq, pkd, info in zip(top10["SeqB"], top10["predicted-pKd"], top10["design_info"]):
            lines.append(f'{pkd:.5f} {info}')
            lines.append(seq)
        lines.append("")

    with open(args.summary_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Saved summary to {args.summary_file}")


if __name__ == "__main__":
    main()
