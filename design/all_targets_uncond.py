import argparse
import random
from datetime import datetime
from typing import List, Tuple, Iterable
import threading
import queue
import os

import pandas as pd
import torch
from tqdm import tqdm
from huggingface_hub import login

from design.binder_info import BINDING_INFO
from models.modeling_esm_diff import ESM_Diff
from synthyra_api.affinity_pred import predict_against_target


# --------------------------- constants & helpers ----------------------------- #
MODEL_PATH = "GleghornLab/ESM_diff_650"
TEMPERATURE = 1.0
REMASKING = "random"
SLOW = False
PREVIEW = False
STEP_DIVISOR = 100
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
    return parser.parse_args()


def chunked(it: Iterable, n: int):
    """Yield successive n-sized chunks from iterable *it*."""
    buf = []
    for x in it:
        buf.append(x)
        if len(buf) == n:
            yield buf
            buf = []
    if buf:
        yield buf


# Added helper function for random sequences
def generate_random_aa_sequence(length: int, alphabet: str) -> str:
    """Generates a random amino acid sequence of a given length."""
    return "".join(random.choice(alphabet) for _ in range(length))


def prediction_worker(design_queue: queue.Queue, result_queue: queue.Queue, args: argparse.Namespace):
    """Worker function to process prediction batches in separate threads."""
    while True:
        item = design_queue.get()
        if item is None:  # Signal to terminate
            design_queue.task_done()
            break
        
        target_seq, designs_batch, infos_batch, target_name, true_pkd = item
        
        # print(f'Thread {threading.get_ident()}: Processing batch of {len(designs_batch)} designs for {target_name}')
        
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


# ------------------------------ main workflow -------------------------------- #
def generate_designs_for_template(
    template: str,
    num_samples: int,
    batch_size: int,
    tokenizer,
    model,
) -> List[Tuple[str, str]]:
    """
    Returns [(design, design_info), …] - size = num_samples (approx; dedup safe).
    design_info is a short note with masking parameters.
    """
    device = next(model.parameters()).device
    designs, infos, seen = [], [], set()

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

        steps = (tokens[0] == tokenizer.mask_token_id).sum().item() // STEP_DIVISOR
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
    return list(zip(designs[:num_samples], infos[:num_samples]))


def main() -> None:
    args = arg_parser()
    if args.token:
        login(args.token)

    # -------------------- load diffusion model once ------------------------- #
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ESM_Diff.from_pretrained(MODEL_PATH).to(device).eval()
    tokenizer = model.tokenizer

    all_results_dfs = []  # Changed: collect dfs from result_queue

    # Create queues for thread communication
    design_queue = queue.Queue()
    result_queue = queue.Queue()

    # Start worker threads
    threads = []
    num_threads_to_start = NUM_WORKER_THREADS
    print(f"Starting {num_threads_to_start} prediction worker threads.")
    for _ in range(num_threads_to_start):
        t = threading.Thread(target=prediction_worker, args=(design_queue, result_queue, args))
        t.daemon = True
        t.start()
        threads.append(t)

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

        print(f"\n=== {target_name} ===")
        # 1. generate designs
        generated_pairs = generate_designs_for_template(
            template,
            args.num_samples,
            args.batch_size,
            tokenizer,
            model,
        )

        # 2. Prepare all items for prediction for this target
        # Ensure the template is the first item in the list to prioritize its processing
        items_to_predict = [(template, "TEMPLATE")] + generated_pairs
        
        # Print confirmation that template is being added to prediction list
        print(f"Added template ({len(template)} aa) for prediction: {template}")

        if template: 
            template_len = len(template)
            if template_len > 0: 
                negative_controls = []
                for _ in range(NUM_NEGATIVE_CONTROLS):
                    random_seq = generate_random_aa_sequence(template_len, AMINO_ACIDS)
                    negative_controls.append((random_seq, "NEGATIVE_CONTROL"))
                items_to_predict.extend(negative_controls)
            else:
                print(f"[{target_name}] - Skipped adding negative controls (template length is 0).")
        else:
            print(f"[{target_name}] - Skipped adding negative controls (template is empty).")

        # Submit prediction tasks to the queue in chunks, ensuring template is in first chunk
        first_chunk = True
        for chunk_data in chunked(items_to_predict, args.api_batch_size):
            designs_for_batch = [d for d, _ in chunk_data]
            infos_for_batch = [info for _, info in chunk_data]
            
            if not designs_for_batch:
                continue
            
            # Print confirmation when submitting a batch containing the template
            if first_chunk:
                if "TEMPLATE" in infos_for_batch:
                    print(f"Submitting batch containing template for {target_name}")
                first_chunk = False
            
            design_queue.put((target_seq, designs_for_batch, infos_for_batch, target_name, true_pkd))

    # Signal worker threads to terminate
    print("\nAll design generation complete. Signaling prediction workers to terminate...")
    for _ in range(num_threads_to_start):
        design_queue.put(None)

    # Wait for all tasks in the design_queue to be processed
    design_queue.join()
    print("All prediction tasks processed by workers.")

    # Collect all results from the result_queue
    while not result_queue.empty():
        try:
            df_from_worker = result_queue.get_nowait()
            all_results_dfs.append(df_from_worker)
        except queue.Empty:
            break
            
    print(f"Collected {len(all_results_dfs)} result DataFrames from workers.")

    if not all_results_dfs:
        print("No designs were processed by workers or no results returned.")
        for t in threads:
            t.join(timeout=5)
        return

    # -------------------------------- outputs ------------------------------- #
    big_df = pd.concat(all_results_dfs, ignore_index=True)
    big_df.to_csv(args.output_file, index=False)
    print(f"\nSaved complete results to {args.output_file}  ({len(big_df)} rows)")

    # -------- create summary.txt with top 10 and stats per target ----------- #
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
        if template_row.empty:
            print(f"WARNING: Template for {tgt} not found in results!")
            # Add the template manually with -inf pKd to ensure it's tracked
            template_from_binding_info = BINDING_INFO.get(tgt, (None, None, None, None, None, None, None))[3]
            if template_from_binding_info:
                print(f"  - Adding missing template from BINDING_INFO: {template_from_binding_info}")
                # Create a new row for the template
                template_dict = {
                    "Target": tgt,
                    "SeqB": template_from_binding_info,
                    "design_info": "TEMPLATE",
                    "predicted-pKd": float('-inf'),
                    "True_template_pKd": BINDING_INFO.get(tgt, (None, None, None, None, float('nan'), None, None))[4]
                }
                # Add this row to both the sub and big_df
                template_df = pd.DataFrame([template_dict])
                sub = pd.concat([sub, template_df], ignore_index=True)
                big_df = pd.concat([big_df, template_df], ignore_index=True)
            template_pred_pkd = float('-inf')
        else:
            # locate template row & template predicted pKd
            template_pred_pkd = template_row["predicted-pKd"].values[0]
            if template_pred_pkd == float('-inf'):
                print(f"WARNING: Template for {tgt} has predicted pKd of -inf")

        true_template_pkd = sub["True_template_pKd"].iloc[0]
        abs_err = abs(template_pred_pkd - true_template_pkd) if pd.notna(true_template_pkd) and pd.notna(template_pred_pkd) and template_pred_pkd != float('-inf') else float('nan')

        # exclude template itself, then sort
        gen_sub = sub[sub["design_info"] != "TEMPLATE"].copy() # Keep this to exclude template for "better than template"
        # Exclude negative controls from "better than template" and top 10 generated designs
        designs_only_sub = gen_sub[gen_sub["design_info"] != "NEGATIVE_CONTROL"].sort_values("predicted-pKd", ascending=False)
        top10 = designs_only_sub.head(10)

        better_cnt = (designs_only_sub["predicted-pKd"] > template_pred_pkd).sum() if template_pred_pkd != float('-inf') else len(designs_only_sub)
        # If you prefer to compare against true pKd instead, replace previous
        # line with:  better_cnt = (designs_only_sub["predicted-pKd"] > true_template_pkd).sum()

        lines.append(f"── {tgt} ──")
        lines.append(f"Template predicted pKd: {template_pred_pkd:.4f}" if template_pred_pkd != float('-inf') else "Template predicted pKd: N/A (not predicted)")
        lines.append(f"Template true pKd     : {true_template_pkd:.4f}" if pd.notna(true_template_pkd) else "Template true pKd     : N/A")
        lines.append(f"|error| (pred-vs-true): {abs_err:.4f}" if pd.notna(abs_err) else "|error| (pred-vs-true): N/A")
        lines.append(f"Designs better than template: {better_cnt} / {args.num_samples}" + (" (all designs, template pKd not available)" if template_pred_pkd == float('-inf') else "")) # num_samples is generated, not total w/ controls
        lines.append("Top 10 designs (predicted-pKd / seq):")
        for seq, pkd in zip(top10["SeqB"], top10["predicted-pKd"]):
            lines.append(f'{pkd:.5f}')
            lines.append(seq)
        lines.append("")

    with open(args.summary_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Saved summary to {args.summary_file}")

    # Ensure all threads are joined before exiting
    for t in threads:
        t.join(timeout=5)
    print("All worker threads have been joined.")


if __name__ == "__main__":
    main()
