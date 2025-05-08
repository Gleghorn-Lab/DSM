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
from huggingface_hub import login, hf_hub_download
from safetensors.torch import load_file

from design.binder_info import BINDING_INFO
from models.modeling_esm_diff import ESM_Diff_Binders, ESMDiffConfig
from models.utils import wrap_lora
from synthyra_api.affinity_pred import predict_against_target


# --------------------------- constants & helpers ----------------------------- #
MODEL_PATH = "lhallee/ESM_diff_bind_650"
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
    parser.add_argument("--output_file", type=str, default='designs.csv', help='Output file name')
    parser.add_argument("--summary_file", type=str, default='summary_conditional.txt', help='Summary file name')
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
        
        # Predict against target
        try:
            df = predict_against_target(
                target=target_seq, # API expects 'target' kwarg for target sequence
                designs=designs_batch,
                test=args.test,
                api_key=args.synthyra_api_key
            )
            
            # Map design_info back.
            # designs_batch contains the sequences, infos_batch contains the corresponding info strings.
            mapping = {design_seq: info_str for design_seq, info_str in zip(designs_batch, infos_batch)}
            df["design_info"] = df["SeqB"].map(mapping) # SeqB contains the generated binder
            
            # Add other annotations
            df["Target"] = target_name
            df["True_template_pKd"] = true_pkd
            
            result_queue.put(df)
        except Exception as e:
            print(f"Error in prediction worker for target {target_name} with {len(designs_batch)} designs: {e}")
            # Optionally, put an empty DataFrame or error marker to avoid blocking
            # For now, just printing error and moving on.
            pass # Or handle more gracefully
        finally:
            design_queue.task_done()


def load_binder_model(model_path):
    local_weight_file = hf_hub_download(
        repo_id=model_path,
        filename='model.safetensors',
        repo_type='model',
    )

    config = ESMDiffConfig.from_pretrained(model_path)
    model = ESM_Diff_Binders(config=config)
    model = wrap_lora(model, r=config.lora_r, lora_alpha=config.lora_alpha, lora_dropout=config.lora_dropout)
    state_dict = load_file(local_weight_file)

    loaded_params = set()
    missing_params = set()

    for name, param in model.named_parameters():
        found = False
        for key in state_dict.keys():
            if key in name: # Simplified matching, original was more complex but this should work for LoRA
                param.data = state_dict[key]
                loaded_params.add(name)
                found = True
                break
        if not found:
            missing_params.add(name)

    print(f"Loaded {len(loaded_params)} parameters")
    print(f"Missing {len(missing_params)} parameters")
    if missing_params:
        print("Missing parameters:")
        for param in sorted(missing_params):
            print(f"  - {param}")
    return model


# ------------------------------ main workflow -------------------------------- #
def generate_designs_for_template(
    target_seq: str, # Added
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

    cls_token = tokenizer.cls_token_id
    eos_token = tokenizer.eos_token_id

    pbar = tqdm(total=num_samples, desc="Generating designs")
    while len(designs) < num_samples:
        mask_percentage = random.uniform(0.01, 0.9)

        if random.random() < 0.5:
            length = len(template)
            sub_template_len = 0
            # Ensure sub_template is at least 1/4 of the original template length
            while sub_template_len < length // 4 or sub_template_len == 0 :
                start = random.randint(0, length // 2)
                end = random.randint(start + 1, length) # Ensure end is after start
                if end - start >= length // 4 : # check if region is large enough
                     sub_template = template[start:end]
                     sub_template_len = len(sub_template)
                elif length // 4 == 0 and length > 0: # handle short templates
                     sub_template = template[start:end]
                     sub_template_len = len(sub_template)
                     if sub_template_len > 0: break # break if any valid sub_template
                if length == 0: # handle empty template string if it occurs
                    start, end = 0,0
                    sub_template = ""
                    break

        else:
            sub_template, start, end = template, 0, len(template)

        # Conditional model tokenization
        target_tokens_encoded = tokenizer.encode(target_seq, add_special_tokens=True, return_tensors="pt").to(device)
        template_tokens_encoded = tokenizer.encode(sub_template, add_special_tokens=False, return_tensors="pt").to(device) # No special tokens for template part
        end_eos_tensor = torch.tensor([eos_token], device=device).unsqueeze(0)

        current_batch_size = batch_size
        if len(designs) + batch_size > num_samples: # Adjust batch size for the last batch
            current_batch_size = num_samples - len(designs)
            if current_batch_size <=0: break


        if current_batch_size > 1:
            target_tokens_batch = target_tokens_encoded.repeat(current_batch_size, 1)
            template_tokens_batch = template_tokens_encoded.repeat(current_batch_size, 1)
            end_eos_batch = end_eos_tensor.repeat(current_batch_size, 1)
        else: # batch_size is 1
            target_tokens_batch = target_tokens_encoded
            template_tokens_batch = template_tokens_encoded
            end_eos_batch = end_eos_tensor
        
        # Mask template tokens only
        mask_idx = torch.rand_like(template_tokens_batch.float()) < mask_percentage
        template_tokens_batch[mask_idx] = tokenizer.mask_token_id

        # Concatenate: [CLS] target_seq [EOS] template_seq [EOS]
        input_tokens = torch.cat([target_tokens_batch, template_tokens_batch, end_eos_batch], dim=1)
        
        # Number of masked tokens in the template part for steps calculation
        # Need to consider the template part of the *first item in the batch* for steps,
        # as masking is random per item.
        # The original conditional_binder.py uses: (template_tokens[0] == tokenizer.mask_token_id).sum().item()
        # Here, template_tokens_batch is already masked.
        # We calculate steps based on the first element of the batch's template part.
        num_masked_in_template_first_item = (template_tokens_batch[0][mask_idx[0]]).sum().item() if current_batch_size > 0 else 0

        steps = num_masked_in_template_first_item // STEP_DIVISOR
        if steps == 0 and num_masked_in_template_first_item > 0 : steps = 1 # Ensure at least 1 step if there are masks

        out = model.mask_diffusion_generate(
            template_tokens=input_tokens, # This is the combined input
            block_wise=False,
            steps=steps,
            temperature=TEMPERATURE,
            remasking=REMASKING,
            preview=PREVIEW,
            slow=SLOW,
            start_with_methionine=False,
        )
        
        # Decode and remove target prefix
        decoded_sequences = [model._decode_seq(out[i])[len(target_seq):] for i in range(out.size(0))]

        for d_seq in decoded_sequences:
            if d_seq not in seen:
                designs.append(d_seq)
                infos.append(f"mask-rate:{mask_percentage:.2f}; positions:{start}-{end}")
                seen.add(d_seq)
            if len(designs) >= num_samples:
                break
        pbar.update(min(current_batch_size, len(designs) - pbar.n))
        if len(designs) >= num_samples:
                break
    pbar.close()
    return list(zip(designs[:num_samples], infos[:num_samples]))


def main() -> None:
    args = arg_parser()
    if args.token:
        login(args.token)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_binder_model(MODEL_PATH)
    model = model.to(device).eval()
    tokenizer = model.tokenizer

    all_results_dfs = [] # Changed: will collect DataFrames from result_queue

    # Create queues for thread communication
    design_queue = queue.Queue()
    result_queue = queue.Queue()

    # Start worker threads
    threads = []
    num_threads_to_start = NUM_WORKER_THREADS
    print(f"Starting {num_threads_to_start} prediction worker threads.")
    for _ in range(num_threads_to_start):
        t = threading.Thread(target=prediction_worker, args=(design_queue, result_queue, args))
        t.daemon = True  # Allow main program to exit even if threads are blocked
        t.start()
        threads.append(t)
    
    # Keep track of how many tasks are submitted for each target to manage result collection if needed,
    # or rely on design_queue.join() and then draining result_queue.
    # For simplicity, we'll add all tasks and then collect all results.

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
        generated_pairs = generate_designs_for_template( # Renamed from 'pairs' to avoid confusion
            target_seq, 
            template,
            args.num_samples,
            args.batch_size,
            tokenizer,
            model,
        )
        
        # Prepare all items for prediction for this target
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
            # chunk_data is a list of (sequence, info_string)
            designs_for_batch = [d for d, _ in chunk_data]
            infos_for_batch = [info for _, info in chunk_data]
            
            if not designs_for_batch: # Should not happen if items_to_predict is not empty
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
            break # Should not happen if join() was effective and queue was drained

    print(f"Collected {len(all_results_dfs)} result DataFrames from workers.")

    if not all_results_dfs: # Changed from all_results
        print("No designs were processed by workers or no results returned.")
        # Ensure threads are joined even if no results
        for t in threads:
            t.join(timeout=5) # Add a timeout to prevent indefinite blocking
        return

    big_df = pd.concat(all_results_dfs, ignore_index=True) # Changed from all_results
    # Rename SeqB to Design as it's the binder sequence
    if "SeqB" in big_df.columns:
        big_df = big_df.rename(columns={"SeqB": "Design"})
    if "SeqA" in big_df.columns: # SeqA is the target, can be dropped if Target column is present
        big_df = big_df.drop(columns=["SeqA"])


    output_csv_name = args.output_file
    summary_txt_name = args.summary_file

    big_df.to_csv(output_csv_name, index=False)
    print(f"\nSaved complete results to {output_csv_name}  ({len(big_df)} rows)")

    lines = [
        f"Multi-target CONDITIONAL design run - {datetime.now():%Y-%m-%d %H:%M UTC}",
        f"num_samples per template: {args.num_samples}",
        f"Negative controls per template: {NUM_NEGATIVE_CONTROLS}", # Added
        f"Model: {MODEL_PATH}",
        "",
    ]
    for tgt_name_unique in big_df["Target"].unique():
        sub_df = big_df[big_df["Target"] == tgt_name_unique].copy()

        # Check if template exists in the results
        template_row = sub_df[sub_df["design_info"] == "TEMPLATE"]
        template_pred_pkd = float('-inf') # Default if template not found or no pKd
        
        if template_row.empty:
            print(f"WARNING: Template for {tgt_name_unique} not found in results!")
            # Add the template manually with -inf pKd to ensure it's tracked
            template_from_binding_info = BINDING_INFO.get(tgt_name_unique, (None, None, None, None, None, None, None))[3]
            if template_from_binding_info:
                print(f"  - Adding missing template from BINDING_INFO: {template_from_binding_info}")
                # Create a new row for the template
                template_dict = {
                    "Target": tgt_name_unique,
                    "Design": template_from_binding_info,
                    "design_info": "TEMPLATE",
                    "predicted-pKd": float('-inf'),
                    "True_template_pKd": BINDING_INFO.get(tgt_name_unique, (None, None, None, None, float('nan'), None, None))[4]
                }
                # Add this row to both the sub_df and big_df
                template_df = pd.DataFrame([template_dict])
                sub_df = pd.concat([sub_df, template_df], ignore_index=True)
                big_df = pd.concat([big_df, template_df], ignore_index=True)
        elif "predicted-pKd" in template_row.columns:
            template_pred_pkd = template_row["predicted-pKd"].values[0]
            if template_pred_pkd == float('-inf'):
                print(f"WARNING: Template for {tgt_name_unique} has predicted pKd of -inf")

        true_template_pkd_val = sub_df["True_template_pKd"].iloc[0] if "True_template_pKd" in sub_df.columns and not sub_df.empty else float('nan')
        
        abs_err = abs(template_pred_pkd - true_template_pkd_val) if pd.notna(true_template_pkd_val) and pd.notna(template_pred_pkd) and template_pred_pkd != float('-inf') else float('nan')

        # Exclude template itself for ranking generated designs, then sort
        gen_sub_df = sub_df[sub_df["design_info"] != "TEMPLATE"].copy()
        # Exclude negative controls from "better than template" and top 10 generated designs
        designs_only_sub_df = gen_sub_df[gen_sub_df["design_info"] != "NEGATIVE_CONTROL"].sort_values("predicted-pKd", ascending=False)
        top10_designs = designs_only_sub_df.head(10)

        # Compare generated designs' pKd to the *predicted* pKd of the template for this run
        better_than_template_count = (designs_only_sub_df["predicted-pKd"] > template_pred_pkd).sum() if template_pred_pkd != float('-inf') else len(designs_only_sub_df)

        lines.append(f"── {tgt_name_unique} ──")
        lines.append(f"Template predicted pKd: {template_pred_pkd:.4f}" if template_pred_pkd != float('-inf') else "Template predicted pKd: N/A (not predicted)")
        lines.append(f"Template true pKd     : {true_template_pkd_val:.4f}" if pd.notna(true_template_pkd_val) else "Template true pKd     : N/A")
        lines.append(f"|error| (pred-vs-true): {abs_err:.4f}" if pd.notna(abs_err) else "|error| (pred-vs-true): N/A")
        lines.append(f"Designs better than template (vs predicted): {better_than_template_count} / {len(designs_only_sub_df)}" + (" (all designs, template pKd not available)" if template_pred_pkd == float('-inf') else ""))
        lines.append("Top 10 designs (predicted-pKd / Design):")
        for _, row in top10_designs.iterrows():
            design_seq = row["Design"]
            pkd_val = row["predicted-pKd"]
            lines.append(f'{pkd_val:.5f}')
            lines.append(design_seq)
        lines.append("")

    with open(summary_txt_name, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Saved {summary_txt_name}")

    # Ensure all threads are joined before exiting
    for t in threads:
        t.join(timeout=5) # Add a timeout to prevent indefinite blocking
    print("All worker threads have been joined.")


if __name__ == "__main__":
    main() 