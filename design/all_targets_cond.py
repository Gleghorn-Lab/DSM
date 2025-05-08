import argparse
import random
from datetime import datetime
from typing import List, Tuple, Iterable

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
MODEL_PATH = "lhallee/ESM_diff_bind_650"  # Changed
TEMPERATURE = 1.0
REMASKING = "random"
SLOW = False
PREVIEW = False
STEP_DIVISOR = 10  # Changed
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY" # Added
NUM_NEGATIVE_CONTROLS = 20 # Added


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", type=str, default=None, help="HuggingFace token")
    parser.add_argument("--num_samples", type=int, default=100, help="Designs / template")
    parser.add_argument("--batch_size", type=int, default=1) # Conditional model might need smaller batch due to longer sequences
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


def load_binder_model(model_path): # Added from conditional_binder.py
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


def predict_in_batches(
    target_seq: str,
    designs_info: List[Tuple[str, str]],
    api_batch_size: int,
    api_key: str,
    test: bool,
) -> pd.DataFrame:
    """
    Calls Synthyra in api_batch_size chunks, returns dataframe with
    columns: SeqA, SeqB, predicted-pKd, … plus 'design_info'.
    """
    dfs = []
    for chunk_data in chunked(designs_info, api_batch_size):
        current_designs = [d for d, _ in chunk_data]
        # infos_for_chunk = [info for _, info in chunk_data] # Not directly used for mapping here
        df = predict_against_target(target_seq, current_designs, test=test, api_key=api_key)
        # Map design_info back. Ensure design sequences in chunk_data are unique for correct mapping.
        # Since 'seen' set is used in generation, designs going into a batch for prediction should be unique.
        mapping = {d: info for d, info in chunk_data}
        df["design_info"] = df["SeqB"].map(mapping) # SeqB contains the generated binder
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def main() -> None:
    args = arg_parser()
    if args.token:
        login(args.token)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load conditional binder model
    model = load_binder_model(MODEL_PATH) # Changed
    model = model.to(device).eval()
    tokenizer = model.tokenizer

    all_results = []

    for target_name, (
        target_seq,
        target_amino_acids, # Not used in this script directly for generation/prediction
        target_amino_idx,   # Not used
        template,
        true_pkd,
        sequence_source,    # Not used
        binder_source,      # Not used
    ) in BINDING_INFO.items():

        if not template or not target_seq:
            print(f"[{target_name}] - skipped (missing template or target sequence).")
            continue

        print(f"\n=== {target_name} ===")
        # 1. generate designs
        pairs = generate_designs_for_template(
            target_seq, # Added
            template,
            args.num_samples,
            args.batch_size,
            tokenizer,
            model,
        )

        # 2. predict affinities (template + designs) - ensure template is included
        # The template itself is also a "design" in this context, ensure it's handled appropriately for conditional model
        # For conditional, the "template" design means generating from the original template sequence given the target
        # The predict_against_target function expects SeqA (target) and SeqB (binder)
        
        # Add original template to the list for affinity prediction
        # The "design_info" for the original template will be "TEMPLATE"
        pairs_with_template = [(template, "TEMPLATE")] + pairs

        # Add negative controls
        if template: # Ensure template is not empty to get a length
            template_len = len(template)
            if template_len > 0: # Ensure template length is positive
                negative_controls = []
                for _ in range(NUM_NEGATIVE_CONTROLS):
                    random_seq = generate_random_aa_sequence(template_len, AMINO_ACIDS)
                    negative_controls.append((random_seq, "NEGATIVE_CONTROL"))
                pairs_with_template.extend(negative_controls)
            else:
                print(f"[{target_name}] - Skipped adding negative controls (template length is 0).")
        else:
            print(f"[{target_name}] - Skipped adding negative controls (template is empty).")
        
        df_target = predict_in_batches(
            target_seq, # This is SeqA for the API
            pairs_with_template, # List of (binder_sequence, design_info)
            args.api_batch_size,
            args.synthyra_api_key,
            args.test,
        )

        df_target["Target"] = target_name
        df_target["True_template_pKd"] = true_pkd # True pKd of the original template for this target

        all_results.append(df_target)

    if not all_results:
        print("No designs generated for any target.")
        return

    big_df = pd.concat(all_results, ignore_index=True)
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
        f"Multi-target CONDITIONAL design run  –  {datetime.now(datetime.timezone.utc):%Y-%m-%d %H:%M UTC}",
        f"num_samples per template: {args.num_samples}",
        f"Negative controls per template: {NUM_NEGATIVE_CONTROLS}", # Added
        f"Model: {MODEL_PATH}",
        "",
    ]
    for tgt_name_unique in big_df["Target"].unique():
        sub_df = big_df[big_df["Target"] == tgt_name_unique].copy()

        template_row = sub_df[sub_df["design_info"] == "TEMPLATE"]
        template_pred_pkd = float('-inf') # Default if template not found or no pKd
        if not template_row.empty and "predicted-pKd" in template_row.columns:
             template_pred_pkd = template_row["predicted-pKd"].values[0]

        true_template_pkd_val = sub_df["True_template_pKd"].iloc[0] if "True_template_pKd" in sub_df.columns and not sub_df.empty else float('nan')
        
        abs_err = abs(template_pred_pkd - true_template_pkd_val) if pd.notna(true_template_pkd_val) and pd.notna(template_pred_pkd) else float('nan')

        # Exclude template itself for ranking generated designs, then sort
        gen_sub_df = sub_df[sub_df["design_info"] != "TEMPLATE"].copy()
        # Exclude negative controls from "better than template" and top 10 generated designs
        designs_only_sub_df = gen_sub_df[gen_sub_df["design_info"] != "NEGATIVE_CONTROL"].sort_values("predicted-pKd", ascending=False)
        top10_designs = designs_only_sub_df.head(10)

        # Compare generated designs' pKd to the *predicted* pKd of the template for this run
        better_than_template_count = (designs_only_sub_df["predicted-pKd"] > template_pred_pkd).sum() if pd.notna(template_pred_pkd) else (designs_only_sub_df["predicted-pKd"] > -float('inf')).sum()


        lines.append(f"── {tgt_name_unique} ──")
        lines.append(f"Template predicted pKd: {template_pred_pkd:.4f}" if pd.notna(template_pred_pkd) else "Template predicted pKd: N/A")
        lines.append(f"Template true pKd     : {true_template_pkd_val:.4f}" if pd.notna(true_template_pkd_val) else "Template true pKd     : N/A")
        lines.append(f"|error| (pred-vs-true): {abs_err:.4f}" if pd.notna(abs_err) else "|error| (pred-vs-true): N/A")
        lines.append(f"Designs better than template (vs predicted): {better_than_template_count} / {len(designs_only_sub_df)}") # Total is num_samples (after dedup), not total w/ controls
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


if __name__ == "__main__":
    main() 