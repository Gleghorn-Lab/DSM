import argparse
import random
from datetime import datetime
from typing import List, Tuple, Iterable

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
    for chunk in chunked(designs_info, api_batch_size):
        designs = [d for d, _ in chunk]
        infos = [info for _, info in chunk]
        df = predict_against_target(target_seq, designs, test=test, api_key=api_key)
        mapping = {d: info for d, info in chunk}
        df["design_info"] = df["SeqB"].map(mapping)
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def main() -> None:
    args = arg_parser()
    if args.token:
        login(args.token)

    # -------------------- load diffusion model once ------------------------- #
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ESM_Diff.from_pretrained(MODEL_PATH).to(device).eval()
    tokenizer = model.tokenizer

    all_results = []  # collect dfs

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
        pairs = generate_designs_for_template(
            template,
            args.num_samples,
            args.batch_size,
            tokenizer,
            model,
        )

        # 2. predict affinities (template + designs) - ensure template is included
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
            target_seq,
            pairs_with_template,
            args.api_batch_size,
            args.synthyra_api_key,
            args.test,
        )

        # annotate
        df_target["Target"] = target_name
        df_target["True_template_pKd"] = true_pkd

        all_results.append(df_target)

    if not all_results:
        print("No designs generated for any target.")
        return

    # -------------------------------- outputs ------------------------------- #
    big_df = pd.concat(all_results, ignore_index=True)
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

        # locate template row & template predicted pKd
        template_pred_pkd = sub.loc[sub["design_info"] == "TEMPLATE", "predicted-pKd"].values[0]
        true_template_pkd = sub["True_template_pKd"].iloc[0]
        abs_err = abs(template_pred_pkd - true_template_pkd)

        # exclude template itself, then sort
        gen_sub = sub[sub["design_info"] != "TEMPLATE"].copy() # Keep this to exclude template for "better than template"
        # Exclude negative controls from "better than template" and top 10 generated designs
        designs_only_sub = gen_sub[gen_sub["design_info"] != "NEGATIVE_CONTROL"].sort_values("predicted-pKd", ascending=False)
        top10 = designs_only_sub.head(10)

        better_cnt = (designs_only_sub["predicted-pKd"] > template_pred_pkd).sum()  # <- compare to predicted
        # If you prefer to compare against true pKd instead, replace previous
        # line with:  better_cnt = (designs_only_sub["predicted-pKd"] > true_template_pkd).sum()

        lines.append(f"── {tgt} ──")
        lines.append(f"Template predicted pKd: {template_pred_pkd:.4f}")
        lines.append(f"Template true pKd     : {true_template_pkd:.4f}")
        lines.append(f"|error| (pred-vs-true): {abs_err:.4f}")
        lines.append(f"Designs better than template: {better_cnt} / {args.num_samples}") # num_samples is generated, not total w/ controls
        lines.append("Top 10 designs (predicted-pKd / seq):")
        for seq, pkd in zip(top10["SeqB"], top10["predicted-pKd"]):
            lines.append(f'{pkd:.5f}')
            lines.append(seq)
        lines.append("")

    with open(args.summary_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Saved summary to {args.summary_file}")


if __name__ == "__main__":
    main()
