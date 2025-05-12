import argparse
import os
import subprocess
import pandas as pd
from datetime import datetime
from pathlib import Path

from evaluation.binder_info import BINDING_INFO


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
    parser.add_argument("--step_divisor", type=int, default=100, help='Step divisor')
    return parser.parse_args()


def main() -> None:
    args = arg_parser()
    
    all_target_results = []
    
    # Create permanent directory for storing individual target results
    output_dir = Path('evaluation/generations')
    output_dir.mkdir(parents=True, exist_ok=True)
    
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
            
        print(f"\n=== Generating designs for {target_name} ===")
        
        # Create output file path for this target
        target_output_file = output_dir / f"{target_name}_designs_uncond.csv"
        
        # Build command to run unconditional_binder.py for this target
        cmd = [
            "python", "-m", "evaluation.unconditional_binder",
            "--target", target_name,
            "--num_samples", str(args.num_samples),
            "--batch_size", str(args.batch_size),
            "--api_batch_size", str(args.api_batch_size),
            "--output_file", str(target_output_file),
            "--step_divisor", str(args.step_divisor)
        ]
        
        # Add optional arguments
        if args.token:
            cmd.extend(["--token", args.token])
            
        if args.synthyra_api_key:
            cmd.extend(["--synthyra_api_key", args.synthyra_api_key])
            
        if args.test:
            cmd.append("--test")
        
        # Run the command
        print(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        
        # Check if output file was created and read it
        if target_output_file.exists():
            target_df = pd.read_csv(target_output_file)
            
            # Add target information
            target_df["Target"] = target_name
            target_df["True_template_pKd"] = true_pkd
            
            all_target_results.append(target_df)
            print(f"Completed processing for {target_name} with {len(target_df)} results")
        else:
            print(f"Warning: No output file generated for {target_name}")

    # Combine all results
    if all_target_results:
        big_df = pd.concat(all_target_results, ignore_index=True)
        output_path = output_dir / args.output_file
        big_df.to_csv(output_path, index=False)
        print(f"\nSaved complete results to {output_path} ({len(big_df)} rows)")
    else:
        print("No results generated")
        return

    # Create summary file with timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    lines = [
        f"Multi-target UNCONDITIONAL design run  -  {datetime.now():%Y-%m-%d %H:%M UTC}",
        f"num_samples per template: {args.num_samples}",
        f"",
    ]
    
    for tgt in big_df["Target"].unique():
        sub = big_df[big_df["Target"] == tgt].copy()

        # Check if template exists in the results
        template_row = sub[sub["design_info"] == "TEMPLATE"]
        template_pred_pkd = template_row["predicted-pKd"].values[0] if not template_row.empty else float('-inf')

        true_template_pkd = sub["True_template_pKd"].iloc[0]
        abs_err = abs(template_pred_pkd - true_template_pkd) if pd.notna(true_template_pkd) and pd.notna(template_pred_pkd) and template_pred_pkd != float('-inf') else float('nan')

        # exclude template itself, then sort
        gen_sub = sub[sub["design_info"] != "TEMPLATE"].copy()
        designs_only_sub = gen_sub[gen_sub["design_info"] != "NEGATIVE_CONTROL"].sort_values("predicted-pKd", ascending=False)
        top10 = designs_only_sub.head(10)

        better_cnt = (designs_only_sub["predicted-pKd"] > template_pred_pkd).sum() if template_pred_pkd != float('-inf') else len(designs_only_sub)

        lines.append(f"── {tgt} ──")
        lines.append(f"Template predicted pKd: {template_pred_pkd:.4f}" if template_pred_pkd != float('-inf') else "Template predicted pKd: N/A (not predicted)")
        lines.append(f"Template true pKd     : {true_template_pkd:.4f}" if pd.notna(true_template_pkd) else "Template true pKd     : N/A")
        lines.append(f"|error| (pred-vs-true): {abs_err:.4f}" if pd.notna(abs_err) else "|error| (pred-vs-true): N/A")
        lines.append(f"Designs better than template: {better_cnt} / {len(designs_only_sub)}" + (" (all designs, template pKd not available)" if template_pred_pkd == float('-inf') else ""))
        lines.append("Top 10 designs (predicted-pKd / design_info / design):")
        for design, pkd, info in zip(top10["Design"], top10["predicted-pKd"], top10["design_info"]):
            lines.append(f'{pkd:.5f} {info}')
            lines.append(design)
        lines.append("")

    # Save summary to timestamped file
    summary_path = output_dir / f"{timestamp}_{args.summary_file}"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
